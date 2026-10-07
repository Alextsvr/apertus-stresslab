"""CPU-only contracts for the prospective cross-model run; never load model weights."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import colab_qwen_context as study
import colab_qwen_adapter as qwen


def test_exact_apertus_cases_rubric_and_cpu_token_registration():
    assert study.validate()['records']==108
    assert study.CASES_PATH.parent.name=='colab-fp32-context-controls-2026-10-07'
    cases=study.load_cases()
    ref=study.load_token_reference(cases)
    assert len(ref['reference_inputs'])==108 and len(ref['triplet_lengths'])==36
    assert ref['model_forward_calls']==0 and not ref['model_weights_loaded']
    lengths=[len(a[0]) for a in ref['reference_inputs'].values()]
    assert (min(lengths),max(lengths))==(98,129)
    assert all(t['dates']==t['attributes']==t['bare']+20 for t in ref['triplet_lengths'])
    assert all(hashlib.sha256(study.prompt_for(c).encode()).hexdigest()==ref['user_text_sha256'][c['id']] for c in cases)


@pytest.mark.parametrize('field',['revision','cases_sha256','rubric_sha256','registered_before_model_responses','model_forward_calls'])
def test_reference_provenance_change_rejected(field):
    cases=study.load_cases(); ref=copy.deepcopy(study.load_token_reference(cases))
    ref[field]='changed'
    with pytest.raises(RuntimeError):study.validate_token_reference(cases,ref)


def test_runtime_token_mismatch_stops_before_any_generation():
    cases=study.load_cases();ref=study.load_token_reference(cases)
    actual=copy.deepcopy(ref['reference_inputs']);actual[cases[0]['id']][0][0]+=1
    with pytest.raises(RuntimeError,match='no forward permitted'):study.check_runtime_inputs(cases,actual,ref)


def test_capture_checks_inputs_and_decoding_before_delegation():
    calls=[]
    original=lambda **kwargs:calls.append(kwargs)
    capture=qwen.GenerateCapture(original,[[1,2]])
    with pytest.raises(RuntimeError,match='inputs changed'):
        capture(input_ids=torch.tensor([[3,4]]),**qwen.OVERRIDES)
    assert not calls
    capture=qwen.GenerateCapture(original,[[1,2]])
    with pytest.raises(RuntimeError,match='decoding overrides'):
        capture(input_ids=torch.tensor([[1,2]]),max_new_tokens=96,do_sample=False,repetition_penalty=1.05)
    assert not calls


def test_qwen_checkpoint_penalty_override_matches_effective_apertus_default():
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    model=SimpleNamespace(config=SimpleNamespace(_get_generation_parameters=lambda:{}),
                          generation_config=GenerationConfig(repetition_penalty=None))
    aperture,_=GenerationMixin._prepare_generation_config(model,None,max_new_tokens=96,do_sample=False)
    model.generation_config=GenerationConfig(repetition_penalty=1.05,do_sample=True,
                                             eos_token_id=[151645,151643],pad_token_id=151643)
    effective,_=GenerationMixin._prepare_generation_config(model,None,**qwen.OVERRIDES)
    assert aperture.repetition_penalty==effective.repetition_penalty==1.0
    assert aperture.use_cache==effective.use_cache is True
    assert aperture.num_beams==effective.num_beams==1
    assert effective.do_sample is False and effective.max_new_tokens==96


def fixture(tmp_path,monkeypatch,text='answer',ids=(7,2)):
    monkeypatch.setattr(study,'DEST',tmp_path)
    observer=SimpleNamespace(step=0,events=0,cache_events=0,trace=[])
    seeds=[];messages=[]
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(set_seed=seeds.append))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        eos_token_id=2;unk_token_id=None
        def apply_chat_template(self,chat,**kwargs):
            assert kwargs=={'add_generation_prompt':True,'tokenize':True,'return_dict':True,'return_tensors':'pt'}
            messages.append(chat)
            return Batch(input_ids=torch.tensor([[4,5]]),attention_mask=torch.ones(1,2,dtype=torch.long))
        def decode(self,tokens,**kwargs):return text
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.device='cpu';self.calls=0
            self.generation_config=SimpleNamespace(eos_token_id=2)
        def generate(self,**kwargs):
            self.calls+=1;observer.step=len(ids)
            assert {k:v for k,v in kwargs.items() if k not in {'input_ids','attention_mask'}}==qwen.OVERRIDES
            return torch.tensor([[4,5,*ids]])
    adapter=qwen.QwenAdapter('unused');adapter._model=Model()
    adapter._processor=SimpleNamespace(tokenizer=Tokenizer())
    cases=study.load_cases()[:3];report={'cases':[]}
    outcome=study.execute_cases(adapter,observer,report,lambda:None,cases,{c['id']:[[4,5]] for c in cases})
    return outcome,adapter,report,seeds,messages,cases


def test_once_per_row_native_template_exact_user_text_and_no_ground_truth(tmp_path,monkeypatch):
    outcome,adapter,report,seeds,messages,cases=fixture(tmp_path,monkeypatch,text='factually wrong')
    assert outcome=='qwen_context_complete' and adapter._model.calls==3 and seeds==[42]*3
    assert messages==[[{'role':'user','content':study.prompt_for(c)}] for c in cases]
    assert all(c['outcome']=='assessable_output' for c in report['cases'])
    assert 'generate' not in adapter._model.__dict__


def test_empty_response_retained_and_stops_without_retry(tmp_path,monkeypatch):
    outcome,adapter,report,seeds,_,_=fixture(tmp_path,monkeypatch,text='')
    assert outcome=='output_quality_stop' and adapter._model.calls==1 and seeds==[42]
    assert report['cases'][0]['generation']['text']==''


def test_cap_retained_and_continues(tmp_path,monkeypatch):
    outcome,adapter,report,_,_,_=fixture(tmp_path,monkeypatch,ids=(7,)*96)
    assert outcome=='qwen_context_complete' and adapter._model.calls==3
    assert all(c['truncated'] and c['stop_reason']=='token_cap' for c in report['cases'])


def test_nonfinite_and_wrong_dtype_stop():
    with pytest.raises(study.NonFiniteObserved):study.Observer().inspect('lm_head','output',torch.tensor([float('inf')]))
    with pytest.raises(study.UnexpectedPrecision):study.Observer().inspect('kv_cache','input',torch.ones(2,dtype=torch.float16))


def test_worker_failure_sealed_without_overwrite(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'scripts').mkdir(parents=True)
    for name in study.HELPERS:(root/'scripts'/name).write_text('# helper')
    for name in ('protocol.md','model_manifest.json','cases.json','audit_rubric.json','tokenization_reference.json'):
        (root/name).write_text('{}')
    for attr,name in (('PLAN','protocol.md'),('CASES_PATH','cases.json'),('RUBRIC','audit_rubric.json'),('TOKEN_REFERENCE','tokenization_reference.json')):
        monkeypatch.setattr(study,attr,root/name)
    monkeypatch.setattr(study,'ROOT',root);dest=root/'results/attempt';monkeypatch.setattr(study,'DEST',dest)
    calls=[]
    class Failed:
        stdout=iter(['failure\n'])
        def __init__(self,*args,**kwargs):calls.append(args)
        def poll(self):return 1
        def wait(self):return 1
    monkeypatch.setattr(study.subprocess,'Popen',Failed)
    assert study.run()==1 and dest.with_suffix('.zip').is_file()
    with pytest.raises(RuntimeError,match='archive exists'):study.run()
    assert len(calls)==1


def test_no_historical_evaluator_and_cell_does_not_request_secret():
    for name in ('colab_qwen_context.py','colab_qwen_adapter.py','colab_qwen_context_cell.py'):
        text=(study.ROOT/'scripts'/name).read_text(encoding='utf-8');tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert node.module not in {'stresslab.evaluators','stresslab.runner'}
    cell=(study.ROOT/'scripts/colab_qwen_context_cell.py').read_text(encoding='utf-8')
    assert 'userdata' not in cell and 'token=False' in cell
