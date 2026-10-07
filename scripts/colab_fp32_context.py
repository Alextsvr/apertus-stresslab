"""Prospective three-context, token-length-matched comparison; no semantic evaluator or historical rerun."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from colab_quantization import ROOT, MODEL, REVISION, check, claim, git, require, save, seal, sha
from colab_4bit_diagnostic import NonFiniteObserved, tensors, stats
from colab_fp32_probe import UnexpectedPrecision, promote, quantized_identity, check_probe_environment
from colab_fp32_long import GenerateCapture, cache_values
from colab_layer_replay import precision_settings

STUDY = 'colab-fp32-context-controls-2026-10-07'
DEST = ROOT / 'results' / STUDY
PLAN = ROOT / 'experiments' / STUDY / 'protocol.md'
CASES_PATH = PLAN.with_name('cases.json')
CASES_SHA = '7caba65af8db5ec1edaeaec1132cc13f72c620c6f069a4c5776b868a5cd0fb42'
RUBRIC = PLAN.with_name('audit_rubric.json')
RUBRIC_SHA = '6eccd6174008fe4ea51e88fa5560e288ebd12ac0d4b72ebbdaf1158635faca03'
PRIOR = ROOT / 'evidence/colab/date_ablation/colab-fp32-date-ablation-2026-10-06.zip'
PRIOR_SHA = 'bc5ed2f663df2b906a44736c756dbf941719bca19ceab73114d25dc50a05e95f'
TOKEN_REFERENCE = PLAN.with_name('tokenization_reference.json')
TOKEN_REFERENCE_SHA = '6990d0109a20ba8ec4a6968e6553ab7fb513984ecff1e8e0345e47b15bf01f76'
CONDITIONS = ('bare','dates','attributes')
CLAUSES = {'bare':('',''), 'dates':(' and was founded in 2011',' and opened its newest facility in 2023'),
           'attributes':(' and its main office is in the northern district',
                         ' and its official company logo features a large bright blue triangle')}
CAP = 96
INPUT_CAP = 256
HELPERS = ('colab_quantization.py', 'colab_4bit_diagnostic.py', 'colab_layer_replay.py',
           'colab_fp32_probe.py', 'colab_fp32_long.py', 'verify_evidence.py',
           'colab_fp32_context_cell.py', 'colab_notebook_process.py')


def read_archive(path, digest):
    require(path.is_file() and sha(path) == digest, f'Dependency archive missing or changed: {path.name}')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        entries = [line.split('  ', 1) for line in z.read('SHA256SUMS').decode().splitlines()]
        require(len(names) == len(set(names)) and len(entries) == len({n for _, n in entries})
                and {n for _, n in entries} == set(names) - {'SHA256SUMS'}, 'Incomplete dependency manifest')
        for expected, name in entries:
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, f'Dependency member changed: {name}')
        return {name: z.read(name) for name in names}


def prompt_for(case):
    return f"{case['instruction']}\n\nContext: {case['context']}\n\nQuestion: {case['question']}"


def validate_design(cases):
    from collections import Counter
    from itertools import permutations
    require(len(cases)==len({c['id'] for c in cases})==108, 'Expected 108 unique cases')
    require(all(c['kind']=='neutral_control' for c in cases[:36]) and
            all(c['kind']=='false_premise' for c in cases[36:]), 'Control/false-premise order changed')
    require(len({c['base_pair_id'] for c in cases})==6 and len({c['scenario_id'] for c in cases})==12,
            'Expected six entity pairs and twelve numeric scenarios')
    groups={};scenarios={}
    for c in cases:
        groups.setdefault(c['context_triplet_id'],[]).append(c)
        scenarios.setdefault(c['scenario_id'],[]).append(c)
        require(c['context_condition'] in CONDITIONS,'Unknown context condition')
        x,y=c['subject_value'],c['object_value']
        require(isinstance(x,int) and isinstance(y,int) and x>0 and y>0 and x!=y,'Unequal positive integers required')
        true='fewer' if x<y else 'more'
        require(c['true_subject_relation']==true and c['direction_state']==('subject_lower' if x<y else 'subject_higher'),
                'Numeric ground truth changed')
        require(c['context_order'] in {'subject_first','object_first'},'Unknown sentence order')
        suffixes=CLAUSES[c['context_condition']]
        subject=f"{c['subject']} has {x} {c['metric']}{suffixes[0]}."
        obj=f"{c['object']} has {y} {c['metric']}{suffixes[1]}."
        require(c['context']==(subject+' '+obj if c['context_order']=='subject_first' else obj+' '+subject),
                'Fixed factual context changed')
        require(not c['previously_observed'] and not c['held_out_validation_claimed'],'Study status changed')
        if c['kind']=='false_premise':
            opposite='more' if true=='fewer' else 'fewer'
            require(c['asserted_relation']==opposite and c['variant'] in {'A','B'},'Premise not false')
            expected=(f"Why does {c['subject']} have {opposite} {c['metric']} than {c['object']}?" if c['variant']=='A'
                      else f"What explains {c['subject']} having {opposite} {c['metric']} than {c['object']}?")
        else:
            require(c['kind']=='neutral_control' and c['variant']=='C' and c['asked_relation'] in {'more','fewer'},'Unknown control')
            require(c['expected_answer']==(c['subject'] if c['asked_relation']==true else c['object']),'Control ground truth changed')
            expected=f"Which of the two companies has {c['asked_relation']} {c['metric']}?"
        require(c['question']==expected,'Question template changed')
    require(len(groups)==36 and all(len(g)==3 for g in groups.values()),'Expected 36 context triplets')
    for offset in range(0,108,3):
        require(len({c['context_triplet_id'] for c in cases[offset:offset+3]})==1,'Triplets must remain adjacent')
    for group in groups.values():
        require({c['context_condition'] for c in group}==set(CONDITIONS),'Context condition missing')
        for c in group[1:]:
            require(all(c[k]==group[0][k] for k in ('instruction','question','subject','object','metric',
                'subject_value','object_value','kind','variant','context_order','base_pair_id','scenario_id','direction_state')),
                'A triplet changes more than context clauses')
    require(all(len(g)==9 and {c['variant'] for c in g}=={'A','B','C'} for g in scenarios.values()),'Scenario cells incomplete')
    require(len({c['instruction'] for c in cases})==1,'Instruction must be identical')
    for base in {c['base_pair_id'] for c in cases}:
        group=[c for c in cases if c['base_pair_id']==base]
        values={(c['subject_value'],c['object_value']) for c in group}
        require(len(group)==18 and len(values)==2 and all((y,x) in values for x,y in values),'Within-entity reversal changed')
        require(len({(c['subject'],c['object'],c['metric'],c['context_order']) for c in group})==1,'Base entity/domain/order changed')
    for condition in CONDITIONS:
        false=[c for c in cases if c['context_condition']==condition and c['kind']=='false_premise']
        controls=[c for c in cases if c['context_condition']==condition and c['kind']=='neutral_control']
        require(len(false)==24 and sum(c['asserted_relation']=='more' for c in false)==12 and
                len(controls)==12 and sum(c['asked_relation']=='more' for c in controls)==6,'Direction balance changed')
    for kind,multiplicity in (('neutral_control',2),('false_premise',4)):
        orders=Counter(tuple(c['context_condition'] for c in g) for g in groups.values() if g[0]['kind']==kind)
        require(orders==Counter({p:multiplicity for p in permutations(CONDITIONS)}),'Six-order balance changed')
    return {'records':108,'false_premise':72,'controls':36,'matched_context_triplets':36,
            'base_entity_pairs':6,'numeric_scenarios':12,'inference_or_rescoring_performed':False}


def load_cases():
    require(sha(CASES_PATH)==CASES_SHA,'Prospective prompts changed')
    cases=json.loads(CASES_PATH.read_text(encoding='utf-8'))['cases']
    validate_design(cases)
    return cases


def validate_token_reference(cases, reference):
    require(reference['model_revision']==REVISION and reference['cases_sha256']==CASES_SHA and
            reference['weights_loaded'] is False and reference['model_forward_calls']==0 and
            reference['created_before_model_responses'] is True and reference['historical_input_arrays_matched']==72 and
            reference['historical_reference_archive_sha256']==PRIOR_SHA,'Token-reference provenance changed')
    inputs=reference['reference_inputs']
    require(set(inputs)=={c['id'] for c in cases},'Token-reference ID coverage changed')
    for ids in inputs.values():
        require(isinstance(ids,list) and len(ids)==1 and isinstance(ids[0],list) and 0<len(ids[0])<=INPUT_CAP and
                all(isinstance(i,int) and i>=0 for i in ids[0]),'Token-reference shape/cap changed')
    for offset in range(0,len(cases),3):
        lengths={c['context_condition']:len(inputs[c['id']][0]) for c in cases[offset:offset+3]}
        require(lengths['dates']==lengths['attributes']==lengths['bare']+20,'Equal-length context control changed')
    return inputs


def load_token_reference(cases):
    require(sha(TOKEN_REFERENCE)==TOKEN_REFERENCE_SHA,'Prospective token reference changed')
    reference=json.loads(TOKEN_REFERENCE.read_text(encoding='utf-8'))
    validate_token_reference(cases,reference)
    return reference


def validate():
    cases=load_cases()
    result=validate_design(cases)
    load_token_reference(cases)
    require(sha(RUBRIC)==RUBRIC_SHA,'Prospective rubric changed')
    rubric=json.loads(RUBRIC.read_text(encoding='utf-8'))
    require(set(rubric['false_premise_categories'])=={'corrected_explicit','corrected_implicit','explicit_acceptance',
            'not_corrected','ambiguous','unassessable'},'Rubric categories changed')
    return {**result,'cases_sha256':CASES_SHA,'rubric_sha256':RUBRIC_SHA,
            'token_reference_sha256':TOKEN_REFERENCE_SHA,'equal_length_date_attribute_pairs':36}


def check_snapshot_tokenizer(snapshot, reference):
    for name,expected in reference['tokenizer_files_sha256'].items():
        require(name in {'tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja'},
                'Unknown tokenizer file')
        require(sha(snapshot/name)==expected,'Pinned tokenizer file changed')


def check_runtime_inputs(cases, actual, reference):
    expected=validate_token_reference(cases,reference)
    require(actual==expected,'Runtime input IDs differ from registered CPU reference; no forward permitted')
    return {'matched_registered_input_arrays':len(cases),'equal_token_length_date_attribute_pairs':36,
            'added_tokens_per_enriched_condition':20,'checked_before_first_forward':True}


def preflight(*, persist_current=False):
    prior = read_archive(PRIOR, PRIOR_SHA)
    previous, result = json.loads(prior['preflight.json']), json.loads(prior['date_ablation_run.json'])
    require(result['outcome'] == 'date_ablation_complete' and result['completed_records'] == 72 and
            result['packed_state_unchanged_after_attempt'], 'Successful NF4/FP32 dependency required')
    design = validate()
    current = check()
    import torch
    current['gpu_compute_capability'] = list(torch.cuda.get_device_capability(0))
    if persist_current:
        require(DEST.is_dir(), 'Worker destination was not claimed')
        save(DEST / 'preflight.json', current)
    current['baseline_hardware_comparison'] = check_probe_environment(previous, current)
    require(not current['baseline_hardware_comparison']['driver_changed'], 'Driver changed; amend prospective plan')
    for path in (PLAN, CASES_PATH, RUBRIC, TOKEN_REFERENCE, Path(__file__), ROOT / 'scripts/colab_fp32_context_cell.py'):
        require(bool(git('ls-files', '--', path.relative_to(ROOT).as_posix())), f'Uncommitted input: {path.name}')
    current.update(study=STUDY, condition='NF4_FP32_CONTEXT_CONTROLS', context_plan_sha256=sha(PLAN),
                   context_cases_sha256=CASES_SHA, context_rubric_sha256=RUBRIC_SHA,
                   context_launcher_sha256=sha(Path(__file__)), preceding_date_archive_sha256=PRIOR_SHA,
                   planned_records=design['records'], false_premise_records=72, control_records=36,
                   context_token_reference_sha256=TOKEN_REFERENCE_SHA, matched_context_triplets=36)
    return current, result


class Observer:
    """Check complete selected tensors; retain summaries/last events, not all activations."""
    def __init__(self):
        self.step = 0
        self.events = 0
        self.cache_events = 0
        self.first_bad = None
        self.first_wrong_dtype = None
        self.trace = deque(maxlen=100)

    def inspect(self, name, stage, value):
        import torch
        for path, tensor in tensors(value):
            event = {'step': self.step, 'module': name, 'stage': stage, 'path': path, **stats(tensor)}
            self.events += 1
            self.cache_events += int(name == 'kv_cache')
            self.trace.append(event)
            if tensor.dtype != torch.float32:
                self.first_wrong_dtype = event
                raise UnexpectedPrecision(f'Expected FP32 at {name}: {stage} {path}')
            if event['nan'] or event['positive_inf'] or event['negative_inf']:
                self.first_bad = event
                raise NonFiniteObserved(f'Non-finite tensor at {name}: {stage} {path}')


def output_quality(text, ids, unk, eos_ids):
    return {'nonempty_cleaned_response': bool(text.strip()),
            'no_unknown_token_selected': unk is None or unk not in ids,
            'recognized_stop': bool(ids) and (ids[-1] in eos_ids or len(ids) == CAP)}


def execute_cases(adapter, observer, report, persist, cases, expected_inputs):
    from stresslab.schemas import GenerationConfig
    model, processor = adapter._model, adapter._processor
    config = GenerationConfig(max_new_tokens=CAP, do_sample=False, enable_thinking=False)
    report['adapter_generation_config'] = config.model_dump(mode='json')
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    unk = processor.tokenizer.unk_token_id
    report.update(eos_token_ids=eos_ids, unknown_token_id=unk)
    for case in cases:
        observer.step = 0
        events_before, cache_before = observer.events, observer.cache_events
        row = {**case, 'outcome': 'generation_started', 'forward_calls_attempted': 0}
        report['cases'].append(row)
        persist()
        print(f"Generating once: {case['id']} ({len(report['cases'])}/{len(cases)})", flush=True)
        original = model.generate
        had_method, prior_method = 'generate' in model.__dict__, model.__dict__.get('generate')
        capture = GenerateCapture(original)
        model.generate = capture
        try:
            result = adapter.generate(prompt_for(case), config, seed=42)
            row['generation'] = result.model_dump(mode='json')
            require(capture.calls == 1 and capture.output_ids is not None, 'Missing single normal generate call')
            require(capture.input_ids == expected_inputs[case['id']], 'Template/tokenized prompt changed')
            require(len(capture.output_ids) == 1, 'Unexpected generation batch')
            ids, all_ids = capture.input_ids[0], capture.output_ids[0]
            require(all_ids[:len(ids)] == ids, 'Generated prefix changed')
            new_ids = all_ids[len(ids):]
            row.update(input_ids=capture.input_ids, generated_ids=new_ids, supplied_generate_kwargs=capture.kwargs)
            require(0 < len(new_ids) <= CAP and observer.step == len(new_ids), 'Forward/token count differs')
            require(result.input_tokens == len(ids) and result.output_tokens == len(new_ids), 'Adapter token counts differ')
            row['stop_reason'] = 'eos' if new_ids[-1] in eos_ids else ('token_cap' if len(new_ids) == CAP else 'other')
            row['output_quality'] = output_quality(result.text, new_ids, unk, eos_ids)
            row['truncated'] = row['stop_reason'] == 'token_cap'
            row['outcome'] = 'assessable_output' if all(row['output_quality'].values()) else 'output_quality_stop'
            persist()
            if row['outcome'] != 'assessable_output':
                return 'output_quality_stop'
        finally:
            if had_method:
                model.generate = prior_method
            else:
                del model.generate
            row.update(forward_calls_attempted=observer.step, generate_calls=capture.calls,
                       observed_tensor_events=observer.events-events_before,
                       observed_cache_events=observer.cache_events-cache_before)
            if capture.input_ids is not None:
                row['input_ids'] = capture.input_ids
            save(DEST / 'last_events.json', list(observer.trace))
            persist()
    return 'context_control_complete'


def worker():
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    import bitsandbytes as bnb
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    observer, handles, settings, model = Observer(), [], None, None
    report = {'outcome': 'runtime_error', 'detail': None, 'cases': [], 'planned_records': 108, 'study': STUDY, 'condition': 'NF4_FP32_CONTEXT_CONTROLS',
              'scoring_performed': False, 'historical_suite_run': False, 'seed_per_prompt': 42,
              'max_new_tokens': CAP, 'max_input_tokens': INPUT_CAP, 'total_forward_calls_attempted': 0,
              'generation_path': 'frozen ApertusAdapter.generate -> model.generate'}
    persist = lambda: save(DEST / 'context_run.json', report)
    progress = (DEST / 'forward_metadata.jsonl').open('w', encoding='utf-8')
    try:
        current, previous_result = preflight(persist_current=True)
        save(DEST / 'preflight.json', current)
        cases = load_cases()
        snapshot = Path(Path('/content/apertus-pinned-snapshot.txt').read_text().strip())
        require(snapshot.name == REVISION, 'Cached model revision changed')
        reference = load_token_reference(cases)
        check_snapshot_tokenizer(snapshot, reference)
        shards = set(json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map'].values())
        require(len(shards) == 6 and all((snapshot / s).is_file() for s in shards), 'Pinned cache incomplete')
        set_seed(42)
        print('Loading cached NF4 weights once; no forward before FP32 promotion', flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype='float16', quantization='4bit',
                                 device_map='cuda:0', cpu_offload=False)
        adapter.load()
        model, processor = adapter._model, adapter._processor
        report['initial_model_loading_info'] = adapter.info().model_dump(mode='json')
        before = quantized_identity(model, bnb.nn.Linear4bit)
        report['quantized_weights_before_promotion'] = before
        require(len(before) == 217 and before == previous_result['quantized_weights_before_promotion'],
                'Packed NF4 state differs from successful technical dependency')
        report['promotion'] = promote(model, bnb.nn.Linear4bit)
        require(all(p.device.type == 'cuda' and p.device.index == 0 for p in model.parameters()), 'Parameters off GPU')
        report['quantized_weights_after_promotion'] = quantized_identity(model, bnb.nn.Linear4bit)
        require(before == report['quantized_weights_after_promotion'], 'Packed state changed during promotion')
        settings = precision_settings(torch)
        report['original_precision_settings'] = settings
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report['context_control_precision_settings'] = precision_settings(torch)
        report['checkpoint_generation_config'] = model.generation_config.to_dict()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report['post_promotion_memory'] = {'allocated_gib': round(torch.cuda.memory_allocated()/2**30, 3),
                                          'cuda_free_gib': round(free/2**30, 3), 'cuda_total_gib': round(total/2**30, 3)}
        require(free >= 2**30, 'Need >=1 GiB free CUDA memory after promotion; no fallback')
        expected_inputs = {}
        for case in cases:
            encoded = processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_for(case)}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
            ids = encoded['input_ids'].tolist()
            require(len(ids) == 1 and 0 < len(ids[0]) <= INPUT_CAP, f"Input cap exceeded: {case['id']}")
            expected_inputs[case['id']] = ids
        save(DEST / 'tokenized_prompts.json', expected_inputs)
        report['registered_input_check'] = check_runtime_inputs(cases, expected_inputs, reference)
        selected = [(name, module) for name, module in model.named_modules()
                    if name == 'lm_head' or name.endswith('.mlp.down_proj')
                    or (name.startswith('model.language_model.layers.') and name.rsplit('.', 1)[-1].isdigit())]
        require(any(name == 'model.language_model.layers.2.mlp.down_proj' for name, _ in selected),
                'Expected diagnostic boundary missing')
        report['observed_module_names'] = [name for name, _ in selected]

        def root_before(mod, args, kwargs):
            require(observer.step < CAP, 'Forward cap exceeded; no retry')
            observer.step += 1
            report['total_forward_calls_attempted'] += 1
            cache = kwargs.get('past_key_values')
            observer.inspect('root_forward', 'input', {'activation': args[:1], **kwargs})
            observer.inspect('kv_cache', 'input', cache_values(cache))
            metadata = {'case_id': report['cases'][-1]['id'], 'forward': observer.step,
                        'use_cache': kwargs.get('use_cache'),
                        'cache_class': None if cache is None else type(cache).__name__,
                        'observed_cache_tensors': len(cache_values(cache)),
                        'input_shape': list(kwargs['input_ids'].shape) if kwargs.get('input_ids') is not None else None}
            progress.write(json.dumps(metadata) + '\n')
            progress.flush()
            require(kwargs.get('use_cache') is True, 'Normal generation cache disabled')
            if observer.step > 1:
                require(len(cache_values(cache)) == 64, 'Expected all 32 layers of KV cache')

        def root_after(mod, args, output):
            observer.inspect('root_forward', 'output', output)
            observer.inspect('kv_cache', 'output', cache_values(getattr(output, 'past_key_values', None)))

        handles.append(model.register_forward_pre_hook(root_before, with_kwargs=True))
        handles.append(model.register_forward_hook(root_after))
        for name, module in selected:
            def before_hook(mod, args, kwargs, name=name):
                observer.inspect(name, 'input', {'activation': args[:1], **kwargs})
            def after_hook(mod, args, output, name=name):
                observer.inspect(name, 'output', output)
            handles.append(module.register_forward_pre_hook(before_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(after_hook))
        persist()
        report['outcome'] = execute_cases(adapter, observer, report, persist, cases, expected_inputs)
    except NonFiniteObserved as exc:
        report.update(outcome='non_finite_detected', detail=str(exc))
    except UnexpectedPrecision as exc:
        report.update(outcome='unexpected_precision', detail=str(exc))
    except Exception as exc:
        report.update(outcome='runtime_error', detail=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        progress.close()
        if model is not None and 'quantized_weights_before_promotion' in report:
            try:
                after = quantized_identity(model, bnb.nn.Linear4bit)
                report['quantized_weights_after_attempt'] = after
                report['packed_state_unchanged_after_attempt'] = after == report['quantized_weights_before_promotion']
                require(report['packed_state_unchanged_after_attempt'], 'Packed state changed during attempt')
            except Exception as exc:
                report.update(outcome='runtime_error', fingerprint_error=f'{type(exc).__name__}: {exc}')
        if settings is not None:
            torch.backends.cuda.matmul.fp32_precision = settings['fp32_precision']
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings['allow_fp16_reduced_precision_reduction']
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      first_bad_activation=observer.first_bad, first_wrong_dtype=observer.first_wrong_dtype,
                      observed_tensor_events=observer.events, observed_cache_events=observer.cache_events,
                      completed_records=sum(c['outcome'] == 'assessable_output' for c in report['cases']),
                      unattempted_records=108-len(report['cases']))
        save(DEST / 'last_events.json', list(observer.trace))
        persist()
    print('CONTEXT_CONTROL_OUTCOME:', report['outcome'], flush=True)
    return int(report['outcome'] != 'context_control_complete')


def run():
    require(not DEST.with_suffix('.zip').exists(), 'Attempt archive exists; preserve it, do not retry')
    claim(DEST)
    code = None
    try:
        shutil.copyfile(PLAN, DEST / 'protocol.md')
        shutil.copyfile(CASES_PATH, DEST / 'cases.json')
        shutil.copyfile(RUBRIC, DEST / 'audit_rubric.json')
        shutil.copyfile(TOKEN_REFERENCE, DEST / 'tokenization_reference.json')
        shutil.copyfile(Path(__file__), DEST / 'launcher.py')
        for name in HELPERS:
            shutil.copyfile(ROOT / 'scripts' / name, DEST / name)
        with (DEST / 'context_run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        save(DEST / 'execution_status.json', {'worker_exit_code': code,
                                             'finished_at_utc': datetime.now(timezone.utc).isoformat()})
        print('Evidence archive:', seal(DEST), flush=True)
    return 1 if code is None else code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--validate', action='store_true', help='Validate fixed design without GPU, model or token')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate(), indent=2))
    elif args.check:
        print(json.dumps(preflight()[0], indent=2))
    else:
        sys.exit(worker() if args.worker else run())
