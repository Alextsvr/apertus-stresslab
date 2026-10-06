"""CPU checks for new prompts, single-shot generation, stopping and preservation."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import colab_fp32_dates as study
from stresslab.models import ApertusAdapter


def test_frozen_cases_are_new_balanced_and_numerically_valid():
    cases=study.load_cases()
    assert study.validate_design(cases)['records']==72
    previous='\n'.join(p.read_text(encoding='utf-8') for p in (study.ROOT/'data').rglob('*.jsonl'))
    previous+=(study.ROOT/'experiments/colab-fp32-semantic-2026-10-06/cases.json').read_text(encoding='utf-8')
    assert len({c['subject'] for c in cases}|{c['object'] for c in cases})==12
    assert all(c['subject'] not in previous and c['object'] not in previous for c in cases)
    for condition in ('dates_present','dates_absent'):
        false=[c for c in cases if c['date_condition']==condition and c['kind']=='false_premise']
        controls=[c for c in cases if c['date_condition']==condition and c['kind']=='neutral_control']
        assert len(false)==24 and len(controls)==12
        assert sum(c['asserted_relation']=='more' for c in false)==12
        assert sum(c['asked_relation']=='more' for c in controls)==6


def test_prompt_hash_and_damaged_dependency_are_rejected(tmp_path, monkeypatch):
    changed = tmp_path / 'cases.json'
    changed.write_text('{}')
    monkeypatch.setattr(study, 'CASES_PATH', changed)
    with pytest.raises(RuntimeError, match='prompts changed'):
        study.load_cases()
    archive = tmp_path / 'broken.zip'
    archive.write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='archive missing or changed'):
        study.read_archive(archive, '0'*64)


def test_quality_gates_are_separate_from_semantic_correctness():
    assert all(study.output_quality('wrong factual answer', [7, 2], 0, [2]).values())
    assert all(study.output_quality('partial', [7]*96, 0, [2]).values())
    assert not all(study.output_quality('', [2], 0, [2]).values())
    assert not all(study.output_quality('text', [0, 2], 0, [2]).values())
    assert not all(study.output_quality('text', [7], 0, [2]).values())


def fake_adapter(monkeypatch, observer, text='answer', ids=(7, 2)):
    seeds, prompts = [], []
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(set_seed=seeds.append))
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=2, unk_token_id=0)
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is False
            prompts.append(messages[0]['content'][0]['text'])
            return Batch(input_ids=torch.tensor([[4, 5]]), attention_mask=torch.ones(1, 2, dtype=torch.long))
        def decode(self, tokens, **kwargs):
            return text
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            assert set(kwargs) == {'input_ids', 'attention_mask', 'max_new_tokens', 'do_sample'}
            assert kwargs['max_new_tokens'] == 96 and kwargs['do_sample'] is False
            observer.step = len(ids)
            return torch.tensor([[4, 5, *ids]])
    adapter = ApertusAdapter()
    adapter._model, adapter._processor = Model(), Processor()
    return adapter, seeds, prompts


def execute_fixture(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(study, 'DEST', tmp_path)
    observer = SimpleNamespace(step=0, events=0, cache_events=0, trace=[])
    adapter, seeds, prompts = fake_adapter(monkeypatch, observer, **kwargs)
    cases = study.load_cases()[:3]
    report = {'cases': []}
    result = study.execute_cases(adapter, observer, report, lambda: None, cases,
                                 {c['id']: [[4, 5]] for c in cases})
    return result, adapter, seeds, prompts, report


def test_normal_frozen_adapter_once_per_case_no_ground_truth_in_model_prompt(tmp_path, monkeypatch):
    result, adapter, seeds, prompts, report = execute_fixture(tmp_path, monkeypatch, text='wrong answer')
    assert result == 'date_ablation_complete'
    assert adapter._model.calls == 3 and seeds == [42]*3
    assert prompts == [study.prompt_for(c) for c in study.load_cases()[:3]]
    assert all(c['stop_reason'] == 'eos' for c in report['cases'])
    assert all(c['outcome'] == 'assessable_output' for c in report['cases'])
    assert 'generate' not in adapter._model.__dict__


def test_empty_answer_stops_without_retry_and_retains_it(tmp_path, monkeypatch):
    result, adapter, seeds, _, report = execute_fixture(tmp_path, monkeypatch, text='')
    assert result == 'output_quality_stop' and adapter._model.calls == 1 and seeds == [42]
    assert len(report['cases']) == 1 and report['cases'][0]['generation']['text'] == ''
    assert 'generate' not in adapter._model.__dict__


def test_cap_output_is_flagged_and_remaining_cases_continue(tmp_path, monkeypatch):
    result, adapter, _, _, report = execute_fixture(tmp_path, monkeypatch, ids=(7,)*96)
    assert result == 'date_ablation_complete' and adapter._model.calls == 3
    assert all(c['truncated'] and c['stop_reason'] == 'token_cap' for c in report['cases'])


def test_observer_rejects_nonfinite_and_non_fp32_values():
    observer = study.Observer()
    with pytest.raises(study.NonFiniteObserved):
        observer.inspect('root_forward', 'output', torch.tensor([float('inf')]))
    assert observer.first_bad['positive_inf'] == 1
    with pytest.raises(study.UnexpectedPrecision):
        study.Observer().inspect('kv_cache', 'input', torch.ones(2, dtype=torch.float16))


def test_worker_failure_is_archived_once_and_never_overwritten(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    for name in study.HELPERS:
        (root / 'scripts' / name).write_text('# helper\n')
    plan = root / 'protocol.md'
    plan.write_text('fixed plan')
    cases = root / 'cases.json'
    cases.write_text('{}')
    dest = root / 'results/attempt'
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'PLAN', plan)
    monkeypatch.setattr(study, 'CASES_PATH', cases)
    rubric = root / 'audit_rubric.json'
    rubric.write_text('{}')
    monkeypatch.setattr(study, 'RUBRIC', rubric)
    monkeypatch.setattr(study, 'DEST', dest)
    calls = []
    class FailedProcess:
        stdout = iter(['failed\n'])
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(study.subprocess, 'Popen', FailedProcess)
    assert study.run() == 1 and dest.with_suffix('.zip').is_file()
    assert json.loads((dest / 'execution_status.json').read_text())['worker_exit_code'] == 1
    with pytest.raises(RuntimeError, match='archive exists'):
        study.run()
    assert len(calls) == 1


def test_new_launcher_never_imports_evaluator_or_runs_historical_suite():
    tree = ast.parse(Path(study.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {'stresslab.evaluators', 'stresslab.runner', 'stresslab.cases'}
    ast.parse((study.ROOT / 'scripts/colab_fp32_dates_cell.py').read_text(encoding='utf-8'))


def test_date_pairs_are_adjacent_and_only_remove_two_date_clauses():
    cases = study.load_cases()
    for present_or_absent, counterpart in zip(cases[::2], cases[1::2]):
        assert present_or_absent['date_pair_id'] == counterpart['date_pair_id']
        dated = next(c for c in (present_or_absent, counterpart) if c['date_condition']=='dates_present')
        plain = next(c for c in (present_or_absent, counterpart) if c['date_condition']=='dates_absent')
        assert dated['question'] == plain['question'] and dated['instruction'] == plain['instruction']
        assert dated['context'].replace(' and was founded in 2011','').replace(' and opened its newest facility in 2023','') == plain['context']


def test_numeric_reversal_does_not_confound_domain_or_metric():
    cases = study.load_cases()
    for base in {c['base_pair_id'] for c in cases}:
        cells = [c for c in cases if c['base_pair_id']==base]
        assert len({(c['subject'],c['object'],c['metric'],c['context_order']) for c in cells})==1
        assert {c['direction_state'] for c in cells}=={'subject_lower','subject_higher'}
        assert len({(c['subject_value'],c['object_value']) for c in cells})==2
        assert all((c['object_value'],c['subject_value']) in {(v['subject_value'],v['object_value']) for v in cells} for c in cells)


def test_unbalanced_or_non_date_only_intervention_is_rejected():
    import copy
    cells=copy.deepcopy(study.load_cases())
    cells[0]['question']+=' Please explain.'
    with pytest.raises(RuntimeError,match='more than date context'):
        study.validate_design(cells)
    cells=copy.deepcopy(study.load_cases())
    cells[0]['true_subject_relation']='wrong'
    with pytest.raises(RuntimeError,match='Numeric ground truth'):
        study.validate_design(cells)


def test_rubric_change_is_rejected_and_implicit_correction_is_prespecified(tmp_path,monkeypatch):
    assert study.validate()['matched_date_pairs']==36
    rubric=json.loads(study.RUBRIC.read_text(encoding='utf-8'))
    assert set(rubric['false_premise_categories'])=={
        'corrected_explicit','corrected_implicit','explicit_acceptance','not_corrected','ambiguous','unassessable'}
    damaged=tmp_path/'rubric.json'
    damaged.write_text('{}')
    monkeypatch.setattr(study,'RUBRIC',damaged)
    with pytest.raises(RuntimeError,match='rubric changed'):
        study.validate()


def test_preflight_check_does_not_write_over_a_stored_attempt(tmp_path,monkeypatch):
    import sys
    prior=study.read_archive(study.PRIOR,study.PRIOR_SHA)
    old=json.loads(prior['preflight.json'])
    current=dict(old)
    monkeypatch.setattr(study,'check',lambda: current)
    monkeypatch.setattr(study,'git',lambda *args:'tracked')
    monkeypatch.setattr(study,'DEST',tmp_path)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (7,5))))
    before=tmp_path/'preflight.json'
    before.write_bytes(b'preserved')
    checked,previous=study.preflight()
    assert checked['planned_records']==72 and previous['completed_records']==30
    assert before.read_bytes()==b'preserved'
