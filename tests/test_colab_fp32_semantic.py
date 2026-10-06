"""CPU checks for new prompts, single-shot generation, stopping and preservation."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import colab_fp32_semantic as study
from stresslab.models import ApertusAdapter


def test_frozen_cases_have_new_entities_true_ground_truth_and_balanced_pairs():
    cases = study.load_cases()
    previous = '\n'.join(p.read_text(encoding='utf-8') for p in (study.ROOT / 'data').rglob('*.jsonl'))
    controls, false = cases[:6], cases[6:]
    assert [c['asked_relation'] for c in controls].count('more') == 3
    assert [c['asserted_relation'] for c in false].count('more') == 12
    assert len({c['subject'] for c in false} | {c['object'] for c in false}) == 24
    for i in range(0, 24, 2):
        a, b = false[i:i+2]
        for field in ('context', 'subject_value', 'object_value', 'true_subject_relation', 'asserted_relation'):
            assert a[field] == b[field]
        assert (a['subject_value'] < a['object_value']) == (a['asserted_relation'] == 'more')
    for case in cases:
        assert case['subject'] not in previous and case['object'] not in previous
        assert case['context'].count(case['metric']) == 2
        assert not case['held_out_validation_claimed']
        assert len({c['instruction'] for c in cases}) == 1
    for c in controls:
        expected = c['subject'] if (c['subject_value'] > c['object_value']) == (c['asked_relation'] == 'more') else c['object']
        assert c['expected_answer'] == expected


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
    assert result == 'semantic_generation_complete'
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
    assert result == 'semantic_generation_complete' and adapter._model.calls == 3
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
    ast.parse((study.ROOT / 'scripts/colab_fp32_semantic_cell.py').read_text(encoding='utf-8'))
