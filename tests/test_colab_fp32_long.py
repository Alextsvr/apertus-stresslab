"""CPU checks for actual token-horizon coverage and unforced normal generation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_fp32_long as probe
from stresslab.models import ApertusAdapter


def test_full_target_prefix_allows_final_partial_number_but_not_wrong_sequence():
    expected = probe.CASES[0]["expected"]
    ids = list(range(10, 106))
    assert all(probe.quality("1 2 3 4 5", ids, 0, [2], expected).values())
    partial = " ".join(str(i) for i in range(1, 41)) + " 4"
    assert probe.quality(partial, ids, 0, [2], expected)["requested_sequence_prefix"]
    assert not probe.quality("1 2 4", ids, 0, [2], expected)["requested_sequence_prefix"]
    assert not probe.quality("1 2 3.", ids, 0, [2], expected)["requested_sequence_prefix"]


def test_token_cap_setting_alone_does_not_establish_actual_96_token_horizon():
    checks = probe.quality("1 2", [10, 2], 0, [2], probe.CASES[0]["expected"])
    assert checks["requested_sequence_prefix"] and not checks["exactly_96_new_tokens"]
    assert not checks["no_eos_selected"]
    assert not probe.quality("1", [0]*96, 0, [2], probe.CASES[0]["expected"])["no_unknown_token_selected"]


def setup_adapter(observer, report, monkeypatch, count=96, eos=False, cache=True, text="1 2 3"):
    seeds = []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=seeds.append))
    class Batch(dict):
        def to(self, device):
            assert str(device) == "cpu"
            return self
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=2, unk_token_id=0, encode=lambda text, **kwargs: list(range(400)))
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["text"] == probe.CASES[0]["prompt"]
            assert kwargs["enable_thinking"] is False
            return Batch(input_ids=torch.tensor([[1, 7]]), attention_mask=torch.ones(1, 2, dtype=torch.long))
        def decode(self, ids, skip_special_tokens):
            return text if skip_special_tokens or not eos else text + "<eos>"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            assert set(kwargs) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample"}
            assert kwargs["max_new_tokens"] == 96 and kwargs["do_sample"] is False
            for step in range(1, count+1):
                observer.step = step
                report["forward_metadata"].append({"case_id": "TECH-LONG-01", "forward": step,
                     "use_cache": cache, "observed_cache_tensors": 64 if cache and step > 1 else 0})
            tokens = list(range(10, 10+count))
            if eos:
                tokens[-1] = 2
            return torch.tensor([[1, 7, *tokens]])
    adapter = ApertusAdapter()
    adapter._model, adapter._processor = Model(), Processor()
    return adapter, seeds


def test_actual_frozen_adapter_96_tokens_without_forcing_eos_or_min_tokens(monkeypatch):
    observer, report = SimpleNamespace(step=0), {"cases": [], "forward_metadata": []}
    adapter, seeds = setup_adapter(observer, report, monkeypatch)
    assert probe.execute_cases(adapter, observer, report, lambda: None) == "long_probe_complete"
    row = report["cases"][0]
    assert seeds == [42] and adapter._model.calls == 1
    assert row["numerical_horizon_reached"] and row["cached_forward_count"] == 95
    assert row["forward_calls_attempted"] == row["generation"]["output_tokens"] == 96
    assert row["stop_reason"] == "token_cap" and all(row["output_quality"].values())
    assert "generate" not in adapter._model.__dict__


def test_early_eos_is_preserved_with_no_retry_and_no_horizon_claim(monkeypatch):
    observer, report = SimpleNamespace(step=0), {"cases": [], "forward_metadata": []}
    adapter, seeds = setup_adapter(observer, report, monkeypatch, count=2, eos=True)
    assert probe.execute_cases(adapter, observer, report, lambda: None) == "early_stop"
    row = report["cases"][0]
    assert not row["numerical_horizon_reached"] and row["stop_reason"] == "eos"
    assert row["generation"]["text"] == "1 2 3" and seeds == [42]
    assert adapter._model.calls == 1 and "generate" not in adapter._model.__dict__


def test_full_length_without_observed_cache_does_not_certify_cached_path(monkeypatch):
    observer, report = SimpleNamespace(step=0), {"cases": [], "forward_metadata": []}
    adapter, _ = setup_adapter(observer, report, monkeypatch, cache=False)
    assert probe.execute_cases(adapter, observer, report, lambda: None) == "output_quality_stop"
    assert report["cases"][0]["numerical_horizon_reached"]
    assert not report["cases"][0]["output_quality"]["all_96_forwards_and_95_cached_decode_forwards"]


def test_wrong_prefix_retains_full_numerical_coverage_as_separate_fact(monkeypatch):
    observer, report = SimpleNamespace(step=0), {"cases": [], "forward_metadata": []}
    adapter, _ = setup_adapter(observer, report, monkeypatch, text="1 2 4")
    assert probe.execute_cases(adapter, observer, report, lambda: None) == "output_quality_stop"
    assert report["cases"][0]["numerical_horizon_reached"]
    assert not report["cases"][0]["output_quality"]["requested_sequence_prefix"]


def test_capture_rejects_force_length_kwargs_before_model_call():
    calls = []
    capture = probe.GenerateCapture(lambda **kwargs: calls.append(kwargs))
    with pytest.raises(RuntimeError, match="decoding overrides"):
        capture(input_ids=torch.tensor([[1]]), max_new_tokens=96, do_sample=False, min_new_tokens=96)
    assert not calls


def test_failed_worker_is_sealed_once_and_existing_attempt_is_not_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in probe.HELPERS:
        (root / "scripts" / name).write_text("# helper\n")
    plan = root / "plan.md"
    plan.write_text("fixed plan")
    dest = root / "results/attempt"
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "PLAN", plan)
    monkeypatch.setattr(probe, "DEST", dest)
    calls = []
    class FailedProcess:
        stdout = iter(["failed\n"])
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(probe.subprocess, "Popen", FailedProcess)
    assert probe.run() == 1 and dest.with_suffix(".zip").is_file()
    assert json.loads((dest / "execution_status.json").read_text())["worker_exit_code"] == 1
    with pytest.raises(RuntimeError, match="Archive exists"):
        probe.run()
    assert len(calls) == 1
