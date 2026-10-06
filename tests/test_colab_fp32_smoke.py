"""CPU-only checks for frozen adapter delegation, quality stops and preservation."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_fp32_smoke as smoke
from stresslab.models import ApertusAdapter


def test_capture_delegates_unchanged_once_and_returns_identical_object():
    ids = torch.tensor([[1, 5]])
    mask = torch.ones_like(ids)
    output = torch.tensor([[1, 5, 9, 2]])
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        assert kwargs["input_ids"] is ids and kwargs["attention_mask"] is mask
        return output
    capture = smoke.GenerateCapture(original)
    kwargs = dict(input_ids=ids, attention_mask=mask, max_new_tokens=32, do_sample=False)
    assert capture(**kwargs) is output
    assert calls == [kwargs] and capture.output_ids == output.tolist()
    with pytest.raises(RuntimeError, match="More than one"):
        capture(**kwargs)
    assert len(calls) == 1


def test_unplanned_override_aborts_before_delegate():
    called = []
    capture = smoke.GenerateCapture(lambda **kwargs: called.append(kwargs))
    with pytest.raises(RuntimeError, match="decoding overrides"):
        capture(input_ids=torch.tensor([[1]]), max_new_tokens=32, do_sample=False, use_cache=False)
    assert not called


def test_quality_distinguishes_eos_unknown_empty_and_wrong_answer():
    assert all(smoke.quality(" RED\n green blue ", [9, 2], 0, [2], "red green blue").values())
    assert not smoke.quality("hello", [9], 0, [2], "hello")["stopped_on_eos_within_cap"]
    assert not smoke.quality("hello", [0, 2], 0, [2], "hello")["no_unknown_token_selected"]
    assert not smoke.quality("hello.", [9, 2], 0, [2], "hello")["exact_expected_after_whitespace_and_case_normalization"]
    assert not smoke.quality("", [2], 0, [2], "hello")["nonempty_cleaned_response"]
    assert not smoke.quality("hello", [9]*32+[2], 0, [2], "hello")["stopped_on_eos_within_cap"]


def setup_frozen_adapter(observer, monkeypatch, wrong_at=None, raise_at=None):
    # Exercise the actual frozen adapter's template, seed, generate and decode flow.
    seeds = []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=seeds.append))
    template_calls = []
    texts = {10: "hello", 11: "red green blue", 12: "1 2 3 4 5 6 7 8", 99: "wrong"}
    class Batch(dict):
        def to(self, device):
            assert str(device) == "cpu"
            return self
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=2, unk_token_id=0)
        def apply_chat_template(self, messages, **kwargs):
            template_calls.append((messages, kwargs))
            return Batch(input_ids=torch.tensor([[1, 7]]), attention_mask=torch.tensor([[1, 1]]))
        def decode(self, ids, skip_special_tokens):
            values = ids.tolist() if hasattr(ids, "tolist") else ids
            text = texts[values[0]]
            return text if skip_special_tokens else text + "<eos>"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.device = torch.device("cpu")
            self.calls = []
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["max_new_tokens"] == 32 and kwargs["do_sample"] is False
            assert "use_cache" not in kwargs
            observer.step += 1
            if raise_at == len(self.calls):
                raise RuntimeError("observed failure")
            token = 99 if wrong_at == len(self.calls) else 9 + len(self.calls)
            return torch.tensor([[1, 7, token, 2]])
    adapter = ApertusAdapter()
    adapter._model, adapter._processor = Model(), Processor()
    return adapter, seeds, template_calls


def test_actual_frozen_adapter_three_calls_no_manual_selection(monkeypatch):
    observer = SimpleNamespace(step=0)
    adapter, seeds, templates = setup_frozen_adapter(observer, monkeypatch)
    report = {"cases": [], "preceding_hello_input_ids": [[1, 7]]}
    assert smoke.execute_cases(adapter, observer, report, lambda: None) == "smoke_complete"
    assert seeds == [42, 42, 42] and len(adapter._model.calls) == 3
    assert [r["generated_ids"] for r in report["cases"]] == [[10, 2], [11, 2], [12, 2]]
    assert all(r["generate_calls"] == 1 and r["forward_calls_attempted"] == 1 for r in report["cases"])
    assert all(kwargs["enable_thinking"] is False for _, kwargs in templates)
    assert "generate" not in adapter._model.__dict__


def test_wrong_second_answer_stops_before_third_and_restores_method(monkeypatch):
    observer = SimpleNamespace(step=0)
    adapter, seeds, _ = setup_frozen_adapter(observer, monkeypatch, wrong_at=2)
    report = {"cases": [], "preceding_hello_input_ids": [[1, 7]]}
    assert smoke.execute_cases(adapter, observer, report, lambda: None) == "output_quality_stop"
    assert len(report["cases"]) == len(seeds) == 2
    assert report["cases"][-1]["generation"]["text"] == "wrong"
    assert "generate" not in adapter._model.__dict__


def test_failure_preserves_attempted_counts_and_does_not_continue(monkeypatch):
    observer = SimpleNamespace(step=0)
    adapter, seeds, _ = setup_frozen_adapter(observer, monkeypatch, raise_at=1)
    report = {"cases": [], "preceding_hello_input_ids": [[1, 7]]}
    with pytest.raises(RuntimeError, match="observed failure"):
        smoke.execute_cases(adapter, observer, report, lambda: None)
    assert len(seeds) == 1 and report["cases"][0]["forward_calls_attempted"] == 1
    assert report["cases"][0]["generate_calls"] == 1 and "generate" not in adapter._model.__dict__


def test_accessible_cache_observations_do_not_mutate_cache():
    keys = torch.tensor([117986.203125])
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=torch.tensor([1.]))])
    values = smoke.cache_values(cache)
    assert values["layer_0_keys"] is keys
    observer = smoke.FP32Observer(io.StringIO())
    observer.inspect("kv_cache", "input", values)
    assert observer.events == 2 and cache.layers[0].keys is keys
    with pytest.raises(smoke.UnexpectedPrecision):
        observer.inspect("kv_cache", "input", {"keys": keys.half()})


def test_failed_worker_seals_and_cannot_retry(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in smoke.HELPERS:
        (root / "scripts" / name).write_text("# helper\n")
    plan = root / "plan.md"
    plan.write_text("frozen plan")
    dest = root / "results/attempt"
    monkeypatch.setattr(smoke, "ROOT", root)
    monkeypatch.setattr(smoke, "PLAN", plan)
    monkeypatch.setattr(smoke, "DEST", dest)
    calls = []
    class FailedProcess:
        stdout = iter(["failed\n"])
        def __init__(self, *args, **kwargs):
            calls.append(args)
        def wait(self):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(smoke.subprocess, "Popen", FailedProcess)
    assert smoke.run() == 1 and dest.with_suffix(".zip").is_file()
    assert json.loads((dest / "execution_status.json").read_text())["worker_exit_code"] == 1
    with pytest.raises(RuntimeError, match="Archive exists"):
        smoke.run()
    assert len(calls) == 1
