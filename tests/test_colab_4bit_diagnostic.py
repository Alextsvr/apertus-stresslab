"""Small CPU probes of the observer; never load model weights."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_4bit_diagnostic as diag


def test_masks_and_integer_ids_are_excluded_but_hidden_states_are_checked():
    observer = diag.Observer()
    observer.inspect("attention", "input", {
        "attention_mask": torch.tensor([-float("inf")]),
        "nested": {"feature_mask": torch.tensor([float("nan")])},
        "input_ids": torch.tensor([0, 1]), "hidden_states": torch.tensor([1.0])})
    assert observer.events == 1
    assert observer.first_bad is None
    with pytest.raises(diag.NonFiniteObserved):
        observer.inspect("attention", "input", {"hidden_states": torch.tensor([float("nan")])})


def test_first_failure_survives_later_inspection_and_stats_are_json_safe():
    observer = diag.Observer()
    for name in ("first", "later"):
        with pytest.raises(diag.NonFiniteObserved):
            observer.inspect(name, "output", torch.tensor([float("inf"), -float("inf"), float("nan"), 2.0]))
    assert observer.first_bad["module"] == "first"
    assert observer.first_bad["nan"] == 1
    assert observer.first_bad["positive_inf"] == 1
    assert observer.first_bad["negative_inf"] == 1
    import json
    json.dumps(list(observer.trace), allow_nan=False)


def test_finite_dtype_minimum_padding_is_accepted_and_trace_is_bounded():
    observer = diag.Observer()
    for _ in range(105):
        observer.inspect("logits", "output", (torch.tensor([torch.finfo(torch.float16).min], dtype=torch.float16),))
    assert observer.events == 105
    assert len(observer.trace) == 100
    assert observer.first_bad is None


def test_worker_runtime_failure_is_archived_without_a_retry(tmp_path, monkeypatch):
    destination = tmp_path / "attempt"
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan")
    monkeypatch.setattr(diag, "DEST", destination)
    monkeypatch.setattr(diag, "PLAN", plan)
    calls = []

    class FailedProcess:
        stdout = iter(["diagnostic failed\n"])
        def __init__(self, command, **kwargs):
            calls.append(command)
        def poll(self):
            return 1
        def wait(self):
            return 1

    monkeypatch.setattr(diag.subprocess, "Popen", FailedProcess)
    assert diag.run() == 1
    assert destination.with_suffix(".zip").exists()
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Archive already exists"):
        diag.run()
    assert len(calls) == 1
