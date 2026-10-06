"""Numerical-reference tests use tiny CPU tensors, never model weights."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_layer_replay as replay


def test_common_operands_separate_final_output_range_overflow():
    x = torch.tensor([[40000.0, 40000.0]], dtype=torch.float16)
    w = torch.tensor([[1.0, 1.0]], dtype=torch.float16)
    before = replay.tensor_identity(x)
    y16 = replay.reference_linear(x, w, torch.float16)
    y32 = replay.reference_linear(x, w, torch.float32)
    assert torch.isposinf(y16).all()
    assert y32.item() == 80000.0
    assert torch.isposinf(y32.half()).all()
    assert replay.tensor_identity(x) == before


def test_in_range_reference_distinguishes_an_infinite_actual_path():
    actual = torch.tensor([[float("inf"), 3.0]], dtype=torch.float16)
    reference = torch.tensor([[10.0, 2.0]], dtype=torch.float32)
    result = replay.comparison(actual, reference)
    assert result["jointly_finite_count"] == 1
    assert result["max_absolute_difference"] == result["rmse"] == 1.0
    assert result["nonfinite_mask_mismatch_count"] == 1
    assert result["positive_inf_mask_mismatch_count"] == 1
    assert replay.finite(reference.half())


def test_nonfinite_statistics_and_coordinate_values_are_json_safe():
    actual = torch.tensor([float("nan"), -float("inf")])
    result = replay.comparison(actual, torch.tensor([float("inf"), -float("inf")]))
    assert result["jointly_finite_count"] == 0
    assert result["rmse"] is None
    assert result["nan_mask_mismatch_count"] == 1
    assert result["positive_inf_mask_mismatch_count"] == 1
    values = [replay.scalar(v) for v in actual.tolist()]
    assert values == ["NaN", "-Inf"]
    json.dumps({"comparison": result, "values": values}, allow_nan=False)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(RuntimeError, match="shapes differ"):
        replay.comparison(torch.zeros(2), torch.zeros(3))


def test_failed_capture_is_sealed_and_existing_attempt_is_preserved(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("colab_quantization.py", "colab_4bit_diagnostic.py", "verify_evidence.py"):
        (root / "scripts" / name).write_text("# frozen helper\n")
    plan = root / "plan.md"
    plan.write_text("frozen plan")
    destination = root / "results/attempt"
    monkeypatch.setattr(replay, "ROOT", root)
    monkeypatch.setattr(replay, "PLAN", plan)
    monkeypatch.setattr(replay, "DEST", destination)
    calls = []

    class FailedProcess:
        stdout = iter(["capture failed\n"])
        def __init__(self, command, **kwargs):
            calls.append(command)
        def poll(self):
            return 1
        def wait(self):
            return 1

    monkeypatch.setattr(replay.subprocess, "Popen", FailedProcess)
    assert replay.run() == 1
    assert destination.with_suffix(".zip").is_file()
    assert json.loads((destination / "execution_status.json").read_text())["worker_exit_code"] == 1
    with pytest.raises(RuntimeError, match="Archive exists"):
        replay.run()
    assert len(calls) == 1
