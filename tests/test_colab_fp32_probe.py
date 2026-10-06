"""Meaningful small CPU tests for promotion, dtype guards and quality checks."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_fp32_probe as probe


def test_promotion_preserves_packed_state_and_exact_loaded_values():
    class PackedLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1, 127, 255], dtype=torch.uint8), requires_grad=False)
            self.weight.quant_state = SimpleNamespace(dtype=torch.float16, marker="unchanged")
            self.compute_dtype = torch.float16
            self.compute_type_is_set = True

    model = torch.nn.Module()
    model.add_module("packed", PackedLayer())
    model.register_parameter("floating", torch.nn.Parameter(torch.tensor([12824.0, -0.02255249], dtype=torch.float16)))
    model.register_buffer("floating_buffer", torch.tensor([0.5], dtype=torch.float16))
    model.register_buffer("integer_buffer", torch.tensor([3]))
    original = model.floating.detach().float().clone()
    packed = model.packed.weight.clone()
    state = model.packed.weight.quant_state
    parameter_object = model.floating
    result = probe.promote(model, PackedLayer)
    assert model.floating is parameter_object
    assert torch.equal(model.floating, original)
    assert model.floating.dtype == model.floating_buffer.dtype == torch.float32
    assert model.integer_buffer.dtype == torch.int64
    assert torch.equal(model.packed.weight, packed)
    assert model.packed.weight.quant_state is state and state.dtype == torch.float16
    assert model.packed.compute_dtype == torch.float32
    assert result["quantized_module_count"] == 1


def test_observer_rejects_finite_fp16_before_it_can_select_a_token():
    stream = io.StringIO()
    observer = probe.FP32Observer(stream)
    with pytest.raises(probe.UnexpectedPrecision, match="Expected FP32"):
        observer.inspect("down_proj", "output", torch.tensor([12824.0], dtype=torch.float16))
    assert observer.first_wrong_dtype["dtype"] == "torch.float16"
    assert observer.first_bad is None
    assert len(stream.getvalue().splitlines()) == 1


def test_observer_preserves_large_finite_fp32_and_stops_on_inf():
    stream = io.StringIO()
    observer = probe.FP32Observer(stream)
    observer.step = 2
    observer.inspect("down_proj", "output", torch.tensor([117986.203125], dtype=torch.float32))
    observer.inspect("attention", "input", {"attention_mask": torch.tensor([-float("inf")]), "hidden": torch.tensor([1.0])})
    with pytest.raises(probe.NonFiniteObserved):
        observer.inspect("logits", "output", torch.tensor([float("inf")]))
    assert observer.events == 3
    assert observer.first_bad["positive_inf"] == 1
    for line in stream.getvalue().splitlines():
        json.dumps(json.loads(line), allow_nan=False)


def test_quality_checks_do_not_equate_finite_or_nonempty_with_success():
    quality = probe.output_quality("", [0] * 8, 0, False)
    assert not any(quality.values())
    assert all(probe.output_quality(" Hello ", [42, 2], 0, True).values())
    wrong = probe.output_quality("world", [43, 2], 0, True)
    assert wrong["nonempty_cleaned_response"] and not wrong["exact_case_insensitive_hello"]


def test_failed_worker_seals_evidence_and_cannot_retry(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("colab_quantization.py", "colab_4bit_diagnostic.py", "colab_layer_replay.py", "verify_evidence.py"):
        (root / "scripts" / name).write_text("# frozen helper\n")
    plan = root / "plan.md"
    plan.write_text("frozen plan")
    destination = root / "results/attempt"
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "PLAN", plan)
    monkeypatch.setattr(probe, "DEST", destination)
    calls = []

    class FailedProcess:
        stdout = iter(["promotion failed\n"])
        def __init__(self, command, **kwargs):
            calls.append(command)
        def poll(self):
            return 1
        def wait(self):
            return 1

    monkeypatch.setattr(probe.subprocess, "Popen", FailedProcess)
    assert probe.run() == 1
    assert destination.with_suffix(".zip").is_file()
    with pytest.raises(RuntimeError, match="Archive exists"):
        probe.run()
    assert len(calls) == 1


def test_dependency_gate_reports_actual_hash_and_does_not_modify_file(tmp_path, monkeypatch):
    import hashlib
    dependency = tmp_path / "wrong.zip"
    content = b"different dependency bytes"
    dependency.write_bytes(content)
    monkeypatch.setattr(probe, "PRIOR", dependency)
    actual = hashlib.sha256(content).hexdigest()
    with pytest.raises(RuntimeError, match=f"expected {probe.PRIOR_SHA}, got {actual}"):
        probe.read_prior()
    assert dependency.read_bytes() == content


def test_missing_dependency_reports_path_without_creating_it(tmp_path, monkeypatch):
    dependency = tmp_path / "missing.zip"
    monkeypatch.setattr(probe, "PRIOR", dependency)
    with pytest.raises(RuntimeError, match="Verified dependency missing"):
        probe.read_prior()
    assert not dependency.exists()


def test_new_t4_uuid_and_driver_are_recorded_without_rejecting_functional_probe():
    keys = ("versions", "all_packages", "torch_cuda", "source_sha256", "dataset_sha256", "model_revision")
    previous = {key: "same" for key in keys}
    previous.update(gpu="Tesla T4", gpu_identity="GPU-old, 580.82.07")
    current = dict(previous, gpu_identity="GPU-new, 580.99.01", gpu_compute_capability=[7, 5])
    result = probe.check_probe_environment(previous, current)
    assert result["uuid_changed"] and result["driver_changed"]
    assert result["current"]["uuid"] == "GPU-new"
    assert result["previous"]["driver_version"] == "580.82.07"
    assert not result["cross_session_identity_equality_required"]


def test_same_class_allowance_does_not_accept_software_source_or_hardware_changes():
    keys = ("versions", "all_packages", "torch_cuda", "source_sha256", "dataset_sha256", "model_revision")
    previous = {key: "same" for key in keys}
    previous.update(gpu="Tesla T4", gpu_identity="GPU-old, 580.82.07")
    current = dict(previous, gpu_identity="GPU-new, 580.82.07", gpu_compute_capability=[7, 5])
    for key in (*keys, "gpu"):
        with pytest.raises(RuntimeError, match=key):
            probe.check_probe_environment(previous, dict(current, **{key: "different"}))
    with pytest.raises(RuntimeError, match="compute capability"):
        probe.check_probe_environment(previous, dict(current, gpu_compute_capability=[8, 0]))
