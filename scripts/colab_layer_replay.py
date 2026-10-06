"""Capture one NF4 projection and compare isolated numerical paths once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from colab_quantization import ROOT, MODEL, REVISION, check, claim, git, require, save, seal, sha
from colab_4bit_diagnostic import PROMPT, Observer, NonFiniteObserved, stats

DEST = ROOT / "results/colab-layer-replay-2026-10-06"
PLAN = ROOT / "docs/colab_layer_replay_protocol.md"
TARGET = "model.language_model.layers.2.mlp.down_proj"
PRIOR = ROOT / "results/colab-4bit-diagnostic-2026-10-06.zip"
PRIOR_SHA = "50f77cd2ba0aa919f53ae4a466a39067576f099aea326c3d8277dce65bdd109a"


class CaptureComplete(RuntimeError):
    pass


def tensor_identity(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def weight_identity(module):
    return {"packed_weight": tensor_identity(module.weight),
            "quant_state": {key: tensor_identity(value)
                            for key, value in module.weight.quant_state.as_dict(packed=True).items()}}


def finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all().item())


def comparison(actual, reference):
    import torch
    require(actual.shape == reference.shape, "Comparison shapes differ")
    a, r = actual.float(), reference.float()
    valid = torch.isfinite(a) & torch.isfinite(r)
    delta = a[valid].double() - r[valid].double()
    return {"jointly_finite_count": int(valid.sum().item()),
            "max_absolute_difference": float(delta.abs().max().item()) if delta.numel() else None,
            "rmse": float(delta.square().mean().sqrt().item()) if delta.numel() else None,
            "nonfinite_mask_mismatch_count": int((torch.isfinite(a) != torch.isfinite(r)).sum().item()),
            "positive_inf_mask_mismatch_count": int((torch.isposinf(a) != torch.isposinf(r)).sum().item()),
            "negative_inf_mask_mismatch_count": int((torch.isneginf(a) != torch.isneginf(r)).sum().item()),
            "nan_mask_mismatch_count": int((torch.isnan(a) != torch.isnan(r)).sum().item())}


def scalar(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def reference_linear(x16, w16, dtype):
    import torch
    import torch.nn.functional as F
    require(x16.dtype == w16.dtype == torch.float16, "Expected common FP16 operands")
    require(dtype in (torch.float16, torch.float32), "Unsupported reference dtype")
    return F.linear(x16.to(dtype), w16.to(dtype))


def read_prior():
    require(sha(PRIOR) == PRIOR_SHA, "Preceding diagnostic archive differs or is missing")
    with zipfile.ZipFile(PRIOR) as z:
        names = z.namelist()
        entries = [line.split("  ", 1) for line in z.read("SHA256SUMS").decode().splitlines()]
        require(len(entries) == 8 and len(names) == 9
                and set(name for _, name in entries) == set(names) - {"SHA256SUMS"}, "Prior manifest incomplete")
        for digest, name in entries:
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"Prior hash differs: {name}")
        return json.loads(z.read("preflight.json")), json.loads(z.read("diagnostic.json"))


def precision_settings(torch):
    return {"fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}


def worker():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    report = {"prompt": PROMPT, "target": TARGET, "model_forward_calls": 0,
              "isolated_matmul_calls": 0, "generation_performed": False,
              "scoring_performed": False, "outcome": "runtime_error", "outputs": {}}
    observer, captured, handles = Observer(), {}, []
    original_settings = None
    target = None
    try:
        previous, previous_result = read_prior()
        preflight = check()
        for relative in ("scripts/colab_layer_replay.py", "docs/colab_layer_replay_protocol.md"):
            require(bool(git("ls-files", "--", relative)), f"Not committed: {relative}")
        for key in ("versions", "all_packages", "gpu_identity", "torch_cuda", "source_sha256", "dataset_sha256", "model_revision"):
            require(previous[key] == preflight[key], f"Environment differs from prior probe: {key}")
        require(previous_result["outcome"] == "non_finite_detected"
                and previous_result["first_bad_activation"]["module"] == TARGET, "Wrong preceding outcome")
        preflight.update(layer_replay_plan_sha256=sha(PLAN), launcher_sha256=sha(Path(__file__)), prior_archive_sha256=PRIOR_SHA)
        import bitsandbytes.nn.modules as bnb_modules
        import bitsandbytes.autograd._functions as bnb_functions
        import bitsandbytes.backends.cuda.ops as bnb_cuda
        preflight["bitsandbytes_source_sha256"] = {
            name: sha(Path(module.__file__)) for name, module in
            (("nn.modules", bnb_modules), ("autograd._functions", bnb_functions), ("backends.cuda.ops", bnb_cuda))}
        save(DEST / "preflight.json", preflight)
        snapshot = Path(Path("/content/apertus-pinned-snapshot.txt").read_text().strip())
        require(snapshot.name == REVISION, "Cached model revision differs")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        require(len(shards) == 6 and all((snapshot / name).is_file() for name in shards), "Cached weights incomplete")
        set_seed(42)
        print("Loading cached NF4/FP16 model; capturing target projection only", flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype="float16", quantization="4bit",
                                 device_map="cuda:0", cpu_offload=False)
        adapter.load()
        model = adapter._model
        require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()), "Parameters off GPU")
        report["model_info"] = adapter.info().model_dump(mode="json")
        target = model.get_submodule(TARGET)
        require(isinstance(target, bnb.nn.Linear4bit) and target.bias is None, "Target is not bias-free Linear4bit")
        require(target.compute_dtype == torch.float16 and target.compute_type_is_set, "Target compute dtype differs")
        require(target.weight.quant_state.dtype == torch.float16
                and list(target.weight.quant_state.shape) == [4096, 21504], "Target weight metadata differs")
        report["weights_before"] = weight_identity(target)
        inputs = adapter._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False).to(adapter._input_device())
        report["input_ids"] = inputs["input_ids"].cpu().tolist()
        require(report["input_ids"] == previous_result["input_ids"], "Processor input IDs differ")
        original_settings = precision_settings(torch)
        report["model_prefix_precision_settings"] = original_settings
        for name, module in model.named_modules():
            if name == "lm_head" or name.startswith("model.language_model"):
                def before(mod, args, kwargs, name=name):
                    observer.inspect(name, "input", {"activation": args[:1], **kwargs})
                    if name == TARGET:
                        require("input" not in captured, "Target called twice")
                        captured["input"] = args[0].detach().clone()
                def after(mod, args, output, name=name):
                    if name == TARGET:
                        captured["bnb_fp16_original"] = output.detach().clone()
                        try:
                            observer.inspect(name, "output", output)
                        except NonFiniteObserved:
                            pass
                        raise CaptureComplete("Target captured; stopping model")
                    observer.inspect(name, "output", output)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
        torch.cuda.reset_peak_memory_stats()
        report["model_forward_calls"] = 1
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=False, logits_to_keep=1)
            except CaptureComplete:
                pass
        for handle in handles:
            handle.remove()
        handles.clear()
        require(set(captured) == {"input", "bnb_fp16_original"}, "Capture incomplete")
        x = captured["input"]
        require(x.dtype == torch.float16 and list(x.shape) == [1, 70, 21504] and finite(x), "Captured input differs or invalid")
        report["captured_input"] = {**stats(x), **tensor_identity(x)}
        np.save(DEST / "input.npy", x.cpu().numpy(), allow_pickle=False)

        def preserve(name, value):
            report["outputs"][name] = {**stats(value), **tensor_identity(value)}
            np.save(DEST / f"{name}.npy", value.detach().cpu().numpy(), allow_pickle=False)
            save(DEST / "comparison.json", report)
            return value

        with torch.inference_mode():
            outputs = {"bnb_fp16_original": preserve("bnb_fp16_original", captured["bnb_fp16_original"])}
            print("Captured projection. Replaying common operands in FP16 and FP32", flush=True)
            w = bnb.functional.dequantize_4bit(target.weight.data, quant_state=target.weight.quant_state)
            require(w.dtype == torch.float16 and list(w.shape) == [4096, 21504] and finite(w), "Dequantized weight invalid")
            report["common_dequantized_weight"] = {**stats(w), **tensor_identity(w)}
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            report["reference_precision_settings"] = precision_settings(torch)
            # Preserve each result before beginning the next fixed operation.
            report["isolated_matmul_calls"] += 1
            outputs["reference_fp16"] = preserve("reference_fp16", reference_linear(x, w, torch.float16))
            report["isolated_matmul_calls"] += 1
            outputs["reference_fp32"] = preserve("reference_fp32", reference_linear(x, w, torch.float32))
            outputs["reference_fp32_cast_fp16"] = preserve("reference_fp32_cast_fp16", outputs["reference_fp32"].half())
            report["isolated_matmul_calls"] += 1
            outputs["bnb_fp32"] = preserve("bnb_fp32", bnb.matmul_4bit(x.float(), target.weight, quant_state=target.weight.quant_state))
            report["comparisons_to_reference_fp32"] = {name: comparison(value, outputs["reference_fp32"])
                                                      for name, value in outputs.items()}
            ref = outputs["reference_fp32"]
            report["reference_fp32_exceeds_fp16_limit_count"] = int((torch.isfinite(ref) & (ref.abs() > torch.finfo(torch.float16).max)).sum().item())
            coords = (~torch.isfinite(outputs["bnb_fp16_original"])).nonzero()[:10].tolist()
            report["original_nonfinite_coordinates"] = [
                {"index": coord, "values": {name: scalar(value[tuple(coord)].item()) for name, value in outputs.items()}}
                for coord in coords]
        report["weights_after"] = weight_identity(target)
        require(report["weights_before"] == report["weights_after"], "Packed weights/state changed")
        require(report["captured_input"]["sha256"] == tensor_identity(x)["sha256"], "Captured input mutated")
        report["outcome"] = "paired_replay_complete"
    except NonFiniteObserved as exc:
        report.update(outcome="nonfinite_before_target", detail=str(exc))
    except Exception as exc:
        report["detail"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        if original_settings is not None:
            torch.backends.cuda.matmul.fp32_precision = original_settings["fp32_precision"]
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_settings["allow_fp16_reduced_precision_reduction"]
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), first_bad_activation=observer.first_bad,
                      observed_tensor_events=observer.events, peak_gpu_memory_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        save(DEST / "trace.json", list(observer.trace))
        save(DEST / "comparison.json", report)
    print(f"LAYER_REPLAY_OUTCOME: {report['outcome']}", flush=True)
    return int(report["outcome"] != "paired_replay_complete")


def run():
    require(not DEST.with_suffix(".zip").exists(), "Archive exists; preserve it, do not retry")
    claim(DEST)
    shutil.copyfile(PLAN, DEST / "protocol.md")
    shutil.copyfile(Path(__file__), DEST / "launcher.py")
    for name in ("colab_quantization.py", "colab_4bit_diagnostic.py", "verify_evidence.py"):
        shutil.copyfile(ROOT / "scripts" / name, DEST / name)
    code = None
    try:
        with (DEST / "replay.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--worker"], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
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
        save(DEST / "execution_status.json", {"worker_exit_code": code, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Evidence archive: {seal(DEST)}", flush=True)
    return 1 if code is None else code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.exit(worker() if args.worker else run())
