"""One technical NF4 continuation with promoted FP32 activations, max eight tokens."""
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
from colab_4bit_diagnostic import PROMPT, NonFiniteObserved, tensors, stats
from colab_layer_replay import TARGET, weight_identity, precision_settings

DEST = ROOT / "results/colab-fp32-activation-probe-2026-10-06"
PLAN = ROOT / "docs/colab_fp32_probe_protocol.md"
PRIOR = ROOT / "results/colab-layer-replay-2026-10-06.zip"
PRIOR_SHA = "ef98d1fe60a71adf4abe2082a9a4974d8ec8ba4eb97e04e596a1d6cd83b2ae23"
CAP = 8


class UnexpectedPrecision(RuntimeError):
    pass


def quantized_identity(model, linear4bit_class):
    return {name: weight_identity(module) for name, module in model.named_modules()
            if isinstance(module, linear4bit_class)}


def promote(model, linear4bit_class):
    import torch
    quantized = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, linear4bit_class)]
    require(bool(quantized), "No 4-bit modules found")
    for name, module in quantized:
        require(module.weight.dtype == torch.uint8 and module.weight.quant_state is not None,
                f"Packed weight/state unavailable: {name}")
    changed_parameters, changed_buffers = [], []
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            changed_parameters.append(name)
    for name, module in model.named_modules():
        for field, value in list(module._buffers.items()):
            if value is not None and value.is_floating_point() and value.dtype != torch.float32:
                module._buffers[field] = value.float()
                changed_buffers.append(f"{name}.{field}")
    for _, module in quantized:
        module.compute_dtype = torch.float32
        module.compute_type_is_set = True
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.parameters()),
            "Floating parameters did not retain FP32")
    require(all(not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers()),
            "Floating registered buffers did not retain FP32")
    return {"promoted_parameters": changed_parameters, "promoted_registered_buffers": changed_buffers,
            "quantized_module_count": len(quantized),
            "parameter_elements_by_dtype": {dtype: sum(p.numel() for p in model.parameters() if str(p.dtype) == dtype)
                                            for dtype in sorted({str(p.dtype) for p in model.parameters()})}}


class FP32Observer:
    def __init__(self, stream):
        self.stream = stream
        self.trace = deque(maxlen=100)
        self.events = 0
        self.step = 0
        self.first_bad = None
        self.first_wrong_dtype = None

    def inspect(self, name, stage, value):
        import torch
        for path, tensor in tensors(value):
            event = {"step": self.step, "module": name, "stage": stage, "path": path, **stats(tensor)}
            self.events += 1
            self.trace.append(event)
            self.stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.stream.flush()
            if tensor.dtype != torch.float32:
                self.first_wrong_dtype = event
                raise UnexpectedPrecision(f"Expected FP32 at {name}: {stage} {path}; got {tensor.dtype}")
            if event["nan"] or event["positive_inf"] or event["negative_inf"]:
                self.first_bad = event
                raise NonFiniteObserved(f"Non-finite activation at {name}: {stage} {path}")


def output_quality(text, ids, unk, stopped_on_eos):
    return {"nonempty_cleaned_response": bool(text.strip()),
            "no_unknown_token_selected": unk is None or unk not in ids,
            "stopped_on_eos_within_cap": stopped_on_eos,
            "exact_case_insensitive_hello": text.strip().casefold() == "hello"}


def read_prior():
    require(sha(PRIOR) == PRIOR_SHA, "Preceding replay archive differs or is missing")
    with zipfile.ZipFile(PRIOR) as z:
        entries = [line.split("  ", 1) for line in z.read("SHA256SUMS").decode().splitlines()]
        require(len(entries) == 16 and len(z.namelist()) == 17
                and {n for _, n in entries} == set(z.namelist()) - {"SHA256SUMS"}, "Prior manifest incomplete")
        for digest, name in entries:
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"Prior hash differs: {name}")
        return json.loads(z.read("preflight.json")), json.loads(z.read("comparison.json"))


def worker():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import bitsandbytes as bnb
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    report = {"prompt": PROMPT, "max_new_tokens": CAP, "forward_calls_attempted": 0,
              "outcome": "runtime_error", "steps": [], "generated_ids": [],
              "raw_response": "", "response": "", "scoring_performed": False,
              "generation_method": "manual greedy argmax; no generation processors; use_cache=False"}
    handles, settings = [], None
    stream = (DEST / "tensor_events.jsonl").open("w", encoding="utf-8")
    observer = FP32Observer(stream)
    model, processor = None, None
    try:
        previous, previous_result = read_prior()
        require(previous_result["outcome"] == "paired_replay_complete", "Previous replay incomplete")
        preflight = check()
        for relative in ("scripts/colab_fp32_probe.py", "docs/colab_fp32_probe_protocol.md"):
            require(bool(git("ls-files", "--", relative)), f"Not committed: {relative}")
        for key in ("versions", "all_packages", "gpu_identity", "torch_cuda", "source_sha256", "dataset_sha256", "model_revision"):
            require(previous[key] == preflight[key], f"Environment differs from replay: {key}")
        preflight.update(probe_plan_sha256=sha(PLAN), probe_launcher_sha256=sha(Path(__file__)),
                         preceding_archive_sha256=PRIOR_SHA)
        save(DEST / "preflight.json", preflight)
        snapshot = Path(Path("/content/apertus-pinned-snapshot.txt").read_text().strip())
        require(snapshot.name == REVISION, "Cached model revision differs")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        require(len(shards) == 6 and all((snapshot / name).is_file() for name in shards), "Cached weights incomplete")
        set_seed(42)
        print("Loading existing NF4 weights; no inference before FP32 promotion", flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype="float16", quantization="4bit",
                                 device_map="cuda:0", cpu_offload=False)
        adapter.load()
        model, processor = adapter._model, adapter._processor
        report["model_loading_info"] = adapter.info().model_dump(mode="json")
        report["quantized_weights_before_promotion"] = quantized_identity(model, bnb.nn.Linear4bit)
        require(report["quantized_weights_before_promotion"][TARGET] == previous_result["weights_before"],
                "Loaded target packed weights/state differ from preceding replay")
        save(DEST / "probe.json", report)
        torch.cuda.reset_peak_memory_stats()
        print("Promoting floating parameters/buffers and compute to FP32", flush=True)
        report["promotion"] = promote(model, bnb.nn.Linear4bit)
        require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()), "Parameters off GPU")
        report["quantized_weights_after_promotion"] = quantized_identity(model, bnb.nn.Linear4bit)
        require(report["quantized_weights_before_promotion"] == report["quantized_weights_after_promotion"],
                "Packed weights or quantization state changed during promotion")
        settings = precision_settings(torch)
        report["original_precision_settings"] = settings
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report["probe_precision_settings"] = precision_settings(torch)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report["post_promotion_memory"] = {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                                          "cuda_free_gib": round(free / 2**30, 3), "cuda_total_gib": round(total / 2**30, 3)}
        require(free >= 2**30, "Need >=1 GiB CUDA memory after promotion; no fallback")
        for name, module in model.named_modules():
            if name == "lm_head" or name.startswith("model.language_model"):
                def before(mod, args, kwargs, name=name):
                    observer.inspect(name, "input", {"activation": args[:1], **kwargs})
                def after(mod, args, output, name=name):
                    observer.inspect(name, "output", output)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False).to("cuda:0")
        require(set(inputs) <= {"input_ids", "attention_mask"}, "Unexpected text-template input fields")
        report["input_ids"] = inputs["input_ids"].cpu().tolist()
        require(report["input_ids"] == previous_result["input_ids"], "Processor input IDs differ")
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = processor.tokenizer.eos_token_id
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        report["eos_token_ids"] = eos_ids
        unk = processor.tokenizer.unk_token_id
        report["unknown_token_id"] = unk
        stopped = False
        with torch.inference_mode():
            for step in range(1, CAP + 1):
                observer.step = step
                report["forward_calls_attempted"] = step
                print(f"FP32 continuation: step {step}/{CAP}", flush=True)
                output = model(**inputs, use_cache=False, logits_to_keep=1)
                observer.inspect("final_logits", "output", output.logits)
                scores = output.logits[0, -1]
                token = int(scores.argmax().item())
                values, ids = torch.topk(scores, 5)
                item = {"step": step, "selected_token_id": token,
                        "selected_raw_token": processor.decode([token], skip_special_tokens=False),
                        "top_tokens": [{"id": int(i), "raw_text": processor.decode([int(i)], skip_special_tokens=False),
                                        "score": float(v)} for i, v in zip(ids.tolist(), values.tolist())],
                        "unknown_token": None if unk is None else {"id": unk, "score": float(scores[unk].item()),
                                                "rank_with_ties": int((scores > scores[unk]).sum().item()) + 1}}
                report["steps"].append(item)
                report["generated_ids"].append(token)
                report["raw_response"] = processor.decode(report["generated_ids"], skip_special_tokens=False)
                report["response"] = processor.decode(report["generated_ids"], skip_special_tokens=True).strip()
                stopped = token in eos_ids
                report["stop_reason"] = "eos" if stopped else "in_progress"
                save(DEST / "probe.json", report)
                if stopped:
                    break
                inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([[token]], dtype=inputs["input_ids"].dtype, device="cuda:0")), dim=-1)
                inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device="cuda:0")), dim=-1)
        report["stop_reason"] = "eos" if stopped else "token_cap"
        report["output_quality"] = output_quality(report["response"], report["generated_ids"], unk, stopped)
        report["quantized_weights_after_continuation"] = quantized_identity(model, bnb.nn.Linear4bit)
        require(report["quantized_weights_before_promotion"] == report["quantized_weights_after_continuation"],
                "Packed weights/state changed during continuation")
        report["outcome"] = "short_probe_complete"
    except NonFiniteObserved as exc:
        report.update(outcome="non_finite_detected", detail=str(exc), stop_reason="non_finite_activation")
    except UnexpectedPrecision as exc:
        report.update(outcome="unexpected_precision", detail=str(exc), stop_reason="unexpected_precision")
    except Exception as exc:
        report["detail"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        stream.close()
        if settings is not None:
            torch.backends.cuda.matmul.fp32_precision = settings["fp32_precision"]
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings["allow_fp16_reduced_precision_reduction"]
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), first_bad_activation=observer.first_bad,
                      first_wrong_dtype=observer.first_wrong_dtype, observed_tensor_events=observer.events,
                      peak_gpu_memory_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        save(DEST / "last_events.json", list(observer.trace))
        save(DEST / "probe.json", report)
    print(f"FP32_PROBE_OUTCOME: {report['outcome']}", flush=True)
    return int(report["outcome"] not in {"short_probe_complete", "non_finite_detected"})


def run():
    require(not DEST.with_suffix(".zip").exists(), "Archive exists; preserve it, do not retry")
    claim(DEST)
    shutil.copyfile(PLAN, DEST / "protocol.md")
    shutil.copyfile(Path(__file__), DEST / "launcher.py")
    for name in ("colab_quantization.py", "colab_4bit_diagnostic.py", "colab_layer_replay.py", "verify_evidence.py"):
        shutil.copyfile(ROOT / "scripts" / name, DEST / name)
    code = None
    try:
        with (DEST / "probe.log").open("w", encoding="utf-8") as log:
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
