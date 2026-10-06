"""Three prospective technical prompts through the frozen ApertusAdapter.generate."""
from __future__ import annotations

import argparse
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
from colab_4bit_diagnostic import NonFiniteObserved
from colab_fp32_probe import FP32Observer, UnexpectedPrecision, promote, quantized_identity, check_probe_environment
from colab_layer_replay import precision_settings

DEST = ROOT / "results/colab-fp32-generation-smoke-2026-10-06"
PLAN = ROOT / "docs/colab_fp32_smoke_protocol.md"
PRIOR_SHA = "96c0c4553e5dd96f28982ac9b4a6504f30807e6b4f870a97198678382e554a2e"
PRIOR = ROOT / "results/evidence-dependencies" / f"{PRIOR_SHA}.zip"
CAP = 32
CASES = (
    {"id": "TECH-01", "prompt": "Reply with exactly one word: hello.", "expected": "hello"},
    {"id": "TECH-02", "prompt": "Reply with exactly these three words: red green blue.", "expected": "red green blue"},
    {"id": "TECH-03", "prompt": "Reply with exactly these numbers separated by spaces: 1 2 3 4 5 6 7 8.",
     "expected": "1 2 3 4 5 6 7 8"},
)
HELPERS = ("colab_quantization.py", "colab_4bit_diagnostic.py", "colab_layer_replay.py",
           "colab_fp32_probe.py", "verify_evidence.py")


def read_prior():
    require(PRIOR.is_file(), f"Verified successful probe missing: {PRIOR}")
    actual = sha(PRIOR)
    require(actual == PRIOR_SHA, f"Prior hash mismatch: expected {PRIOR_SHA}, got {actual}")
    with zipfile.ZipFile(PRIOR) as z:
        names = z.namelist()
        entries = [line.split("  ", 1) for line in z.read("SHA256SUMS").decode().splitlines()]
        require(len(names) == len(set(names)) == 13 and len(entries) == 12
                and len({n for _, n in entries}) == 12
                and {n for _, n in entries} == set(names) - {"SHA256SUMS"}, "Prior manifest incomplete")
        for digest, name in entries:
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"Prior hash differs: {name}")
        pre, result = json.loads(z.read("preflight.json")), json.loads(z.read("probe.json"))
        require(result["outcome"] == "short_probe_complete" and all(result["output_quality"].values()),
                "Expected successful preceding probe")
        return pre, result


def quality(text, ids, unk, eos_ids, expected):
    return {"nonempty_cleaned_response": bool(text.strip()),
            "no_unknown_token_selected": unk is None or unk not in ids,
            "stopped_on_eos_within_cap": bool(ids) and ids[-1] in eos_ids and len(ids) <= CAP,
            "exact_expected_after_whitespace_and_case_normalization":
                " ".join(text.split()).casefold() == " ".join(expected.split()).casefold()}


class GenerateCapture:
    """Delegate once with unchanged arguments and return the identical output object."""
    def __init__(self, original):
        self.original = original
        self.calls = 0
        self.input_ids = None
        self.output_ids = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        require(self.calls == 1, "More than one generate call for a technical prompt")
        require(not args and "input_ids" in kwargs, "Unexpected adapter generation interface")
        self.input_ids = kwargs["input_ids"].detach().cpu().tolist()
        self.kwargs = {k: v for k, v in kwargs.items() if k not in {"input_ids", "attention_mask"}}
        require(self.kwargs == {"max_new_tokens": CAP, "do_sample": False}, "Unexpected decoding overrides")
        result = self.original(*args, **kwargs)
        self.output_ids = result.detach().cpu().tolist()
        return result


def cache_values(cache):
    """Read recognized Cache layer storage without updating or replacing it."""
    values = {}
    if cache is None:
        return values
    for index, layer in enumerate(getattr(cache, "layers", ())):
        for field in ("keys", "values"):
            value = getattr(layer, field, None)
            if value is not None:
                values[f"layer_{index}_{field}"] = value
    return values


def execute_cases(adapter, observer, report, persist):
    from stresslab.schemas import GenerationConfig
    model, processor = adapter._model, adapter._processor
    config = GenerationConfig(max_new_tokens=CAP, do_sample=False, enable_thinking=False)
    report["adapter_generation_config"] = config.model_dump(mode="json")
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    unk = processor.tokenizer.unk_token_id
    report.update(eos_token_ids=eos_ids, unknown_token_id=unk)
    for case in CASES:
        observer.step = 0
        row = {**case, "outcome": "generation_started", "forward_calls_attempted": 0}
        report["cases"].append(row)
        persist()
        print(f"Normal adapter generation: {case['id']} ({len(report['cases'])}/3)", flush=True)
        original = model.generate
        had_instance_method = "generate" in model.__dict__
        prior_instance_method = model.__dict__.get("generate")
        capture = GenerateCapture(original)
        model.generate = capture
        try:
            result = adapter.generate(case["prompt"], config, seed=42)
            row["generation"] = result.model_dump(mode="json")
            require(capture.calls == 1 and capture.output_ids is not None, "No captured normal generation")
            require(len(capture.input_ids) == len(capture.output_ids) == 1, "Unexpected batch size")
            prompt_ids = capture.input_ids[0]
            all_ids = capture.output_ids[0]
            require(all_ids[:len(prompt_ids)] == prompt_ids, "Generation changed prompt prefix")
            new_ids = all_ids[len(prompt_ids):]
            row.update(input_ids=capture.input_ids, generated_ids=new_ids, supplied_generate_kwargs=capture.kwargs)
            require(0 < observer.step <= CAP and len(new_ids) <= CAP, "Unexpected forward/token count")
            require(result.input_tokens == len(prompt_ids) and result.output_tokens == len(new_ids),
                    "Adapter token counts differ from captured IDs")
            if case["id"] == "TECH-01":
                require(capture.input_ids == report["preceding_hello_input_ids"], "Hello template IDs changed")
            row["stop_reason"] = "eos" if new_ids and new_ids[-1] in eos_ids else "token_cap_or_other"
            row["output_quality"] = quality(result.text, new_ids, unk, eos_ids, case["expected"])
            row["outcome"] = "technical_output_checks_passed" if all(row["output_quality"].values()) else "output_quality_stop"
            persist()
            if row["outcome"] == "output_quality_stop":
                return "output_quality_stop"
        finally:
            if had_instance_method:
                model.generate = prior_instance_method
            else:
                del model.generate
            row["forward_calls_attempted"] = observer.step
            row["generate_calls"] = capture.calls
            if capture.input_ids is not None:
                row["input_ids"] = capture.input_ids
            persist()
    return "smoke_complete"


def worker():
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import bitsandbytes as bnb
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    report = {"outcome": "runtime_error", "cases": [], "planned_cases": list(CASES), "max_new_tokens": CAP,
              "scoring_performed": False, "generation_path": "frozen ApertusAdapter.generate -> model.generate",
              "total_forward_calls_attempted": 0, "forward_metadata": []}
    handles, settings, model = [], None, None
    stream = (DEST / "tensor_events.jsonl").open("w", encoding="utf-8")
    observer = FP32Observer(stream)
    persist = lambda: save(DEST / "smoke.json", report)
    try:
        previous, previous_result = read_prior()
        current = check()
        current.update(gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
                       smoke_plan_sha256=sha(PLAN), smoke_launcher_sha256=sha(Path(__file__)),
                       preceding_archive_sha256=PRIOR_SHA, dependency_path=str(PRIOR))
        save(DEST / "preflight.json", current)
        for relative in ("scripts/colab_fp32_smoke.py", "docs/colab_fp32_smoke_protocol.md"):
            require(bool(git("ls-files", "--", relative)), f"Not committed: {relative}")
        report["hardware_identity_comparison"] = check_probe_environment(previous, current)
        report["preceding_hello_input_ids"] = previous_result["input_ids"]
        current["hardware_identity_comparison"] = report["hardware_identity_comparison"]
        save(DEST / "preflight.json", current)
        snapshot = Path(Path("/content/apertus-pinned-snapshot.txt").read_text().strip())
        require(snapshot.name == REVISION, "Cached model revision differs")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        require(len(shards) == 6 and all((snapshot / name).is_file() for name in shards), "Cached weights incomplete")
        set_seed(42)
        print("Loading cached NF4 weights once, before FP32 promotion", flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype="float16", quantization="4bit",
                                 device_map="cuda:0", cpu_offload=False)
        adapter.load()
        model = adapter._model
        report["initial_model_loading_info"] = adapter.info().model_dump(mode="json")
        before = quantized_identity(model, bnb.nn.Linear4bit)
        report["quantized_weights_before_promotion"] = before
        require(before == previous_result["quantized_weights_before_promotion"], "Packed NF4 state differs from successful probe")
        torch.cuda.reset_peak_memory_stats()
        report["promotion"] = promote(model, bnb.nn.Linear4bit)
        require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()), "Parameters off GPU")
        after = quantized_identity(model, bnb.nn.Linear4bit)
        report["quantized_weights_after_promotion"] = after
        require(before == after, "Packed NF4 state changed during promotion")
        settings = precision_settings(torch)
        report["original_precision_settings"] = settings
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report["smoke_precision_settings"] = precision_settings(torch)
        report["checkpoint_generation_config"] = model.generation_config.to_dict()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report["post_promotion_memory"] = {"allocated_gib": round(torch.cuda.memory_allocated()/2**30, 3),
                                          "cuda_free_gib": round(free/2**30, 3), "cuda_total_gib": round(total/2**30, 3)}
        require(free >= 2**30, "Need >=1 GiB CUDA memory after promotion; no fallback")

        def root_before(mod, args, kwargs):
            require(observer.step < CAP, "Forward cap exceeded; stop without retry")
            observer.step += 1
            report["total_forward_calls_attempted"] += 1
            cache = kwargs.get("past_key_values")
            ids = kwargs.get("input_ids")
            report["forward_metadata"].append({"case_id": report["cases"][-1]["id"], "forward": observer.step,
                 "input_shape": None if ids is None else list(ids.shape), "use_cache": kwargs.get("use_cache"),
                 "cache_class": None if cache is None else type(cache).__name__,
                 "observed_cache_tensors": len(cache_values(cache))})
            observer.inspect("root_forward", "input", {"activation": args[:1], **kwargs})
            observer.inspect("kv_cache", "input", cache_values(cache))

        def root_after(mod, args, output):
            observer.inspect("root_forward", "output", output)
            observer.inspect("kv_cache", "output", cache_values(getattr(output, "past_key_values", None)))

        handles.append(model.register_forward_pre_hook(root_before, with_kwargs=True))
        handles.append(model.register_forward_hook(root_after))
        for name, module in model.named_modules():
            if name == "lm_head" or name.startswith("model.language_model"):
                def module_before(mod, args, kwargs, name=name):
                    observer.inspect(name, "input", {"activation": args[:1], **kwargs})
                def module_after(mod, args, output, name=name):
                    observer.inspect(name, "output", output)
                handles.append(module.register_forward_pre_hook(module_before, with_kwargs=True))
                handles.append(module.register_forward_hook(module_after))
        report["outcome"] = execute_cases(adapter, observer, report, persist)
    except NonFiniteObserved as exc:
        report.update(outcome="non_finite_detected", detail=str(exc))
    except UnexpectedPrecision as exc:
        report.update(outcome="unexpected_precision", detail=str(exc))
    except Exception as exc:
        report["outcome"] = "runtime_error"
        report["detail"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        stream.close()
        if model is not None and "quantized_weights_before_promotion" in report:
            try:
                report["quantized_weights_after_attempt"] = quantized_identity(model, bnb.nn.Linear4bit)
                report["packed_state_unchanged_after_attempt"] = (
                    report["quantized_weights_before_promotion"] == report["quantized_weights_after_attempt"])
                require(report["packed_state_unchanged_after_attempt"], "Packed NF4 state changed during attempt")
            except Exception as exc:
                report.update(outcome="runtime_error", fingerprint_error=f"{type(exc).__name__}: {exc}")
        if settings is not None:
            torch.backends.cuda.matmul.fp32_precision = settings["fp32_precision"]
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings["allow_fp16_reduced_precision_reduction"]
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), first_bad_activation=observer.first_bad,
                      first_wrong_dtype=observer.first_wrong_dtype, observed_tensor_events=observer.events,
                      final_allocated_gpu_memory_gib=round(torch.cuda.memory_allocated()/2**30, 3))
        save(DEST / "last_events.json", list(observer.trace))
        persist()
    print(f"FP32_SMOKE_OUTCOME: {report['outcome']}", flush=True)
    return int(report["outcome"] != "smoke_complete")


def run():
    require(not DEST.with_suffix(".zip").exists(), "Archive exists; preserve it, do not retry")
    claim(DEST)
    shutil.copyfile(PLAN, DEST / "protocol.md")
    shutil.copyfile(Path(__file__), DEST / "launcher.py")
    for name in HELPERS:
        shutil.copyfile(ROOT / "scripts" / name, DEST / name)
    code = None
    try:
        with (DEST / "smoke.log").open("w", encoding="utf-8") as log:
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
