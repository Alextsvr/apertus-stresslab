"""One prospective observational NF4 forward; no generation or scoring."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from colab_quantization import ROOT, MODEL, REVISION, check, claim, git, require, save, seal, sha

DEST = ROOT / "results/colab-4bit-diagnostic-2026-10-06"
PLAN = ROOT / "docs/colab_4bit_diagnostic_protocol.md"
PROMPT = "Reply with exactly one word: hello."


class NonFiniteObserved(RuntimeError):
    pass


def tensors(value, path="value"):
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if "mask" not in str(key).lower():
                yield from tensors(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{path}[{index}]")


def stats(tensor):
    import torch
    t = tensor.detach()
    finite = torch.isfinite(t)
    result = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
              "nan": int(torch.isnan(t).sum().item()),
              "positive_inf": int(torch.isposinf(t).sum().item()),
              "negative_inf": int(torch.isneginf(t).sum().item())}
    values = t if bool(finite.all().item()) else t[finite]
    result.update(finite_min=float(values.min().item()) if values.numel() else None,
                  finite_max=float(values.max().item()) if values.numel() else None)
    return result


class Observer:
    def __init__(self):
        self.trace = deque(maxlen=100)
        self.first_bad = None
        self.events = 0

    def inspect(self, module, stage, value):
        for path, tensor in tensors(value):
            event = {"module": module, "stage": stage, "path": path, **stats(tensor)}
            self.events += 1
            self.trace.append(event)
            if event["nan"] or event["positive_inf"] or event["negative_inf"]:
                if self.first_bad is None:
                    self.first_bad = event
                raise NonFiniteObserved(f"Non-finite tensor at {module}: {stage} {path}")


def worker():
    # Download completed beforehand. No credentials or weights enter the archive.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    report = {"prompt": PROMPT, "forward_calls": 0, "generation_performed": False,
              "scoring_performed": False, "outcome": "runtime_error"}
    observer = Observer()
    handles = []
    parameter_report = {"floating_parameters_checked": 0, "quant_state_tensors_checked": 0,
                        "parameter_dtypes": {}, "xielu_parameters": {}, "first_bad": None}
    try:
        preflight = check()
        preflight.update(diagnostic_plan_sha256=sha(PLAN), diagnostic_launcher_sha256=sha(Path(__file__)))
        save(DEST / "preflight.json", preflight)
        for relative in ("scripts/colab_4bit_diagnostic.py", "docs/colab_4bit_diagnostic_protocol.md"):
            require(bool(git("ls-files", "--", relative)), f"Not committed: {relative}")
        snapshot = Path(Path("/content/apertus-pinned-snapshot.txt").read_text().strip())
        require(snapshot.name == REVISION, "Cached snapshot revision differs")
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        require(len(shards) == 6 and all((snapshot / name).is_file() for name in shards),
                "Pinned six-shard cache incomplete")
        report["cached_shards"] = shards
        set_seed(42)
        print("Loading cached NF4/FP16 model on T4", flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype="float16",
                                 quantization="4bit", device_map="cuda:0", cpu_offload=False)
        adapter.load()
        model = adapter._model
        require(all(p.device.type == "cuda" and p.device.index == 0 for p in model.parameters()),
                "Parameters off GPU; abort")
        report["model_info"] = adapter.info().model_dump(mode="json")
        dtypes = Counter()
        with torch.inference_mode():
            for name, parameter in model.named_parameters():
                dtypes[str(parameter.dtype)] += parameter.numel()
                if parameter.is_floating_point():
                    parameter_report["floating_parameters_checked"] += 1
                    if not bool(torch.isfinite(parameter).all().item()):
                        parameter_report["first_bad"] = {"name": name, **stats(parameter)}
                        raise NonFiniteObserved(f"Non-finite parameter: {name}")
                if any(part in name for part in ("alpha_p", "alpha_n")):
                    parameter_report["xielu_parameters"][name] = parameter.detach().float().cpu().tolist()
                quant = getattr(parameter, "quant_state", None)
                for prefix, state in (("quant_state", quant), ("quant_state.state2", getattr(quant, "state2", None))):
                    for field in ("absmax", "code", "offset"):
                        value = getattr(state, field, None)
                        if isinstance(value, torch.Tensor) and value.is_floating_point():
                            parameter_report["quant_state_tensors_checked"] += 1
                            if not bool(torch.isfinite(value).all().item()):
                                parameter_report["first_bad"] = {"name": f"{name}.{prefix}.{field}", **stats(value)}
                                raise NonFiniteObserved(f"Non-finite quantization state: {name}.{prefix}.{field}")
        parameter_report["parameter_dtypes"] = dict(dtypes)
        save(DEST / "parameter_checks.json", parameter_report)
        print("Loaded values finite; attaching language-path observers", flush=True)
        observed = []
        for name, module in model.named_modules():
            if name == "lm_head" or name.startswith("model.language_model"):
                observed.append({"name": name, "class": type(module).__name__})
                def before(mod, args, kwargs, name=name):
                    observer.inspect(name, "input", {"activation": args[:1], **kwargs})
                def after(mod, args, output, name=name):
                    observer.inspect(name, "output", output)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
        require(bool(observed), "No language modules matched; abort before forward")
        report["observed_modules"] = observed
        inputs = adapter._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False).to(adapter._input_device())
        report["input_ids"] = inputs["input_ids"].cpu().tolist()
        torch.cuda.reset_peak_memory_stats()
        report["forward_calls"] = 1
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
            observer.inspect("final_logits", "output", output.logits)
            scores = output.logits[0, -1]
            values, ids = torch.topk(scores, 5)
            report["top_tokens"] = [{"id": int(i), "raw_text": adapter._processor.decode([int(i)], skip_special_tokens=False),
                                     "score": float(v)} for i, v in zip(ids.tolist(), values.tolist())]
            unk = adapter._processor.tokenizer.unk_token_id
            report["unknown_token"] = None if unk is None else {
                "id": unk, "score": float(scores[unk].item()),
                "rank_with_ties": int((scores > scores[unk]).sum().item()) + 1}
        report["outcome"] = "one_forward_finite"
        report["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except NonFiniteObserved as exc:
        report.update(outcome="non_finite_detected", detail=str(exc))
    except Exception as exc:
        report.update(detail=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      first_bad_activation=observer.first_bad, observed_tensor_events=observer.events)
        save(DEST / "parameter_checks.json", parameter_report)
        save(DEST / "trace.json", list(observer.trace))
        save(DEST / "diagnostic.json", report)
    print(f"DIAGNOSTIC_OUTCOME: {report['outcome']}", flush=True)
    return int(report["outcome"] == "runtime_error")


def run():
    require(not DEST.with_suffix(".zip").exists(), "Archive already exists; preserve, do not retry")
    claim(DEST)
    shutil.copyfile(PLAN, DEST / "protocol.md")
    shutil.copyfile(Path(__file__), DEST / "launcher.py")
    code = None
    try:
        with (DEST / "diagnostic.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--worker"],
                                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace")
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
        save(DEST / "execution_status.json", {"worker_exit_code": code,
             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Evidence archive: {seal(DEST)}", flush=True)
    return 1 if code is None else code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.exit(worker() if args.worker else run())
