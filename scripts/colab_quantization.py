"""Prospective Colab T4 study. --check never loads weights or generates text."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "cd4e1438937c23ae7ff4919645d1ae00c3b29912"
REVISION = "a411d838600baf0e3635a3daf66fb7c55fc97bb6"
FORK = "3797303dda74844e3d1f8977ff5518bb91f818b4"
MODEL = "swiss-ai/Apertus-v1.5-8B"
STUDY = ROOT / "results" / "colab-t4-quantization-2026-10-05"
PROTOCOL = ROOT / "docs" / "colab_quantization_protocol.md"
EXPECTED = {"torch": "2.11.0+cu130", "transformers": "5.14.0.dev0",
            "accelerate": "1.15.0", "bitsandbytes": "0.50.2",
            "tokenizers": "0.22.2", "numpy": "2.1.3"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def check_vram(free, total):
    """Use CUDA free memory, not the nominal capacity displayed by nvidia-smi."""
    require(0 <= free <= total and total > 0, "Invalid CUDA memory readings")
    require(free / 2**30 >= 13.0,
            f"Need >=13 GiB free VRAM; CUDA reports {free / 2**30:.3f} GiB free "
            f"of {total / 2**30:.3f} GiB total")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def claim(path):
    """A failed or partial attempt may never be silently overwritten or resumed."""
    path.mkdir(parents=True, exist_ok=False)


def seal(folder):
    """Hash existing files, then export only this study, never credentials/cache/weights."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    manifest = "".join(f"{sha(p)}  {p.relative_to(folder).as_posix()}\n" for p in files)
    (folder / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in [*files, folder / "SHA256SUMS"]:
            z.write(path, path.relative_to(folder).as_posix())
    archive.with_suffix(".zip.sha256").write_text(f"{sha(archive)}  {archive.name}\n", encoding="utf-8")
    return archive


def check():
    # Imports are intentionally deferred: unit tests and --help need no GPU.
    import psutil
    import torch
    from verify_evidence import verify

    require(not git("status", "--porcelain", "--untracked-files=no"), "Tracked files changed")
    require(not git("diff", BASE, "--", "src", "data"), "Frozen source or dataset differs")
    for relative in ("scripts/colab_quantization.py", "docs/colab_quantization_protocol.md"):
        require(bool(git("ls-files", "--", relative)), f"Not committed: {relative}")
    require(sys.version_info[:3] == (3, 13, 15), "Python version changed; amend protocol first")
    versions = {k: metadata.version(k) for k in EXPECTED}
    require(versions == EXPECTED, f"Unexpected versions: {versions}")
    source = json.loads(metadata.distribution("transformers").read_text("direct_url.json") or "{}")
    require(source.get("vcs_info", {}).get("commit_id") == FORK, "Wrong transformers fork commit")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Need exactly one CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    require("T4" in props.name and (props.major, props.minor) == (7, 5), "GPU differs from protocol")
    free, total = torch.cuda.mem_get_info()
    check_vram(free, total)
    ram = psutil.virtual_memory().available / 2**30
    require(ram >= 8.0, "Need >=8 GiB available RAM before loading")
    identity = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    packages = sorted([d.metadata.get("Name", ""), d.version] for d in metadata.distributions())
    disk = shutil.disk_usage(ROOT).free / 2**30
    require(disk >= 22.0, "Need >=22 GiB free disk before each condition")
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git("rev-parse", "HEAD"), "protocol_sha256": sha(PROTOCOL),
            "model_revision": REVISION, "versions": versions, "all_packages": packages,
            "transformers_source": source, "gpu": props.name, "gpu_identity": identity,
            "torch_cuda": torch.version.cuda, "free_vram_gib": round(free / 2**30, 3),
            "cuda_total_vram_gib": round(total / 2**30, 3),
            "available_ram_gib": round(ram, 3), "free_disk_gib": round(disk, 3),
            "evidence_check": verify(ROOT),
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p)
                              for p in sorted((ROOT / "src").rglob("*.py"))},
            "dataset_sha256": sha(ROOT / "data/test_cases/false_premise_heldout.jsonl"),
            "weights_loaded": False}


def check_same_environment(before, after):
    for key in ("git_commit", "protocol_sha256", "model_revision", "versions", "all_packages",
                "gpu_identity", "torch_cuda", "source_sha256", "dataset_sha256"):
        require(before[key] == after[key], f"Environment changed between conditions: {key}")


def condition(mode):
    import torch
    from stresslab.cases import load_cases, suite_path
    from stresslab.config import SUITE_VERSIONS
    from stresslab.environment import collect_environment
    from stresslab.evaluators import evaluator_for
    from stresslab.models import ApertusAdapter
    from stresslab.runner import run_cases
    from stresslab.schemas import GenerationConfig

    before = json.loads((STUDY / "preflight.json").read_text(encoding="utf-8"))
    current = check()
    check_same_environment(before, current)
    destination = STUDY / mode
    claim(destination)
    save(destination / "before_load.json", current)

    class FullGPUAdapter(ApertusAdapter):
        def load(self):
            super().load()
            require(all(p.device.type == "cuda" and p.device.index == 0
                        for p in self._model.parameters()), "Parameters off GPU; abort before inference")

        def generate(self, *args, **kwargs):
            self.completed = getattr(self, "completed", 0) + 1
            print(f"{mode}: generating case {self.completed}/30", flush=True)
            return super().generate(*args, **kwargs)

    # Explicit placement prevents auto CPU/disk fallback. No evaluator/source changes.
    adapter = FullGPUAdapter(model_id=MODEL, revision=REVISION, dtype="float16",
                             quantization=mode, device_map="cuda:0", cpu_offload=False)
    cases = load_cases(suite_path("false_premise_heldout"))
    require(len(cases) == 30, "Expected all 30 cases")
    environment = collect_environment()
    environment["prospective_study"] = current
    result = run_cases(cases=cases, adapter=adapter,
                       gen_config=GenerationConfig(max_new_tokens=96, do_sample=False, enable_thinking=False),
                       seed=42, results_root=destination,
                       command=f"python scripts/colab_quantization.py --condition {mode}",
                       suite="false_premise_heldout", test_suite_version=SUITE_VERSIONS["false_premise_heldout"],
                       environment=environment,
                       notes="New Colab quantization study on previously observed prompts; not new held-out validation.",
                       evaluator=evaluator_for("false_premise_heldout"))
    require(len(result.results) == 30 and result.errors == 0,
            f"Incomplete/error condition: {len(result.results)} records, {result.errors} errors")
    print(f"{mode}: 30 records saved; no responses printed", flush=True)


def run():
    require(not STUDY.exists() and not STUDY.with_suffix(".zip").exists(),
            "Attempt already exists. Do not delete/retry; preserve it and amend protocol.")
    preflight = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--check-json"],
                                       cwd=ROOT, text=True)
    # The check process exits before loading: no extra parent CUDA context occupies VRAM.
    preflight = json.loads(preflight)
    claim(STUDY)
    save(STUDY / "preflight.json", preflight)
    (STUDY / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    (STUDY / "launcher.py").write_bytes(Path(__file__).read_bytes())
    exit_codes = {}
    try:
        for mode in ("4bit", "8bit"):
            print(f"Starting {mode}. Weights download/load may take several minutes.", flush=True)
            with (STUDY / f"{mode}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                            "--condition", mode], cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    exit_codes[mode] = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            if exit_codes[mode] != 0:
                break
    finally:
        save(STUDY / "execution_status.json", {"exit_codes": exit_codes,
             "complete": exit_codes == {"4bit": 0, "8bit": 0},
             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        archive = seal(STUDY)
        print(f"Evidence archive: {archive}", flush=True)
    require(exit_codes == {"4bit": 0, "8bit": 0}, "Study incomplete; preserve archive, do not retry")
    print("STUDY_COMPLETE: 60 records preserved and hashed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Read-only resource/provenance check; no weights")
    group.add_argument("--run", action="store_true", help="Execute the preregistered pair once")
    group.add_argument("--check-json", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--condition", choices=("4bit", "8bit"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check or args.check_json:
        report = check()
        if args.check_json:
            print(json.dumps(report))
        else:
            print(json.dumps({k: v for k, v in report.items()
                              if k not in {"all_packages", "source_sha256", "transformers_source"}}, indent=2))
            print("PROTOCOL_PREFLIGHT_OK")
    elif args.condition:
        condition(args.condition)
    else:
        run()


if __name__ == "__main__":
    main()
