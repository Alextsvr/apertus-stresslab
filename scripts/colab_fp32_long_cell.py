"""Notebook helper for the current prepared Colab session; no dependency installs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

from google.colab import files, userdata

root = Path("/content/apertus-stresslab")
python = Path("/content/apertus-env/bin/python")
assert root.is_dir() and python.is_file(), "Нужна текущая подготовленная сессия Colab"
destination = root / "results/colab-fp32-generation-long-2026-10-06"
assert not destination.exists() and not destination.with_suffix(".zip").exists(), "Попытка уже существует — не запускай повторно"
expected = "e50e058df196207d0c9712f7ea4950a0168a39989b5596f800622a849d880b4a"
dependency = root / "results/evidence-dependencies" / (expected + ".zip")
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if not dependency.exists():
    source = root / "results/colab-fp32-generation-smoke-2026-10-06.zip"
    if not source.is_file():
        print("Выбери сохранённый colab-fp32-generation-smoke-2026-10-06.zip")
        upload_dir = Path(tempfile.mkdtemp(prefix="apertus-long-dependency-"))
        files.upload(target_dir=str(upload_dir))
        source = upload_dir / "colab-fp32-generation-smoke-2026-10-06.zip"
    assert source.is_file() and digest(source) == expected, "Архив предыдущей проверки не совпал"
    dependency.parent.mkdir(parents=True, exist_ok=True)
    with dependency.open("xb") as target:
        target.write(source.read_bytes())
assert digest(dependency) == expected, "Хеш зависимости не совпал"
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
process = subprocess.run([str(python), "-u", "scripts/colab_fp32_long.py", "--run"], cwd=root)
print("Код завершения:", process.returncode)
report_path = destination / "long_probe.json"
if report_path.is_file():
    report = json.loads(report_path.read_text())
    print(json.dumps({
        "outcome": report["outcome"], "detail": report.get("detail"),
        "total_forward_calls_attempted": report["total_forward_calls_attempted"],
        "observed_tensor_events": report.get("observed_tensor_events"),
        "first_bad_activation": report.get("first_bad_activation"),
        "first_wrong_dtype": report.get("first_wrong_dtype"),
        "packed_state_unchanged": report.get("packed_state_unchanged_after_attempt"),
        "cases": [{"id": c["id"], "response": c.get("generation", {}).get("text"),
                   "output_tokens": c.get("generation", {}).get("output_tokens"),
                   "stop_reason": c.get("stop_reason"), "outcome": c["outcome"],
                   "numerical_horizon_reached": c.get("numerical_horizon_reached"),
                   "output_quality": c.get("output_quality"),
                   "peak_gpu_memory_gib": c.get("generation", {}).get("peak_gpu_memory_gib")}
                  for c in report["cases"]],
    }, indent=2, ensure_ascii=False))
for suffix in (".zip", ".zip.sha256"):
    artifact = destination.with_suffix(suffix)
    if artifact.is_file():
        files.download(str(artifact))
