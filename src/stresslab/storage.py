"""Result persistence: timestamped run directories, JSONL results, JSON metadata.

Layout:
    results/<YYYY-MM-DD_HHMMSS>[_<n>]/
        metadata.json
        results.jsonl

Existing run directories are never overwritten.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from stresslab.schemas import RunMetadata, TestResult

METADATA_FILE = "metadata.json"
RESULTS_FILE = "results.jsonl"


def new_run_id(now: Optional[datetime] = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")


def create_run_dir(results_root: Path, run_id: Optional[str] = None) -> Path:
    """Create a fresh run directory. Adds a numeric suffix instead of reusing an existing one."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    base = run_id or new_run_id()
    candidate = results_root / base
    n = 1
    while True:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            n += 1
            candidate = results_root / f"{base}_{n}"


def write_metadata(run_dir: Path, metadata: RunMetadata) -> Path:
    path = Path(run_dir) / METADATA_FILE
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_metadata(run_dir: Path) -> RunMetadata:
    path = Path(run_dir) / METADATA_FILE
    return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def append_result(run_dir: Path, result: TestResult) -> Path:
    """Append one record. Flushed per line so partial runs are never lost."""
    path = Path(run_dir) / RESULTS_FILE
    line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()
    return path


def iter_results(run_dir: Path) -> Iterator[TestResult]:
    path = Path(run_dir) / RESULTS_FILE
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield TestResult.model_validate_json(line)


def read_results(run_dir: Path) -> list[TestResult]:
    return list(iter_results(run_dir))
