"""Create dist/apertus-stresslab-bundle.zip with everything needed to run on Colab.

Usage (from the project root):
    python scripts/make_bundle.py

Excludes .venv, results, caches and git data. Contains no secrets.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "src",
    "data",
    "tests",
    "scripts",
    "pyproject.toml",
    "requirements.txt",
    "requirements-inference.txt",
    "README.md",
    "LICENSE",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", "results", ".git", "dist"}


def main() -> None:
    out = ROOT / "dist" / "apertus-stresslab-bundle.zip"
    out.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(ROOT)
                if SKIP_PARTS.intersection(rel.parts) or f.suffix == ".pyc" or "egg-info" in str(rel):
                    continue
                zf.write(f, Path("apertus-stresslab") / rel)
                count += 1
    print(f"Wrote {out} ({count} files)")


if __name__ == "__main__":
    main()
