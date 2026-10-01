"""Best-effort collection of runtime environment metadata.

Nothing here may fail a run: every probe is wrapped and degrades to None/"unavailable".
torch is only inspected if it is already installed; it is never required.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

from stresslab.config import PROJECT_ROOT

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "bitsandbytes",
    "tokenizers",
    "safetensors",
    "pydantic",
)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def transformers_source() -> Optional[str]:
    """Return the VCS URL/commit transformers was installed from (PEP 610), if any."""
    try:
        dist = importlib_metadata.distribution("transformers")
        direct_url = dist.read_text("direct_url.json")
        return direct_url.strip() if direct_url else "pypi"
    except Exception:  # noqa: BLE001
        return None


def git_commit(repo_dir: Path = PROJECT_ROOT) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, timeout=5, check=False
        )
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return None


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch  # noqa: PLC0415 - optional heavy import
    except Exception:  # noqa: BLE001
        return info
    info["available"] = True
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info


def collect_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "transformers_source": transformers_source(),
        "torch": torch_info(),
        "git_commit": git_commit(),
        "in_colab": "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ,
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
