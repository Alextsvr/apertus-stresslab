"""Study safety checks; no GPU, model, downloads or rescoring."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_quantization as study


def test_claim_rejects_existing_attempt_and_preserves_it(tmp_path):
    folder = tmp_path / "attempt"
    study.claim(folder)
    raw = folder / "results.jsonl"
    raw.write_bytes(b"original evidence\n")
    with pytest.raises(FileExistsError):
        study.claim(folder)
    assert raw.read_bytes() == b"original evidence\n"


def test_seal_hashes_exact_bytes_and_excludes_neighbor_secrets(tmp_path):
    folder = tmp_path / "attempt"
    folder.mkdir()
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    raw = folder / "results.jsonl"
    payload = b'{"answer":"test"}\r\n'
    raw.write_bytes(payload)
    archive = study.seal(folder)
    assert raw.read_bytes() == payload
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"results.jsonl", "SHA256SUMS"}
        assert z.read("results.jsonl") == payload
        assert hashlib.sha256(payload).hexdigest() in z.read("SHA256SUMS").decode()


def test_environment_change_is_rejected():
    keys = ("git_commit", "protocol_sha256", "model_revision", "versions", "all_packages",
            "gpu_identity", "torch_cuda", "source_sha256", "dataset_sha256")
    before = {k: "same" for k in keys}
    study.check_same_environment(before, dict(before))
    for key in keys:
        after = dict(before, **{key: "changed"})
        with pytest.raises(RuntimeError, match=key):
            study.check_same_environment(before, after)


def test_failed_first_condition_does_not_launch_second_and_is_archived(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    folder = root / "results" / "study"
    protocol = root / "protocol.md"
    protocol.write_text("fixed protocol", encoding="utf-8")
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "STUDY", folder)
    monkeypatch.setattr(study, "PROTOCOL", protocol)
    monkeypatch.setattr(study.subprocess, "check_output", lambda *a, **k: '{}')
    modes = []

    class FailedProcess:
        stdout = iter(["load failed\n"])

        def __init__(self, command, **kwargs):
            modes.append(command[-1])

        def wait(self, **kwargs):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(study.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="incomplete"):
        study.run()
    assert modes == ["4bit"]
    assert folder.with_suffix(".zip").exists()
    assert json.loads((folder / "execution_status.json").read_text())["complete"] is False
