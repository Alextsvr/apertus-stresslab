"""Meaningful corruption/coverage checks for the stdlib Colab evidence verifier."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_colab_evidence as evidence


def make_archive(path, payload, *, omit=None, extra=None):
    manifest = "".join(f"{hashlib.sha256(v).hexdigest()}  {n}\n" for n, v in payload.items() if n != omit)
    with zipfile.ZipFile(path, "w") as z:
        for n, v in payload.items():
            z.writestr(n, v)
        if extra:
            z.writestr(*extra)
        z.writestr("SHA256SUMS", manifest)


def test_internal_manifest_detects_unlisted_file(tmp_path):
    p = tmp_path / "archive.zip"
    make_archive(p, {"result.json": b"{}"}, extra=("hidden.txt", b"unexpected"))
    with zipfile.ZipFile(p) as z, pytest.raises(ValueError, match="Incomplete"):
        evidence.check_manifest(z, 1)


def test_internal_manifest_detects_corruption(tmp_path):
    p = tmp_path / "archive.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("result.json", b"changed")
        z.writestr("SHA256SUMS", f"{hashlib.sha256(b'original').hexdigest()}  result.json\n")
    with zipfile.ZipFile(p) as z, pytest.raises(ValueError, match="Internal SHA256"):
        evidence.check_manifest(z, 1)


@pytest.mark.parametrize("name", ["../secret", "/absolute", "C:/secret", "folder\\secret", "a/../secret"])
def test_unsafe_paths_are_rejected_without_extraction(name):
    with pytest.raises(ValueError, match="Unsafe"):
        evidence.safe_name(name)


def fixture_catalog(tmp_path):
    folder = tmp_path / "evidence/colab"
    folder.mkdir(parents=True)
    report, metadata = folder / "report.md", folder / "metadata.json"
    report.write_bytes(b"Report\n")
    metadata.write_bytes(b"{}")
    archive = folder / "test.zip"
    make_archive(archive, {"result.json": b'{"outcome":"complete"}', "launcher.py": b"# source\n"})
    h = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{h}  test.zip\n")
    item = {"id": "test", "archive": "evidence/colab/test.zip", "archive_bytes": archive.stat().st_size,
            "archive_sha256": h, "report": "evidence/colab/report.md", "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "audit_metadata": "evidence/colab/metadata.json", "audit_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "internal_files": 2, "registered_sources": {"scripts/test.py": "launcher.py"}, "procedure_commit": "fixture",
            "result_member": "result.json", "stored_result": {"outcome": "complete"}}
    (folder / "catalog.json").write_text(json.dumps({"schema_version": 1, "attempts": [item]}))
    (folder / "SHA256SUMS").write_text("")
    return item, folder


def test_valid_fixture_is_read_only_and_does_not_claim_git_check(tmp_path):
    _, folder = fixture_catalog(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = evidence.verify(tmp_path, verify_git=False, include_phase3b=False)
    assert result["attempts_verified"] == 1 and not result["inference_or_rescoring_performed"]
    assert not result["attempts"][0]["registered_sources_checked_against_git"]
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_corrupted_archive_stops_before_trusting_its_results(tmp_path):
    _, folder = fixture_catalog(tmp_path)
    (folder / "test.zip").write_bytes(b"not the archive")
    with pytest.raises(ValueError, match="Archive SHA256"):
        evidence.verify(tmp_path, verify_git=False, include_phase3b=False)


def test_changed_report_is_detected(tmp_path):
    _, folder = fixture_catalog(tmp_path)
    (folder / "report.md").write_bytes(b"Altered claim\n")
    with pytest.raises(ValueError, match="Changed report"):
        evidence.verify(tmp_path, verify_git=False, include_phase3b=False)


def test_packager_excludes_untracked_local_audits(tmp_path):
    import subprocess
    from package_evidence import TOP_FILES, payload_files
    from verify_evidence import RAW_NAMES, RUN
    required = [*TOP_FILES, *(RUN / n for n in RAW_NAMES), "docs/public.md", "docs/audits/local/private.md"]
    for relative in required:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--", "docs/public.md"], cwd=tmp_path, check=True)
    selected = {p.relative_to(tmp_path).as_posix() for p in payload_files(tmp_path)}
    assert "docs/public.md" in selected and "docs/audits/local/private.md" not in selected


def test_actual_publication_checks_eight_archives_without_model():
    root = Path(__file__).resolve().parents[1]
    result = evidence.verify(root)
    assert result["attempts_verified"] == 8 and result["original_phase3b"]["raw_sha256"] == "4/4 match"
    assert not result["inference_or_rescoring_performed"]
