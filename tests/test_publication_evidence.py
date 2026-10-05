"""Checks for evidence corruption and accidental inclusion in release archives."""
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_evidence import AUDIT, EvidenceError, RAW_NAMES, RUN, verify
from package_evidence import TOP_FILES, payload_files


@pytest.fixture
def evidence_copy(tmp_path):
    for relative in [*(RUN / name for name in RAW_NAMES), AUDIT / 'integrity.json',
                     AUDIT / 'SHA256SUMS', AUDIT / 'semantic_annotations.json',
                     Path('data/test_cases/false_premise_heldout.jsonl')]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def test_published_evidence_is_readable_without_inference():
    result = verify(ROOT)
    assert result['raw_sha256'] == '4/4 match'
    assert result['annotation_source_matches'] == 30
    assert result['inference_or_rescoring_performed'] is False


def test_tampered_raw_bytes_are_rejected(evidence_copy):
    path = evidence_copy / RUN / 'results.jsonl'
    path.write_bytes(path.read_bytes().replace(b'Dravic', b'Dravix', 1))
    with pytest.raises(EvidenceError, match='SHA256 mismatch'):
        verify(evidence_copy)


def test_missing_raw_file_is_rejected(evidence_copy):
    (evidence_copy / RUN / 'metadata.json').unlink()
    with pytest.raises(FileNotFoundError):
        verify(evidence_copy)


def test_changed_prompt_is_rejected_even_if_raw_hashes_match(evidence_copy):
    path = evidence_copy / 'data/test_cases/false_premise_heldout.jsonl'
    path.write_text(path.read_text(encoding='utf-8').replace('Why does', 'Why would', 1), encoding='utf-8')
    with pytest.raises(EvidenceError, match='Prompt mismatch'):
        verify(evidence_copy)


def test_packager_excludes_private_and_unrelated_files(tmp_path):
    for relative in [*TOP_FILES, *(RUN / name for name in RAW_NAMES),
                     'src/stresslab/__init__.py', 'docs/report.md', 'scripts/verify_evidence.py',
                     '.env', 'docs/.env', 'src/__pycache__/example.pyc', 'results/other/results.jsonl',
                     RUN / 'extra.json', 'docs/model.safetensors', '.git/config']:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    names = {path.relative_to(tmp_path).as_posix() for path in payload_files(tmp_path)}
    assert 'docs/report.md' in names
    assert (RUN / 'results.jsonl').as_posix() in names
    assert not names.intersection({'.env', 'docs/.env', 'src/__pycache__/example.pyc',
        'results/other/results.jsonl', (RUN / 'extra.json').as_posix(), 'docs/model.safetensors', '.git/config'})
