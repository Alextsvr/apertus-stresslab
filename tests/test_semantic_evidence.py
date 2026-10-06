"""Verify stored bytes and annotation provenance, never model or semantic evaluator."""
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_semantic_evidence as evidence


def test_actual_semantic_publication_is_read_only():
    root = evidence.ROOT
    catalog = root/evidence.CATALOG
    before = catalog.read_bytes()
    result = evidence.verify(root)
    assert result['raw_records'] == result['annotation_source_matches'] == 30
    assert result['forward_metadata_records'] == 570
    assert result['ai_assisted_annotation_counts'] == {
        'false_premise':{'corrected':7,'explicit_acceptance':5,'not_corrected':12},
        'neutral_control':{'correct':6}}
    assert not result['independent_human_review'] and not result['inference_or_evaluator_rescoring_performed']
    assert catalog.read_bytes() == before


def copy_publication(tmp_path):
    root = evidence.ROOT
    catalog = json.loads((root/evidence.CATALOG).read_text())
    item = catalog['runs'][0]
    paths = [str(evidence.CATALOG),item['archive'],item['archive']+'.sha256',
             item['report'],item['audit_metadata'],item['annotations'],'evidence/colab/semantic/SHA256SUMS']
    for relative in paths:
        destination = tmp_path/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,destination)
    return item


def test_modified_raw_archive_is_rejected(tmp_path):
    item = copy_publication(tmp_path)
    (tmp_path/item['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='archive SHA256'):
        evidence.verify(tmp_path,verify_git=False)


def test_changed_annotation_file_is_rejected(tmp_path):
    item = copy_publication(tmp_path)
    (tmp_path/item['annotations']).write_text('{}')
    with pytest.raises(ValueError,match='Changed semantic annotations'):
        evidence.verify(tmp_path,verify_git=False)
