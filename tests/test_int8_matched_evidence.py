"""Protect raw bytes and source provenance; no model or semantic evaluator."""
import copy
import json
from pathlib import Path
import shutil
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_int8_matched_evidence as evidence


def artifacts():
    item = json.loads((evidence.ROOT / evidence.CATALOG).read_text())['runs'][0]
    comparison = json.loads((evidence.ROOT / item['paired_comparison']).read_text(encoding='utf-8'))
    audit = json.loads((evidence.ROOT / item['annotations']).read_text(encoding='utf-8'))
    baseline = json.loads((evidence.ROOT / item['baseline_annotations']).read_text(encoding='utf-8'))
    return item, comparison, audit, baseline


def test_actual_matched_evidence_is_read_only():
    item, comparison, _, _ = artifacts()
    before = (evidence.ROOT / item['archive']).read_bytes()
    result = evidence.verify()
    assert result['raw_records'] == result['matched_input_arrays'] == result['annotation_source_matches'] == 30
    assert result['forward_metadata_records'] == 610
    assert result['ai_assisted_annotation_counts']['false_premise'] == {
        'corrected': 8, 'explicit_acceptance': 4, 'not_corrected': 12}
    assert result['paired_correction_counts'] == {
        'int8_only_corrected': 3, 'nf4_only_corrected': 2, 'both_corrected': 5, 'neither_corrected': 14}
    assert (evidence.ROOT / item['archive']).read_bytes() == before
    assert not result['inference_or_evaluator_rescoring_performed']
    assert evidence.paired_summary(comparison['pairs']) == comparison['summary']


def test_changed_received_archive_is_rejected(tmp_path):
    item, _, _, _ = artifacts()
    for relative in (str(evidence.CATALOG), item['archive']):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence.ROOT / relative, target)
    (tmp_path / item['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive SHA256'):
        evidence.verify(tmp_path, verify_git=False)


def test_changed_pair_cannot_silently_relabel_baseline():
    item, comparison, audit, baseline = artifacts()
    import zipfile
    with zipfile.ZipFile(evidence.ROOT / item['archive']) as z:
        records = json.loads(z.read('int8_run.json'))['cases']
    changed = copy.deepcopy(comparison['pairs'])
    changed[6]['nf4_fp32']['label'] = 'corrected'
    with pytest.raises(ValueError, match='Paired source differs: nf4_fp32'):
        evidence.check_pairs(changed, baseline['annotations'], audit['annotations'], records)


def test_changed_pair_cannot_substitute_a_response():
    item, comparison, audit, baseline = artifacts()
    import zipfile
    with zipfile.ZipFile(evidence.ROOT / item['archive']) as z:
        records = json.loads(z.read('int8_run.json'))['cases']
    changed = copy.deepcopy(comparison['pairs'])
    changed[0]['int8_fp16']['response'] = 'wrong entity'
    with pytest.raises(ValueError, match='Paired source differs: int8_fp16'):
        evidence.check_pairs(changed, baseline['annotations'], audit['annotations'], records)
