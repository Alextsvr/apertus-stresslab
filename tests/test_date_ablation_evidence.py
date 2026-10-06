"""Corruption/provenance guards for the paired date evidence; no inference."""
import copy
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_date_ablation_evidence as evidence


def artifacts():
    item=json.loads((evidence.ROOT/evidence.CATALOG).read_text())['runs'][0]
    audit=json.loads((evidence.ROOT/item['annotations']).read_text(encoding='utf-8'))
    comparison=json.loads((evidence.ROOT/item['paired_comparison']).read_text(encoding='utf-8'))
    with zipfile.ZipFile(evidence.ROOT/item['archive']) as z:
        records=json.loads(z.read('date_ablation_run.json'))['cases']
        rubric=json.loads(z.read('audit_rubric.json'))
    return item,audit,comparison,records,rubric


def test_actual_publication_is_verified_without_writing_or_inference():
    item,_,comparison,_,_=artifacts()
    before=(evidence.ROOT/item['archive']).read_bytes()
    result=evidence.verify()
    assert result['raw_records']==result['annotation_source_matches']==72
    assert result['forward_metadata_records']==1088
    assert result['primary_explicit_acceptance']['matched_2x2']=={
        'accepts_both':0,'only_dates_present':7,'only_dates_absent':1,'neither_explicit_acceptance':16}
    assert result['paired_combined_correction_counts']=={
        'both_corrected':8,'only_dates_present_corrected':1,'only_dates_absent_corrected':13,'neither_corrected':2}
    assert comparison['summary']==evidence.paired_summary(comparison['pairs'])
    assert not result['independent_human_review'] and not result['inference_or_evaluator_rescoring_performed']
    assert before==(evidence.ROOT/item['archive']).read_bytes()


def test_changed_received_archive_cannot_be_accepted(tmp_path):
    item,*_=artifacts()
    for relative in (str(evidence.CATALOG),item['archive']):
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(evidence.ROOT/relative,path)
    (tmp_path/item['archive']).write_bytes(b'changed evidence')
    with pytest.raises(ValueError,match='Date archive SHA256'):
        evidence.verify(tmp_path,verify_git=False)


def test_annotation_cannot_substitute_response_text():
    _,audit,_,records,rubric=artifacts()
    changed=copy.deepcopy(audit['annotations'])
    changed[24]['source_response']='The premise is false.'
    with pytest.raises(ValueError,match='Annotation source response differs'):
        evidence.check_annotations(changed,records,rubric)


def test_pair_cannot_silently_swap_date_labels():
    _,audit,comparison,records,_=artifacts()
    changed=copy.deepcopy(comparison['pairs'])
    pair=next(p for p in changed if p['date_pair_id']=='DATE-004-HIGH-A')
    pair['dates_present']['label']='corrected_explicit'
    with pytest.raises(ValueError,match='Matched annotation source differs: dates_present'):
        evidence.check_pairs(changed,audit['annotations'],records)


def test_d0_cannot_claim_to_use_supplied_dates():
    _,audit,_,records,rubric=artifacts()
    changed=copy.deepcopy(audit['annotations'])
    row=next(a for a in changed if a['date_condition']=='dates_absent')
    row['secondary_flags']=['uses_supplied_dates']
    with pytest.raises(ValueError,match='D0 cannot use supplied dates'):
        evidence.check_annotations(changed,records,rubric)


def test_ambiguous_is_not_acceptance_and_sensitivity_excludes_its_pair():
    _,_,comparison,_,_=artifacts()
    pair=copy.deepcopy(next(p for p in comparison['pairs'] if p['kind']=='false_premise'))
    pair['dates_absent']['label']='ambiguous'
    pair['dates_present']['label']='explicit_acceptance'
    result=evidence.paired_summary([pair])
    assert result['primary_explicit_acceptance']['net_only_dates_present_minus_only_dates_absent']==1
    assert result['prespecified_sensitivity_excluding_ambiguous_pairs']['assessable_matched_pairs']==0
    assert result['false_premise_transition_matrix_d0_rows_d1_columns']['ambiguous']['explicit_acceptance']==1


def test_unassessable_pair_is_excluded_without_imputing_a_rejection():
    _,_,comparison,_,_=artifacts()
    pair=copy.deepcopy(next(p for p in comparison['pairs'] if p['kind']=='false_premise'))
    pair['dates_absent']['label']='unassessable'
    pair['dates_present']['label']='explicit_acceptance'
    result=evidence.paired_summary([pair])
    assert result['primary_explicit_acceptance']['missing_or_unassessable_pairs']==1
    assert result['primary_explicit_acceptance']['assessable_matched_pairs']==0
    assert result['annotation_counts']['dates_present']['false_premise']['explicit_acceptance']==1
    assert result['primary_explicit_acceptance']['assessable_records_per_condition']=={'dates_absent':0,'dates_present':1}


def test_packager_rejects_unrelated_date_archive_path(tmp_path):
    from package_evidence import payload_files
    catalog=tmp_path/'evidence/colab/date_ablation/catalog.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'runs':[{'archive':'evidence/colab/semantic/unrelated.zip'}]}))
    with pytest.raises(ValueError,match='Unsafe date-study archive path'):
        payload_files(tmp_path)
