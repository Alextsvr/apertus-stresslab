"""Evidence integrity and paired arithmetic; no classifier, weights or model forward."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_qwen_context_evidence as verifier
from qwen_comparison_summary import cross_model_summary
from verify_context_controls_evidence import check_annotations,CORRECTIONS


def artifacts():
    root=verifier.ROOT
    doc=root/'docs/colab_qwen_context/2026-10-07'
    new=json.loads((doc/'semantic_annotations.json').read_text(encoding='utf-8'))
    old=json.loads((root/'docs/colab_context_controls/2026-10-07/semantic_annotations.json').read_text(encoding='utf-8'))
    comparison=json.loads((doc/'cross_model_comparison.json').read_text(encoding='utf-8'))
    return new,old,comparison


def test_returned_evidence_full_integrity_and_no_human_scope_inferred():
    result=verifier.verify()
    assert result['raw_records']==result['annotation_source_matches']==result['registered_input_arrays_matched']==108
    assert result['forward_metadata_records']==3347
    assert result['human_confirmed_qwen_records']==0 and result['inference_or_evaluator_rescoring_performed'] is False


def test_preserved_annotations_pair_by_exact_ids_and_same_user_text():
    new,old,c=artifacts()
    verifier.check_cross_pairs(c['pairs'],new['annotations'],old['annotations'])
    assert new['human_confirmed_records']==0 and not any(a['human_confirmed'] for a in new['annotations'])
    assert len({p['id'] for p in c['pairs']})==108


@pytest.mark.parametrize('field',['context_condition','subject_value','question'])
def test_cross_model_prompt_or_ground_truth_mismatch_rejected(field):
    new,old,c=artifacts();old=copy.deepcopy(old)
    old['annotations'][0][field]='changed'
    with pytest.raises(ValueError):verifier.check_cross_pairs(c['pairs'],new['annotations'],old['annotations'])


def test_cross_model_label_change_cannot_keep_stored_pairs():
    new,old,c=artifacts();c=copy.deepcopy(c)
    c['pairs'][36]['apertus']['label']='ambiguous'
    with pytest.raises(ValueError,match='annotation source'):verifier.check_cross_pairs(c['pairs'],new['annotations'],old['annotations'])


def test_pair_order_change_rejected():
    new,old,c=artifacts();pairs=copy.deepcopy(c['pairs']);pairs[36],pairs[37]=pairs[37],pairs[36]
    with pytest.raises(ValueError,match='coverage/order'):verifier.check_cross_pairs(pairs,new['annotations'],old['annotations'])


def test_all_prior_acceptances_pair_with_correction_but_two_noncorrections_remain():
    _,_,c=artifacts();false=[p for p in c['pairs'] if p['kind']=='false_premise']
    accepted=[p for p in false if p['apertus']['label']=='explicit_acceptance']
    assert len(accepted)==11 and all(p['qwen']['label'] in CORRECTIONS for p in accepted)
    assert {p['id'] for p in false if p['qwen']['label']=='not_corrected'}=={'CTX-001-HIGH-D-B','CTX-003-HIGH-F-A'}
    assert all(p['apertus']['label']=='not_corrected' for p in false if p['qwen']['label']=='not_corrected')


def test_fixed_sign_and_transition_totals():
    _,_,c=artifacts();s=cross_model_summary(c['pairs'])
    assert s==c['summary']
    for condition,delta in (('bare',-1),('dates',-6),('attributes',-4)):
        cell=s['by_condition'][condition]
        assert cell['explicit_acceptance_qwen_minus_apertus']['net_left_minus_right']==delta
        matrix=cell['false_premise_transitions_qwen_rows_apertus_columns']
        assert sum(sum(row.values()) for row in matrix.values())==24
        controls=cell['control_transitions_qwen_rows_apertus_columns']
        assert sum(sum(row.values()) for row in controls.values())==12


def test_missing_output_reduces_paired_denominator_without_imputation():
    _,_,c=artifacts();pairs=copy.deepcopy(c['pairs'])
    candidate=next(p for p in pairs if p['kind']=='false_premise' and p['apertus']['label']=='explicit_acceptance')
    condition=candidate['context_condition'];candidate['qwen']=None
    s=cross_model_summary(pairs)
    assert s['available_matched_records']==107
    cell=s['by_condition'][condition]['explicit_acceptance_qwen_minus_apertus']
    assert cell['assessable_matched_pairs']==23 and cell['missing_or_unassessable_pairs']==1
    assert sum(cell['matched_2x2'].values())==23


def test_ambiguous_pair_is_visible_and_excluded_in_prespecified_sensitivity():
    _,_,c=artifacts();pairs=copy.deepcopy(c['pairs'])
    candidate=next(p for p in pairs if p['kind']=='false_premise');condition=candidate['context_condition']
    candidate['qwen']['label']='ambiguous'
    cell=cross_model_summary(pairs)['by_condition'][condition]
    assert cell['explicit_acceptance_qwen_minus_apertus']['assessable_matched_pairs']==24
    assert cell['sensitivity_excluding_ambiguous']['assessable_matched_pairs']==23


def test_implicit_boundary_labels_and_date_flags_remain_separate_from_acceptance():
    new,_,_=artifacts();rows={a['id']:a for a in new['annotations']}
    implicit={'CTX-001-HIGH-F-A','CTX-005-HIGH-F-A','CTX-005-LOW-F-B'}
    assert {a['id'] for a in rows.values() if a['label']=='corrected_implicit'}==implicit
    assert rows['CTX-001-HIGH-D-B']['secondary_flags']==['date_or_milestone_mentioned','uses_supplied_dates']
    assert not any('comparative_contradiction' in a['secondary_flags'] for a in rows.values())
