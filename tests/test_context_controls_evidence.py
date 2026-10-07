"""Evidence corruption, provenance and prespecified missingness guards; no inference."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_context_controls_evidence as evidence


def artifacts():
    item=json.loads((evidence.ROOT/evidence.CATALOG).read_text())['runs'][0]
    audit=json.loads((evidence.ROOT/item['annotations']).read_text(encoding='utf-8'))
    comparison=json.loads((evidence.ROOT/item['paired_comparison']).read_text(encoding='utf-8'))
    with zipfile.ZipFile(evidence.ROOT/item['archive']) as z:
        records=json.loads(z.read('context_run.json'))['cases']
        rubric=json.loads(z.read('audit_rubric.json'))
    return item,audit,comparison,records,rubric


def test_actual_evidence_and_fixed_endpoint_without_inference_or_writing():
    item,_,comparison,_,_=artifacts()
    before=(evidence.ROOT/item['archive']).read_bytes()
    result=evidence.verify()
    assert result['raw_records']==result['annotation_source_matches']==result['registered_input_arrays_matched']==108
    assert result['forward_metadata_records']==1597
    assert result['primary_explicit_acceptance_dates_minus_attributes']['matched_2x2']=={
        'both':1,'only_left':5,'only_right':3,'neither':15}
    assert result['primary_explicit_acceptance_dates_minus_attributes']['net_left_minus_right']==2
    assert result['combined_corrections']=={'bare':20,'dates':10,'attributes':3}
    assert comparison['summary']==evidence.paired_summary(comparison['triplets'])
    assert not result['independent_human_review'] and result['original_ai_annotation_human_confirmed_records']==0
    assert result['selected_ai_assisted_human_confirmations']==4
    assert result['confirmed_primary_label_records']==2 and result['secondary_flags_only_records']==2
    assert not result['inference_or_evaluator_rescoring_performed']
    assert before==(evidence.ROOT/item['archive']).read_bytes()


def test_received_archive_corruption_is_rejected(tmp_path):
    item,*_=artifacts()
    for relative in (str(evidence.CATALOG),item['archive']):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(evidence.ROOT/relative,path)
    (tmp_path/item['archive']).write_bytes(b'changed evidence')
    with pytest.raises(ValueError,match='Context archive SHA256'):
        evidence.verify(tmp_path,verify_git=False)


def test_annotation_cannot_substitute_response_or_factual_context():
    _,audit,_,records,rubric=artifacts()
    changed=copy.deepcopy(audit['annotations']);changed[36]['source_response']='The premise is false.'
    with pytest.raises(ValueError,match='Annotation source response differs'):
        evidence.check_annotations(changed,records,rubric)
    changed=copy.deepcopy(audit['annotations']);changed[36]['context']='Alternative context.'
    with pytest.raises(ValueError,match='Annotation metadata differs: context'):
        evidence.check_annotations(changed,records,rubric)


def test_triplet_cannot_swap_condition_label():
    _,audit,comparison,records,_=artifacts()
    changed=copy.deepcopy(comparison['triplets'])
    pair=next(p for p in changed if p['context_triplet_id']=='CTX-001-HIGH-A')
    pair['dates']['label']='corrected_explicit'
    with pytest.raises(ValueError,match='Matched annotation source differs: dates'):
        evidence.check_triplets(changed,audit['annotations'],records)


@pytest.mark.parametrize('condition,flag,message',[
    ('bare','uses_supplied_dates','Only dates'),
    ('dates','uses_supplied_attributes','Only attributes')])
def test_invented_content_cannot_claim_supplied_fact_use(condition,flag,message):
    _,audit,_,records,rubric=artifacts()
    changed=copy.deepcopy(audit['annotations'])
    next(a for a in changed if a['context_condition']==condition)['secondary_flags']=[flag]
    with pytest.raises(ValueError,match=message):evidence.check_annotations(changed,records,rubric)


def test_missing_bare_does_not_drop_primary_dates_attributes_pair():
    _,_,comparison,_,_=artifacts();changed=copy.deepcopy(comparison['triplets'])
    next(p for p in changed if p['kind']=='false_premise')['bare']=None
    summary=evidence.paired_summary(changed)
    assert summary['primary_explicit_acceptance_dates_minus_attributes']['assessable_matched_pairs']==24
    assert summary['primary_explicit_acceptance_dates_minus_attributes']['net_left_minus_right']==2
    assert summary['pairwise_comparisons']['bare_vs_dates']['explicit_acceptance']['assessable_matched_pairs']==23
    assert summary['available_raw_records']==107


def test_missing_attribute_is_not_imputed_as_rejection():
    _,_,comparison,_,_=artifacts();changed=copy.deepcopy(comparison['triplets'])
    next(p for p in changed if p['context_triplet_id']=='CTX-001-HIGH-A')['attributes']=None
    summary=evidence.paired_summary(changed);primary=summary['primary_explicit_acceptance_dates_minus_attributes']
    assert primary['assessable_matched_pairs']==23 and primary['missing_or_unassessable_pairs']==1
    assert primary['net_left_minus_right']==1
    assert primary['available_records_per_condition']=={'bare':24,'dates':24,'attributes':23}


def test_ambiguity_is_not_acceptance_and_prespecified_sensitivity_excludes_pair():
    _,_,comparison,_,_=artifacts();changed=copy.deepcopy(comparison['triplets'])
    next(p for p in changed if p['context_triplet_id']=='CTX-001-HIGH-A')['attributes']['label']='ambiguous'
    summary=evidence.paired_summary(changed)
    assert summary['primary_explicit_acceptance_dates_minus_attributes']['net_left_minus_right']==2
    assert summary['prespecified_sensitivity_excluding_ambiguous_df_pairs']['net_left_minus_right']==1
    assert summary['prespecified_sensitivity_excluding_ambiguous_df_pairs']['assessable_matched_pairs']==23


def test_false_human_signoff_cannot_be_added_to_ai_annotations():
    _,audit,_,records,rubric=artifacts();changed=copy.deepcopy(audit['annotations'])
    changed[0]['human_confirmed']=True
    with pytest.raises(ValueError,match='Annotation provenance differs'):
        evidence.check_annotations(changed,records,rubric)


def test_packager_rejects_unrelated_context_archive_path(tmp_path):
    from package_evidence import payload_files
    path=tmp_path/'evidence/colab/context_controls/catalog.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'runs':[{'archive':'evidence/colab/date_ablation/unrelated.zip'}]}))
    with pytest.raises(ValueError,match='Unsafe context-control archive path'):payload_files(tmp_path)


def test_verifier_has_no_model_or_semantic_evaluator_dependency():
    tree=ast.parse(Path(evidence.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports.extend(alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names)
    assert all(not name.startswith(('torch','transformers','stresslab')) for name in imports)


def human_artifacts():
    item,audit,*_=artifacts()
    human=json.loads((evidence.ROOT/item['human_confirmation_batches'][0]['path']).read_text())
    return item,audit,human


def test_contradiction_only_replies_cannot_be_expanded_to_primary_acceptance_confirmation():
    item,audit,human=human_artifacts();changed=copy.deepcopy(human)
    changed['confirmations'][0]['confirmed_primary_label']='explicit_acceptance'
    with pytest.raises(ValueError,match='Flag-only human scope differs'):
        evidence.check_human_batch(changed,item,audit['annotations'],set())


def test_actual_primary_only_implicit_confirmation_and_flag_only_scopes():
    item,audit,human=human_artifacts()
    assert evidence.check_human_batch(human,item,audit['annotations'],set())=={
        'CTX-006-LOW-F-B','CTX-002-HIGH-F-B','CTX-004-LOW-F-B'}
    assert human['confirmations'][2]['confirmed_primary_label']=='corrected_implicit'
    assert all(c['confirmed_primary_label'] is None for c in human['confirmations'][:2])


def test_duplicate_human_review_cannot_inflate_distinct_rows():
    item,audit,human=human_artifacts()
    with pytest.raises(ValueError,match='Unknown/duplicate human-confirmed ID'):
        evidence.check_human_batch(human,item,audit['annotations'],{'CTX-006-LOW-F-B'})


def test_human_review_cannot_substitute_response_or_claim_independence():
    item,audit,human=human_artifacts();changed=copy.deepcopy(human)
    changed['confirmations'][0]['source_response']='Different answer'
    with pytest.raises(ValueError,match='Human-confirmed response differs'):
        evidence.check_human_batch(changed,item,audit['annotations'],set())
    changed=copy.deepcopy(human);changed['independent_human_review']=True
    with pytest.raises(ValueError,match='Human confirmation provenance differs'):
        evidence.check_human_batch(changed,item,audit['annotations'],set())


def test_bare_acceptance_human_confirmation_is_primary_only_and_source_bound():
    item,audit,*_=artifacts()
    human=json.loads((evidence.ROOT/item['human_confirmation_batches'][1]['path']).read_text())
    assert evidence.check_human_batch(human,item,audit['annotations'],set())=={'CTX-005-HIGH-B-B'}
    confirmation=human['confirmations'][0]
    assert confirmation['confirmed_primary_label']=='explicit_acceptance'
    assert confirmation['confirmed_secondary_flags']==[]
    changed=copy.deepcopy(human)
    changed['confirmations'][0]['confirmed_secondary_flags']=['denies_false_premise_exists']
    with pytest.raises(ValueError,match='Primary-only human scope differs'):
        evidence.check_human_batch(changed,item,audit['annotations'],set())


def test_human_confirmations_across_batches_have_four_distinct_rows_and_preserve_scope():
    item,audit,*_=artifacts();seen=set();primary=flags_only=0
    for batch in item['human_confirmation_batches']:
        human=json.loads((evidence.ROOT/batch['path']).read_text())
        evidence.check_human_batch(human,item,audit['annotations'],seen)
        primary+=human['confirmed_primary_label_records'];flags_only+=human['secondary_flags_only_records']
    assert len(seen)==4 and primary==flags_only==2
