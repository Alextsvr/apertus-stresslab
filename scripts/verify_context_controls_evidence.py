"""Count fixed indicators from preserved annotations; never classify response text."""
from collections import Counter, defaultdict
from itertools import combinations

CONDITIONS=('bare','dates','attributes')
CATEGORIES=('corrected_explicit','corrected_implicit','explicit_acceptance','not_corrected','ambiguous','unassessable')
CONTROL_CATEGORIES=('correct','incorrect','ambiguous','unassessable')
CORRECTIONS={'corrected_explicit','corrected_implicit'}
FLAGS=('comparative_contradiction','date_or_milestone_mentioned','uses_supplied_dates',
       'unsupported_date_or_chronology','unsupported_causal_claim','entity_spelling_error',
       'incorrect_numeric_fact','misstates_supplied_fact','denies_false_premise_exists',
       'attribute_or_location_mentioned','uses_supplied_attributes','unsupported_attribute_claim')

def present(p,c):
    return p.get(c) is not None

def assessable(p,conditions):
    return all(present(p,c) and p[c]['label']!='unassessable' for c in conditions)

def contrast(pairs,left,right,labels):
    valid=[p for p in pairs if assessable(p,(left,right))]
    counts=Counter()
    for p in valid:
        a,b=(p[c]['label'] in labels for c in (left,right))
        counts['both' if a and b else 'only_left' if a else 'only_right' if b else 'neither']+=1
    table={k:counts[k] for k in ('both','only_left','only_right','neither')}
    return {'left_condition':left,'right_condition':right,'assessable_matched_pairs':len(valid),
            'missing_or_unassessable_pairs':len(pairs)-len(valid),'matched_2x2':table,
            'net_left_minus_right':table['only_left']-table['only_right']}

def paired_summary(triplets):
    false=[p for p in triplets if p['kind']=='false_premise']
    controls=[p for p in triplets if p['kind']=='neutral_control']
    def counts(group,c,categories):
        counter=Counter(p[c]['label'] for p in group if present(p,c))
        return {label:counter[label] for label in categories}
    def transitions(group,left,right,categories):
        return {a:{b:sum(present(p,left) and present(p,right) and p[left]['label']==a and p[right]['label']==b
                    for p in group) for b in categories} for a in categories}
    primary=contrast(false,'dates','attributes',{'explicit_acceptance'})
    primary.update(planned_false_premise_records_per_condition=24,
        available_records_per_condition={c:sum(present(p,c) for p in false) for c in CONDITIONS},
        assessable_records_per_condition={c:sum(assessable(p,(c,)) for p in false) for c in CONDITIONS},
        ambiguous_indicator_rule='Assessable ambiguous is zero for clear acceptance, not a confirmed rejection.')
    sensitivity=[p for p in false if assessable(p,('dates','attributes')) and
                 all(p[c]['label']!='ambiguous' for c in ('dates','attributes'))]
    pairwise={}
    for left,right in combinations(CONDITIONS,2):
        pairwise[left+'_vs_'+right]={
            'explicit_acceptance':contrast(false,left,right,{'explicit_acceptance'}),
            'combined_correction':contrast(false,left,right,CORRECTIONS),
            'false_premise_transition_matrix_left_rows_right_columns':transitions(false,left,right,CATEGORIES),
            'control_transition_matrix_left_rows_right_columns':transitions(controls,left,right,CONTROL_CATEGORIES),
            'identical_cleaned_response_pairs':sum(present(p,left) and present(p,right) and
                                                  p[left]['response']==p[right]['response'] for p in triplets)}
    flags=FLAGS
    return {
        'planned_raw_records':108,'available_raw_records':sum(present(p,c) for p in triplets for c in CONDITIONS),
        'matched_context_triplets':len(triplets),'false_premise_triplets':len(false),'control_triplets':len(controls),
        'annotation_counts':{c:{'false_premise':counts(false,c,CATEGORIES),
            'neutral_control':counts(controls,c,CONTROL_CATEGORIES)} for c in CONDITIONS},
        'primary_explicit_acceptance_dates_minus_attributes':primary,
        'prespecified_sensitivity_excluding_ambiguous_df_pairs':contrast(sensitivity,'dates','attributes',{'explicit_acceptance'}),
        'combined_corrections':{c:sum(present(p,c) and p[c]['label'] in CORRECTIONS for p in false) for c in CONDITIONS},
        'pairwise_comparisons':pairwise,
        'false_premise_label_triplet_counts':dict(Counter('|'.join(p[c]['label'] if present(p,c) else 'missing'
             for c in CONDITIONS) for p in false)),
        'control_label_triplet_counts':dict(Counter('|'.join(p[c]['label'] if present(p,c) else 'missing'
             for c in CONDITIONS) for p in controls)),
        'secondary_flag_counts':{c:{kind:{f:sum(present(p,c) and f in p[c]['secondary_flags'] for p in group)
            for f in flags} for kind,group in (('false_premise',false),('neutral_control',controls))} for c in CONDITIONS},
        'comparative_contradiction_ids':{c:[p[c]['id'] for p in false if present(p,c) and
            'comparative_contradiction' in p[c]['secondary_flags']] for c in CONDITIONS},
        'breakdowns':{field:{value:{c:counts([p for p in false if p[field]==value],c,CATEGORIES) for c in CONDITIONS}
            for value in sorted({p[field] for p in false})} for field in ('asserted_relation','variant','direction_state')},
        'base_pair_breakdown':{base:{'primary':contrast([p for p in false if p['base_pair_id']==base],
             'dates','attributes',{'explicit_acceptance'}),'annotation_counts':{c:counts([p for p in false
             if p['base_pair_id']==base],c,CATEGORIES) for c in CONDITIONS}} for base in sorted({p['base_pair_id'] for p in false})},
    }

import json
from pathlib import Path
import subprocess
import zipfile
from verify_colab_evidence import check_manifest,digest,require,safe_name

ROOT=Path(__file__).resolve().parents[1]
CATALOG=Path('evidence/colab/context_controls/catalog.json')

def check_annotations(annotations, records, rubric):
    require([a['id'] for a in annotations] == [r['id'] for r in records], 'Annotation coverage/order differs')
    for annotation, row in zip(annotations,records):
        for key in ('kind','base_pair_id','scenario_id','context_triplet_id','context_condition','direction_state','variant','subject','object','metric','subject_value','object_value','instruction','context','question','true_subject_relation'):
            require(annotation[key] == row[key], f'Annotation metadata differs: {key}')
        text = row['generation']['text']
        require(annotation['source_response'] == text and annotation['source_response_sha256'] == digest(text.encode())
                and annotation['source_raw_response_sha256'] == digest(row['generation']['raw_text'].encode()),
                'Annotation source response differs')
        categories = CATEGORIES if row['kind']=='false_premise' else CONTROL_CATEGORIES
        require(annotation['label'] in categories, 'Unknown annotation label')
        require(annotation['human_confirmed'] is False and annotation['truncated']==row['truncated'], 'Annotation provenance differs')
        require(isinstance(annotation['rationale'],str) and bool(annotation['rationale'].strip()), 'Missing annotation rationale')
        flags = annotation['secondary_flags']
        require(len(flags)==len(set(flags)) and set(flags).issubset(rubric['secondary_flags']), 'Unknown/duplicate annotation flags')
        require('uses_supplied_dates' not in flags or row['context_condition']=='dates', 'Only dates can use supplied dates')
        require('uses_supplied_attributes' not in flags or row['context_condition']=='attributes', 'Only attributes can use supplied attributes')


def check_triplets(pairs, annotations, records):
    grouped = defaultdict(list)
    lookup = {a['id']:a for a in annotations}
    for row in records:
        grouped[row['context_triplet_id']].append(row)
    require([p['context_triplet_id'] for p in pairs] == list(grouped), 'Matched pair coverage/order differs')
    for pair in pairs:
        rows = {r['context_condition']:r for r in grouped[pair['context_triplet_id']]}
        require(set(rows)==set(CONDITIONS), 'Matched condition missing')
        for condition, row in rows.items():
            for field in ('kind','base_pair_id','scenario_id','variant','direction_state','subject_value','object_value'):
                require(pair[field]==row[field], f'Matched metadata differs: {field}')
            require(pair['asserted_relation']==row.get('asserted_relation'), 'Matched direction differs')
            a = lookup[row['id']]
            expected = {'id':row['id'],'label':a['label'],'response':a['source_response'],
                        'response_sha256':a['source_response_sha256'],'secondary_flags':a['secondary_flags'],
                        'truncated':a['truncated']}
            require(pair[condition]==expected, f'Matched annotation source differs: {condition}')



def check_human_batch(human,item,annotations,seen):
    sources={a['id']:a for a in annotations}
    require(human['source_archive_sha256']==item['archive_sha256'] and
            human['source_ai_annotations_sha256']==item['annotations_sha256'] and
            human['independent_human_review'] is False and human['blinded'] is False and
            human['review_mode']=='selected, unblinded, AI-assisted human confirmation' and
            human['raw_chat_quote_included'] is False, 'Human confirmation provenance differs')
    require(len(human['confirmations'])==human['confirmed_records'], 'Human confirmation count differs')
    primary=flag_only=0
    for confirmation in human['confirmations']:
        case_id=confirmation['id']
        require(case_id in sources and case_id not in seen, 'Unknown/duplicate human-confirmed ID')
        source=sources[case_id]
        require(confirmation['source_ai_primary_label']==source['label'], 'Human source AI label differs')
        action=confirmation['action'];flags=confirmation['confirmed_secondary_flags']
        if action=='confirmed_presented_secondary_flags':
            require(confirmation['confirmed_primary_label'] is None and bool(flags), 'Flag-only human scope differs')
            flag_only+=1
        elif action=='confirmed_presented_primary_label':
            require(confirmation['confirmed_primary_label']==source['label'] and flags==[], 'Primary-only human scope differs')
            primary+=1
        else:
            raise ValueError('Unknown human confirmation action')
        require(len(flags)==len(set(flags)) and set(flags).issubset(source['secondary_flags']), 'Human-confirmed flags differ')
        for field in ('source_response','source_response_sha256','source_raw_response_sha256'):
            require(confirmation[field]==source[field], 'Human-confirmed response differs')
        seen.add(case_id)
    require(primary==human['confirmed_primary_label_records'] and flag_only==human['secondary_flags_only_records'],
            'Human scope totals differ')
    return seen


def verify(root=ROOT, *, verify_git=True):
    root = Path(root)
    catalog = json.loads((root/CATALOG).read_text(encoding='utf-8'))
    require(catalog['schema_version']==1 and len(catalog['runs'])==1, 'Unknown context catalog')
    item = catalog['runs'][0]
    path = root/safe_name(item['archive'])
    content = path.read_bytes()
    require(digest(content)==item['archive_sha256'] and len(content)==item['archive_bytes'], 'Context archive SHA256/size')
    require(path.with_suffix('.zip.sha256').read_text().split()==[item['archive_sha256'],path.name], 'Context sidecar differs')
    for field in ('report','annotations','paired_comparison','audit_metadata','receipt'):
        require(digest((root/safe_name(item[field])).read_bytes())==item[field+'_sha256'], f'Changed context {field}')
    with zipfile.ZipFile(path) as z:
        check_manifest(z,item['internal_files'])
        pre = json.loads(z.read('preflight.json'))
        result = json.loads(z.read('context_run.json'))
        execution = json.loads(z.read('execution_status.json'))
        cases = json.loads(z.read('cases.json'))['cases']
        rubric = json.loads(z.read('audit_rubric.json'))
        inputs = json.loads(z.read('tokenized_prompts.json'))
        reference = json.loads(z.read('tokenization_reference.json'))
        forwards = [json.loads(line) for line in z.read('forward_metadata.jsonl').splitlines()]
        require(pre['git_commit']==item['procedure_commit'], 'Context execution commit differs')
        for relative,member in item['registered_sources'].items():
            safe_name(relative)
            if verify_git:
                expected = subprocess.check_output(['git','show',item['procedure_commit']+':'+relative],cwd=root)
                require(z.read(member)==expected, f'Context registered source differs: {relative}')
        for key,member in (('context_plan_sha256','protocol.md'),('context_cases_sha256','cases.json'),
                           ('context_rubric_sha256','audit_rubric.json'),('context_launcher_sha256','launcher.py'),
                           ('context_token_reference_sha256','tokenization_reference.json')):
            require(pre[key]==digest(z.read(member)), 'Registered artifact hash differs')
        for relative,expected in pre['source_sha256'].items():
            require(digest((root/safe_name(relative)).read_bytes())==expected, 'Frozen current source changed')
            if verify_git:
                blob = subprocess.check_output(['git','show',item['procedure_commit']+':'+relative],cwd=root)
                require(digest(blob)==expected, 'Frozen registered source changed')
    require(inputs==reference['reference_inputs'] and set(inputs)=={c['id'] for c in cases}, 'Registered input arrays differ')
    require(reference['model_revision']==pre['model_revision'] and reference['cases_sha256']==pre['context_cases_sha256']
            and reference['weights_loaded'] is False and reference['model_forward_calls']==0
            and reference['created_before_model_responses'] is True and reference['historical_input_arrays_matched']==72
            and reference['historical_reference_archive_sha256']==item['baseline_archive_sha256'], 'Token registration provenance differs')
    for offset in range(0,len(cases),3):
        block=cases[offset:offset+3]
        require(len(block)==3 and len({c['context_triplet_id'] for c in block})==1, 'Adjacent triplet changed')
        lengths={c['context_condition']:len(inputs[c['id']][0]) for c in block}
        require(lengths['dates']==lengths['attributes']==lengths['bare']+20, 'Matched token lengths differ')
    require(result['registered_input_check']=={'matched_registered_input_arrays':108,
            'equal_token_length_date_attribute_pairs':36,'added_tokens_per_enriched_condition':20,
            'checked_before_first_forward':True}, 'Pre-forward registered input gate differs')
    for key,expected in item['stored_result'].items():
        require(result.get(key)==expected, f'Context stored result differs: {key}')
    require(result['scoring_performed'] is False and result['historical_suite_run'] is False, 'Unexpected scoring/historical rerun')
    require(execution['worker_exit_code']==0 and result['outcome']=='context_control_complete', 'Context completion differs')
    require(result['seed_per_prompt']==42 and result['max_new_tokens']==96 and result['max_input_tokens']==256, 'Generation settings differ')
    require(result['adapter_generation_config']=={'max_new_tokens':96,'do_sample':False,'temperature':None,
             'top_p':None,'enable_thinking':False}, 'Adapter config differs')
    require(result['context_control_precision_settings']=={'fp32_precision':'ieee','allow_fp16_reduced_precision_reduction':False},
            'Numerical settings differ')
    baseline = root/safe_name(item['baseline_archive'])
    require(digest(baseline.read_bytes())==pre['preceding_date_archive_sha256']==item['baseline_archive_sha256'], 'Baseline archive changed')
    with zipfile.ZipFile(baseline) as old:
        check_manifest(old,19)
        old_pre = json.loads(old.read('preflight.json'))
        old_run = json.loads(old.read('date_ablation_run.json'))
    for key in ('versions','all_packages','torch_cuda','source_sha256','dataset_sha256','model_revision','gpu'):
        require(pre[key]==old_pre[key], f'Context environment differs: {key}')
    driver = lambda p:p['gpu_identity'].partition(',')[2].strip()
    require(driver(pre)==driver(old_pre) and pre['gpu_compute_capability']==[7,5], 'GPU class/driver differs')
    weights = result['quantized_weights_before_promotion']
    require(len(weights)==217 and weights==result['quantized_weights_after_promotion']==result['quantized_weights_after_attempt']
            ==old_run['quantized_weights_before_promotion'], 'Packed state changed')
    records = result['cases']
    require(len(cases)==len(records)==result['completed_records']==result['planned_records']==108 and result['unattempted_records']==0,
            'Context raw coverage differs')
    require(len({r['id'] for r in records})==108 and set(inputs)=={r['id'] for r in records}, 'Context IDs/input coverage differs')
    for fixed,row in zip(cases,records):
        require(all(row.get(k)==v for k,v in fixed.items()), 'Context prompt metadata changed')
        require(row['input_ids']==inputs[row['id']] and len(row['input_ids'])==1 and 0<len(row['input_ids'][0])<=256, 'Context input IDs differ')
        require(row['generate_calls']==1 and row['outcome']=='assessable_output', 'Single-shot technical contract differs')
        require(row['supplied_generate_kwargs']=={'max_new_tokens':96,'do_sample':False}, 'Generate kwargs changed')
        ids = row['generated_ids']
        require(row['generation']['output_tokens']==len(ids)==row['forward_calls_attempted'] and 0<len(ids)<=96
                and row['generation']['input_tokens']==len(row['input_ids'][0]), 'Token/forward counts differ')
        require(row['stop_reason']=='eos' and not row['truncated'] and ids[-1] in result['eos_token_ids']
                and result['unknown_token_id'] not in ids and row['generation']['text'].strip(), 'Context technical output differs')
        require(all(row['output_quality'].values()), 'Technical quality flag differs')
        captured = [f for f in forwards if f['case_id']==row['id']]
        require([f['forward'] for f in captured]==list(range(1,len(ids)+1)) and all(f['use_cache'] is True for f in captured)
                and captured[0]['input_shape']==[1,len(row['input_ids'][0])]
                and all(f['observed_cache_tensors']==64 and f['input_shape']==[1,1] for f in captured[1:]), 'Forward/cache metadata differs')
    require(len(forwards)==sum(r['forward_calls_attempted'] for r in records)==result['total_forward_calls_attempted'], 'Total forwards differ')
    for field in ('observed_tensor_events','observed_cache_events'):
        require(sum(r[field] for r in records)==result[field], 'Observed event totals differ')
    require(result['first_bad_activation'] is None and result['first_wrong_dtype'] is None
            and result['packed_state_unchanged_after_attempt'] is True, 'Numerical outcome differs')
    audit = json.loads((root/item['annotations']).read_text(encoding='utf-8'))
    require(audit['source_archive_sha256']==item['archive_sha256'] and audit['analysis_source']=='Codex AI-assisted post-hoc semantic audit'
            and audit['independent_human_review'] is False and audit['human_confirmed_records']==0
            and audit['prospective_rubric_sha256']==pre['context_rubric_sha256'], 'Audit provenance differs')
    annotations = audit['annotations']
    check_annotations(annotations,records,rubric)
    comparison = json.loads((root/item['paired_comparison']).read_text(encoding='utf-8'))
    check_triplets(comparison['triplets'],annotations,records)
    summary = paired_summary(comparison['triplets'])
    require(comparison['summary']==summary and item['annotation_counts']==summary['annotation_counts'], 'Stored paired counts differ')
    receipt=json.loads((root/item['receipt']).read_text(encoding='utf-8'))
    integrity=json.loads((root/item['audit_metadata']).read_text(encoding='utf-8'))
    for record in (receipt,integrity):
        require(record['archive_sha256']==item['archive_sha256'] and record['procedure_commit']==item['procedure_commit']
                and record['hashes_saved_before_semantic_annotations'] is True and record['responses_read_before_anchor'] is True
                and record['sender_checksum_received_and_matched'] is True
                and record['semantic_evaluator_invoked'] is False, 'Receipt/integrity provenance differs')
    for field,actual in (('records',len(records)),('forward_calls',len(forwards)),
                        ('tensor_events',result['observed_tensor_events']),('cache_events',result['observed_cache_events'])):
        require(integrity[field]==actual, 'Integrity count differs')
    human_ids=set();primary_confirmed=flag_only_confirmed=0
    for batch in item.get('human_confirmation_batches',[]):
        content=(root/safe_name(batch['path'])).read_bytes()
        require(digest(content)==batch['sha256'], 'Human confirmation artifact changed')
        human=json.loads(content)
        for field in ('confirmed_records','confirmed_primary_label_records','secondary_flags_only_records'):
            require(human[field]==batch[field], 'Catalog human scope count differs')
        check_human_batch(human,item,annotations,human_ids)
        primary_confirmed+=human['confirmed_primary_label_records']
        flag_only_confirmed+=human['secondary_flags_only_records']
    for line in (root/'evidence/colab/context_controls/SHA256SUMS').read_text().splitlines():
        expected,relative = line.split('  ',1)
        require(digest((root/safe_name(relative)).read_bytes())==expected, 'Context publication manifest differs')
    return {'archive_sha256':'match','internal_hashes':'20/20 match','registered_sources_checked_against_git':verify_git,
            'raw_records':len(records),'forward_metadata_records':len(forwards),'annotation_source_matches':len(annotations),
            'registered_input_arrays_matched':len(inputs),'equal_length_date_attribute_triplets':36,
            'ai_assisted_annotation_counts':summary['annotation_counts'],
            'primary_explicit_acceptance_dates_minus_attributes':summary['primary_explicit_acceptance_dates_minus_attributes'],
            'combined_corrections':summary['combined_corrections'],'original_ai_annotation_human_confirmed_records':0,
            'selected_ai_assisted_human_confirmations':len(human_ids),'distinct_human_reviewed_records':len(human_ids),
            'confirmed_primary_label_records':primary_confirmed,'secondary_flags_only_records':flag_only_confirmed,
            'independent_human_review':False,'inference_or_evaluator_rescoring_performed':False}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
