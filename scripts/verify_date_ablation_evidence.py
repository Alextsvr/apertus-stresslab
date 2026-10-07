"""Verify preserved date-study bytes, provenance and stored labels; no semantic scoring."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import zipfile

from verify_colab_evidence import check_manifest, digest, require, safe_name

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path('evidence/colab/date_ablation/catalog.json')
CONDITIONS = ('dates_absent', 'dates_present')
CATEGORIES = ('corrected_explicit', 'corrected_implicit', 'explicit_acceptance',
              'not_corrected', 'ambiguous', 'unassessable')
CONTROL_CATEGORIES = ('correct', 'incorrect', 'ambiguous', 'unassessable')
CORRECTIONS = {'corrected_explicit', 'corrected_implicit'}


def endpoint(pairs):
    """Count a predeclared indicator from stored labels, never from response text."""
    counts = Counter()
    for p in pairs:
        d0, d1 = (p[c]['label'] == 'explicit_acceptance' for c in CONDITIONS)
        counts['accepts_both' if d0 and d1 else 'only_dates_present' if d1
               else 'only_dates_absent' if d0 else 'neither_explicit_acceptance'] += 1
    table = {key: counts[key] for key in ('accepts_both', 'only_dates_present',
             'only_dates_absent', 'neither_explicit_acceptance')}
    return {'assessable_matched_pairs': len(pairs), 'matched_2x2': table,
            'net_only_dates_present_minus_only_dates_absent': table['only_dates_present']-table['only_dates_absent']}


def paired_summary(pairs):
    false = [p for p in pairs if p['kind'] == 'false_premise']
    controls = [p for p in pairs if p['kind'] == 'neutral_control']
    assessable = [p for p in false if all(p[c]['label'] != 'unassessable' for c in CONDITIONS)]
    sensitivity = [p for p in assessable if all(p[c]['label'] != 'ambiguous' for c in CONDITIONS)]
    correction_counts = Counter()
    for p in assessable:
        d0, d1 = (p[c]['label'] in CORRECTIONS for c in CONDITIONS)
        correction_counts['both_corrected' if d0 and d1 else 'only_dates_present_corrected' if d1
                          else 'only_dates_absent_corrected' if d0 else 'neither_corrected'] += 1
    def counts(group, condition, categories):
        counter = Counter(p[condition]['label'] for p in group)
        return {label: counter[label] for label in categories}
    matrix = {a: {b:sum(p['dates_absent']['label'] == a and p['dates_present']['label'] == b for p in false)
                  for b in CATEGORIES} for a in CATEGORIES}
    flags = sorted({flag for p in pairs for c in CONDITIONS for flag in p[c]['secondary_flags']})
    base = {}
    for name in sorted({p['base_pair_id'] for p in false}):
        group = [p for p in false if p['base_pair_id'] == name]
        base[name] = {**endpoint([p for p in group if p in assessable]),
                      'annotation_counts': {c: counts(group,c,CATEGORIES) for c in CONDITIONS}}
    return {
        'planned_raw_records':72, 'available_raw_records':len(pairs)*2,
        'matched_date_pairs':len(pairs),'false_premise_pairs':len(false),'control_pairs':len(controls),
        'annotation_counts':{c:{'false_premise':counts(false,c,CATEGORIES),
                                'neutral_control':counts(controls,c,CONTROL_CATEGORIES)} for c in CONDITIONS},
        'primary_explicit_acceptance':{
            'planned_false_premise_records_per_condition':24,
            'available_records_per_condition':{c:len(false) for c in CONDITIONS},
            'assessable_records_per_condition':{c:sum(p[c]['label']!='unassessable' for p in false) for c in CONDITIONS},
            'missing_or_unassessable_pairs':len(false)-len(assessable),
            'ambiguous_indicator_rule':'Assessable ambiguous is not clear explicit_acceptance; zero does not establish rejection.',
            **endpoint(assessable)},
        'prespecified_sensitivity_excluding_ambiguous_pairs':endpoint(sensitivity),
        'false_premise_transition_categories':list(CATEGORIES),
        'false_premise_transition_matrix_d0_rows_d1_columns':matrix,
        'paired_combined_correction_counts':{k:correction_counts[k] for k in (
            'both_corrected','only_dates_present_corrected','only_dates_absent_corrected','neither_corrected')},
        'combined_corrections':{c:sum(p[c]['label'] in CORRECTIONS for p in false) for c in CONDITIONS},
        'control_transition_matrix_d0_rows_d1_columns':{a:{b:sum(p['dates_absent']['label']==a and p['dates_present']['label']==b
                for p in controls) for b in CONTROL_CATEGORIES} for a in CONTROL_CATEGORIES},
        'secondary_flag_counts':{c:{'false_premise':{f:sum(f in p[c]['secondary_flags'] for p in false) for f in flags},
                                   'neutral_control':{f:sum(f in p[c]['secondary_flags'] for p in controls) for f in flags}}
                                 for c in CONDITIONS},
        'comparative_contradiction_ids':{c:[p[c]['id'] for p in false if 'comparative_contradiction' in p[c]['secondary_flags']]
                                        for c in CONDITIONS},
        'breakdowns':{field:{value:{c:counts([p for p in false if p[field]==value],c,CATEGORIES) for c in CONDITIONS}
                            for value in sorted({p[field] for p in false})}
                      for field in ('asserted_relation','variant','direction_state')},
        'base_pair_breakdown':base,
        'identical_cleaned_response_pairs':sum(p['dates_absent']['response']==p['dates_present']['response'] for p in pairs),
    }


def check_annotations(annotations, records, rubric):
    require([a['id'] for a in annotations] == [r['id'] for r in records], 'Annotation coverage/order differs')
    for annotation, row in zip(annotations,records):
        for key in ('kind','base_pair_id','scenario_id','date_pair_id','date_condition','direction_state','variant'):
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
        require('uses_supplied_dates' not in flags or row['date_condition']=='dates_present', 'D0 cannot use supplied dates')


def check_pairs(pairs, annotations, records):
    grouped = defaultdict(list)
    lookup = {a['id']:a for a in annotations}
    for row in records:
        grouped[row['date_pair_id']].append(row)
    require([p['date_pair_id'] for p in pairs] == list(grouped), 'Matched pair coverage/order differs')
    for pair in pairs:
        rows = {r['date_condition']:r for r in grouped[pair['date_pair_id']]}
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



def check_human_batch(human, item, annotations, seen):
    sources = {a['id']:a for a in annotations}
    require(human['source_archive_sha256']==item['archive_sha256'] and
            human['source_ai_annotations_sha256']==item['annotations_sha256'] and
            human['independent_human_review'] is False and human['blinded'] is False and
            human['review_mode']=='selected, unblinded, AI-assisted human confirmation' and
            human['raw_chat_quote_included'] is False, 'Human confirmation provenance differs')
    require(len(human['confirmations'])==human['confirmed_records'], 'Human confirmation count differs')
    for confirmation in human['confirmations']:
        case_id = confirmation['id']
        require(case_id in sources and case_id not in seen, 'Unknown/duplicate human-confirmed ID')
        source = sources[case_id]
        action = confirmation['action']
        require(action in {'confirmed_presented_label_and_flag','confirmed_presented_primary_label'} and
                (bool(confirmation['confirmed_secondary_flags']) if action=='confirmed_presented_label_and_flag'
                 else confirmation['confirmed_secondary_flags']==[]) and
                confirmation['confirmed_primary_label']==source['label'] and
                len(confirmation['confirmed_secondary_flags'])==len(set(confirmation['confirmed_secondary_flags'])) and
                set(confirmation['confirmed_secondary_flags']).issubset(source['secondary_flags']), 'Human-confirmed scope differs')
        for field in ('source_response','source_response_sha256','source_raw_response_sha256'):
            require(confirmation[field]==source[field], 'Human-confirmed response differs')
        seen.add(case_id)
    return seen


def verify(root=ROOT, *, verify_git=True):
    root = Path(root)
    catalog = json.loads((root/CATALOG).read_text(encoding='utf-8'))
    require(catalog['schema_version']==1 and len(catalog['runs'])==1, 'Unknown date catalog')
    item = catalog['runs'][0]
    path = root/safe_name(item['archive'])
    content = path.read_bytes()
    require(digest(content)==item['archive_sha256'] and len(content)==item['archive_bytes'], 'Date archive SHA256/size')
    require(path.with_suffix('.zip.sha256').read_text().split()==[item['archive_sha256'],path.name], 'Date sidecar differs')
    for field in ('report','annotations','paired_comparison','audit_metadata'):
        require(digest((root/safe_name(item[field])).read_bytes())==item[field+'_sha256'], f'Changed date {field}')
    with zipfile.ZipFile(path) as z:
        check_manifest(z,item['internal_files'])
        pre = json.loads(z.read('preflight.json'))
        result = json.loads(z.read('date_ablation_run.json'))
        execution = json.loads(z.read('execution_status.json'))
        cases = json.loads(z.read('cases.json'))['cases']
        rubric = json.loads(z.read('audit_rubric.json'))
        inputs = json.loads(z.read('tokenized_prompts.json'))
        forwards = [json.loads(line) for line in z.read('forward_metadata.jsonl').splitlines()]
        require(pre['git_commit']==item['procedure_commit'], 'Date execution commit differs')
        for relative,member in item['registered_sources'].items():
            safe_name(relative)
            if verify_git:
                expected = subprocess.check_output(['git','show',item['procedure_commit']+':'+relative],cwd=root)
                require(z.read(member)==expected, f'Date registered source differs: {relative}')
        for key,member in (('date_plan_sha256','protocol.md'),('date_cases_sha256','cases.json'),
                           ('date_rubric_sha256','audit_rubric.json'),('date_launcher_sha256','launcher.py')):
            require(pre[key]==digest(z.read(member)), 'Registered artifact hash differs')
        for relative,expected in pre['source_sha256'].items():
            require(digest((root/safe_name(relative)).read_bytes())==expected, 'Frozen current source changed')
            if verify_git:
                blob = subprocess.check_output(['git','show',item['procedure_commit']+':'+relative],cwd=root)
                require(digest(blob)==expected, 'Frozen registered source changed')
    for key,expected in item['stored_result'].items():
        require(result.get(key)==expected, f'Date stored result differs: {key}')
    require(result['scoring_performed'] is False and result['historical_suite_run'] is False, 'Unexpected scoring/historical rerun')
    require(execution['worker_exit_code']==0 and result['outcome']=='date_ablation_complete', 'Date completion differs')
    require(result['seed_per_prompt']==42 and result['max_new_tokens']==96 and result['max_input_tokens']==256, 'Generation settings differ')
    require(result['adapter_generation_config']=={'max_new_tokens':96,'do_sample':False,'temperature':None,
             'top_p':None,'enable_thinking':False}, 'Adapter config differs')
    require(result['date_ablation_precision_settings']=={'fp32_precision':'ieee','allow_fp16_reduced_precision_reduction':False},
            'Numerical settings differ')
    baseline = root/safe_name(item['baseline_archive'])
    require(digest(baseline.read_bytes())==pre['preceding_semantic_archive_sha256']==item['baseline_archive_sha256'], 'Baseline archive changed')
    with zipfile.ZipFile(baseline) as old:
        check_manifest(old,18)
        old_pre = json.loads(old.read('preflight.json'))
        old_run = json.loads(old.read('semantic_run.json'))
    for key in ('versions','all_packages','torch_cuda','source_sha256','dataset_sha256','model_revision','gpu'):
        require(pre[key]==old_pre[key], f'Date environment differs: {key}')
    driver = lambda p:p['gpu_identity'].partition(',')[2].strip()
    require(driver(pre)==driver(old_pre) and pre['gpu_compute_capability']==[7,5], 'GPU class/driver differs')
    weights = result['quantized_weights_before_promotion']
    require(len(weights)==217 and weights==result['quantized_weights_after_promotion']==result['quantized_weights_after_attempt']
            ==old_run['quantized_weights_before_promotion'], 'Packed state changed')
    records = result['cases']
    require(len(cases)==len(records)==result['completed_records']==result['planned_records']==72 and result['unattempted_records']==0,
            'Date raw coverage differs')
    require(len({r['id'] for r in records})==72 and set(inputs)=={r['id'] for r in records}, 'Date IDs/input coverage differs')
    for fixed,row in zip(cases,records):
        require(all(row.get(k)==v for k,v in fixed.items()), 'Date prompt metadata changed')
        require(row['input_ids']==inputs[row['id']] and len(row['input_ids'])==1 and 0<len(row['input_ids'][0])<=256, 'Date input IDs differ')
        require(row['generate_calls']==1 and row['outcome']=='assessable_output', 'Single-shot technical contract differs')
        ids = row['generated_ids']
        require(row['generation']['output_tokens']==len(ids)==row['forward_calls_attempted'] and 0<len(ids)<=96
                and row['generation']['input_tokens']==len(row['input_ids'][0]), 'Token/forward counts differ')
        require(row['stop_reason']=='eos' and not row['truncated'] and ids[-1] in result['eos_token_ids']
                and result['unknown_token_id'] not in ids and row['generation']['text'].strip(), 'Date technical output differs')
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
            and audit['prospective_rubric_sha256']==pre['date_rubric_sha256'], 'Audit provenance differs')
    annotations = audit['annotations']
    check_annotations(annotations,records,rubric)
    comparison = json.loads((root/item['paired_comparison']).read_text(encoding='utf-8'))
    check_pairs(comparison['pairs'],annotations,records)
    summary = paired_summary(comparison['pairs'])
    require(comparison['summary']==summary and item['annotation_counts']==summary['annotation_counts'], 'Stored paired counts differ')
    human_ids = set()
    for batch in item.get('human_confirmation_batches', []):
        content = (root/safe_name(batch['path'])).read_bytes()
        require(digest(content)==batch['sha256'], 'Human confirmation artifact changed')
        human = json.loads(content)
        require(human['confirmed_records']==batch['confirmed_records'], 'Catalog human confirmation count differs')
        check_human_batch(human,item,annotations,human_ids)
    for line in (root/'evidence/colab/date_ablation/SHA256SUMS').read_text().splitlines():
        expected,relative = line.split('  ',1)
        require(digest((root/safe_name(relative)).read_bytes())==expected, 'Date publication manifest differs')
    return {'archive_sha256':'match','internal_hashes':'19/19 match','registered_sources_checked_against_git':verify_git,
            'raw_records':len(records),'forward_metadata_records':len(forwards),'annotation_source_matches':len(annotations),
            'ai_assisted_annotation_counts':summary['annotation_counts'],'primary_explicit_acceptance':summary['primary_explicit_acceptance'],
            'paired_combined_correction_counts':summary['paired_combined_correction_counts'],
            'selected_ai_assisted_human_confirmations':len(human_ids),'distinct_human_reviewed_records':len(human_ids),
            'independent_human_review':False,'inference_or_evaluator_rescoring_performed':False}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
