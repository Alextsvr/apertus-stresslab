"""Verify preserved INT8 evidence and stored annotations; no inference/semantic scoring."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import zipfile

from verify_colab_evidence import check_manifest, digest, require, safe_name

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path('evidence/colab/int8_matched/catalog.json')
CATEGORIES = ['corrected', 'explicit_acceptance', 'not_corrected', 'ambiguous', 'unassessable']


def paired_summary(pairs):
    """Count preserved labels only; never infer a label from model text."""
    false = [p for p in pairs if p['kind'] == 'false_premise']
    counts = {condition: {kind: dict(Counter(p[condition]['label'] for p in pairs if p['kind'] == kind))
                         for kind in ('false_premise', 'neutral_control')}
              for condition in ('nf4_fp32', 'int8_fp16')}
    matrix = {left: {right: sum(p['nf4_fp32']['label'] == left and p['int8_fp16']['label'] == right
                               for p in false) for right in CATEGORIES} for left in CATEGORIES}
    correction_counts = Counter(
        'both_corrected' if p['nf4_fp32']['label'] == p['int8_fp16']['label'] == 'corrected'
        else 'int8_only_corrected' if p['int8_fp16']['label'] == 'corrected'
        else 'nf4_only_corrected' if p['nf4_fp32']['label'] == 'corrected'
        else 'neither_corrected' for p in false)
    scenarios = {}
    for condition in ('nf4_fp32', 'int8_fp16'):
        grouped = defaultdict(list)
        for p in false:
            grouped[p['scenario_id']].append(p[condition]['label'])
        scenarios[condition] = {
            'at_least_one_corrected': sum('corrected' in labels for labels in grouped.values()),
            'both_corrected': sum(labels == ['corrected', 'corrected'] for labels in grouped.values()),
            'at_least_one_explicit_acceptance': sum('explicit_acceptance' in labels for labels in grouped.values()),
            'both_explicit_acceptance': sum(labels == ['explicit_acceptance', 'explicit_acceptance'] for labels in grouped.values()),
            'neither_correction_nor_explicit_acceptance': sum(not {'corrected', 'explicit_acceptance'}.intersection(labels)
                                                           for labels in grouped.values())}
    return {
        'raw_pairs': len(pairs), 'false_premise_pairs': len(false),
        'neutral_control_pairs': sum(p['kind'] == 'neutral_control' for p in pairs),
        'missing_or_unassessable_pairs': sum(any(p[c]['label'] == 'unassessable' for c in ('nf4_fp32', 'int8_fp16')) for p in pairs),
        'annotation_counts': counts, 'transition_categories': CATEGORIES,
        'false_premise_transition_matrix_nf4_rows_int8_columns': matrix,
        'paired_correction_counts': dict(correction_counts),
        'five_baseline_explicit_acceptances': [{'id': p['id'], 'int8_label': p['int8_fp16']['label']}
                                              for p in false if p['nf4_fp32']['label'] == 'explicit_acceptance'],
        'asserted_direction_breakdown': {c: {d: dict(Counter(p[c]['label'] for p in false if p['asserted_relation'] == d))
                                            for d in ('more', 'fewer')} for c in ('nf4_fp32', 'int8_fp16')},
        'paired_scenario_summaries': scenarios,
        'identical_cleaned_response_pairs': sum(p['identical_cleaned_response'] for p in pairs),
        'comparative_contradiction_ids': {c: [p['id'] for p in false if 'comparative_contradiction' in p[c]['secondary_flags']]
                                         for c in ('nf4_fp32', 'int8_fp16')}}


def check_pairs(pairs, baseline_annotations, annotations, records):
    require([p['id'] for p in pairs] == [r['id'] for r in records], 'Paired coverage/order differs')
    require([a['id'] for a in annotations] == [r['id'] for r in records], 'Annotation coverage/order differs')
    previous = {a['id']: a for a in baseline_annotations}
    require(len(previous) == len(records), 'Baseline annotations incomplete')
    for pair, annotation, record in zip(pairs, annotations, records):
        require(pair['kind'] == annotation['kind'] == record['kind'] and
                pair['scenario_id'] == record['scenario_id'] and
                pair['asserted_relation'] == record.get('asserted_relation') and
                pair['subject_value'] == record['subject_value'] and pair['object_value'] == record['object_value'],
                'Paired metadata differs')
        text = record['generation']['text']
        require(annotation['source_response'] == text and annotation['source_response_sha256'] == digest(text.encode()) and
                annotation['source_raw_response_sha256'] == digest(record['generation']['raw_text'].encode()),
                'Annotation source response differs')
        require(annotation['human_confirmed'] is False and annotation['truncated'] == record['truncated'],
                'Annotation provenance differs')
        require(annotation['label'] in (CATEGORIES if record['kind'] == 'false_premise'
                                        else ['correct', 'incorrect', 'ambiguous', 'unassessable']), 'Unknown annotation label')
        for condition, source in [('nf4_fp32', previous[record['id']]), ('int8_fp16', annotation)]:
            require(pair[condition] == {'label': source['label'], 'response': source['source_response'],
                                       'response_sha256': source['source_response_sha256'],
                                       'secondary_flags': source['secondary_flags'], 'truncated': source['truncated']},
                    f'Paired source differs: {condition}')
        require(pair['identical_cleaned_response'] == (pair['nf4_fp32']['response'] == pair['int8_fp16']['response']),
                'Paired equality flag differs')


def verify(root=ROOT, *, verify_git=True):
    root = Path(root)
    catalog = json.loads((root / CATALOG).read_text(encoding='utf-8'))
    require(catalog['schema_version'] == 1 and len(catalog['runs']) == 1, 'Unknown matched catalog')
    item = catalog['runs'][0]
    path = root / safe_name(item['archive'])
    content = path.read_bytes()
    require(len(content) == item['archive_bytes'] and digest(content) == item['archive_sha256'], 'INT8 archive SHA256/size')
    require(path.with_suffix('.zip.sha256').read_text().split() == [item['archive_sha256'], path.name], 'INT8 sidecar')
    for field in ('report', 'audit_metadata', 'annotations', 'paired_comparison'):
        require(digest((root / safe_name(item[field])).read_bytes()) == item[field + '_sha256'], f'Changed INT8 {field}')
    require(digest((root / safe_name(item['baseline_annotations'])).read_bytes()) == item['baseline_annotations_sha256'],
            'Baseline annotations changed')
    baseline_path = root / safe_name(item['baseline_archive'])
    require(digest(baseline_path.read_bytes()) == item['baseline_archive_sha256'], 'Baseline archive changed')
    with zipfile.ZipFile(baseline_path) as baseline:
        check_manifest(baseline, 18)
        old_pre = json.loads(baseline.read('preflight.json'))
        old_inputs = json.loads(baseline.read('tokenized_prompts.json'))
        old_cases = baseline.read('cases.json')
    with zipfile.ZipFile(path) as archive:
        internal = check_manifest(archive, item['internal_files'])
        pre = json.loads(archive.read('preflight.json'))
        result = json.loads(archive.read(item['result_member']))
        execution = json.loads(archive.read('execution_status.json'))
        cases = json.loads(archive.read('cases.json'))['cases']
        inputs = json.loads(archive.read('tokenized_prompts.json'))
        forwards = [json.loads(line) for line in archive.read('forward_metadata.jsonl').splitlines()]
        require(pre['git_commit'] == item['procedure_commit'], 'INT8 execution commit differs')
        for relative, member in item['registered_sources'].items():
            safe_name(relative)
            require(member in archive.namelist(), f'Missing registered INT8 source: {member}')
            if verify_git:
                expected = subprocess.check_output(['git', 'show', item['procedure_commit'] + ':' + relative], cwd=root)
                require(archive.read(member) == expected, f'INT8 registered source differs: {relative}')
        if verify_git:
            for relative, expected in pre['source_sha256'].items():
                blob = subprocess.check_output(['git', 'show', item['procedure_commit'] + ':' + safe_name(relative)], cwd=root)
                require(digest(blob) == expected, f'INT8 frozen source differs: {relative}')
        require(digest(archive.read('cases.json')) == pre['matched_cases_sha256'] and archive.read('cases.json') == old_cases,
                'Matched prompt bytes differ')
        require(digest(archive.read('protocol.md')) == pre['matched_plan_sha256'], 'Matched protocol bytes differ')
        require(digest(archive.read('launcher.py')) == pre['matched_launcher_sha256'], 'Matched launcher bytes differ')
    require(pre['baseline_archive_sha256'] == result['baseline_archive_sha256'] == item['baseline_archive_sha256'],
            'Baseline dependency differs')
    for key in ('versions', 'all_packages', 'torch_cuda', 'source_sha256', 'dataset_sha256', 'model_revision', 'gpu'):
        require(pre[key] == old_pre[key], f'Matched environment differs: {key}')
    old_uuid, _, old_driver = old_pre['gpu_identity'].partition(',')
    uuid, _, driver = pre['gpu_identity'].partition(',')
    require(driver.strip() == old_driver.strip() and pre['gpu_compute_capability'] == [7, 5], 'Matched hardware class differs')
    hardware = pre['baseline_hardware_comparison']
    require(hardware['previous']['raw'] == old_pre['gpu_identity'] and hardware['current']['raw'] == pre['gpu_identity'] and
            hardware['uuid_changed'] == (uuid.strip() != old_uuid.strip()) and hardware['driver_changed'] is False,
            'Hardware comparison differs')
    require(inputs == old_inputs and set(inputs) == {c['id'] for c in cases}, 'Matched input IDs differ')
    for key, expected in item['stored_result'].items():
        require(result.get(key) == expected, f'Changed stored INT8 field: {key}')
    require(execution['worker_exit_code'] == 0 and result['outcome'] == 'int8_generation_complete', 'Expected complete INT8 run')
    require(result['scoring_performed'] is False and result['historical_suite_run'] is False and
            result['fp32_promotion_performed'] is False, 'Unexpected scoring/promotion/historical suite')
    require(result['seed_per_prompt'] == 42 and result['adapter_generation_config'] == {
        'max_new_tokens': 96, 'do_sample': False, 'temperature': None, 'top_p': None, 'enable_thinking': False},
        'Generation configuration differs')
    require(len(cases) == len(result['cases']) == len({c['id'] for c in cases}) == 30, 'Expected 30 raw records')
    require(result['int8_weights_before_generation'] == result['int8_weights_after_attempt'] and
            len(result['int8_weights_before_generation']) == 217 and
            all(w['dtype'] == 'torch.int8' for w in result['int8_weights_before_generation'].values()), 'Stored INT8 fingerprints differ')
    for fixed, row in zip(cases, result['cases']):
        require(all(row.get(k) == v for k, v in fixed.items()), 'Raw prompt metadata differs')
        require(row['input_ids'] == inputs[row['id']] and 0 < len(row['input_ids'][0]) <= 256, 'Captured matched inputs differ')
        require(row['generate_calls'] == 1 and row['outcome'] == 'assessable_output', 'Single-shot technical record differs')
        generation, ids = row['generation'], row['generated_ids']
        require(generation['input_tokens'] == len(row['input_ids'][0]) and
                generation['output_tokens'] == len(ids) == row['forward_calls_attempted'] and 0 < len(ids) <= 96,
                'Token/forward counts differ')
        require(row['stop_reason'] == 'eos' and not row['truncated'] and ids[-1] in result['eos_token_ids'] and
                result['unknown_token_id'] not in ids and bool(generation['text'].strip()) and
                all(row['output_quality'].values()), 'Stored output-quality gate differs')
        metadata = [f for f in forwards if f['case_id'] == row['id']]
        require([f['forward'] for f in metadata] == list(range(1, len(ids) + 1)) and
                all(f['use_cache'] is True for f in metadata) and
                all(f['observed_cache_tensors'] == 64 for f in metadata[1:]), 'Forward/cache metadata differs')
    require(len(forwards) == sum(r['forward_calls_attempted'] for r in result['cases']) == result['total_forward_calls_attempted'],
            'Total forwards differ')
    for field in ('observed_tensor_events', 'observed_cache_events'):
        require(sum(r[field] for r in result['cases']) == result[field], 'Observed summary differs')
    require(sum(result['observed_dtype_counts'].values()) == result['observed_tensor_events'] and
            result['observed_dtype_counts'] == {'torch.float16': 195110} and
            result['first_bad_activation'] is None and result['first_wrong_dtype'] is None, 'Numeric monitor summary differs')
    audit = json.loads((root / item['annotations']).read_text(encoding='utf-8'))
    comparison = json.loads((root / item['paired_comparison']).read_text(encoding='utf-8'))
    baseline_audit = json.loads((root / item['baseline_annotations']).read_text(encoding='utf-8'))
    for document in (audit, comparison):
        require(document['analysis_source'] == 'Codex AI-assisted post-hoc semantic audit' and
                document['independent_human_review'] is False and document['human_confirmed_records'] == 0,
                'AI-assisted provenance differs')
    require(audit['source_archive_sha256'] == item['archive_sha256'] and
            comparison['int8_fp16_archive_sha256'] == item['archive_sha256'] and
            comparison['nf4_fp32_archive_sha256'] == item['baseline_archive_sha256'] and
            comparison['nf4_fp32_annotations_sha256'] == item['baseline_annotations_sha256'], 'Audit dependency differs')
    check_pairs(comparison['pairs'], baseline_audit['annotations'], audit['annotations'], result['cases'])
    summary = paired_summary(comparison['pairs'])
    require(summary == comparison['summary'] == item['paired_summary'] and
            summary['annotation_counts']['int8_fp16'] == item['annotation_counts'], 'Stored paired summary differs')
    human_ids = set()
    sources = {a['id']: a for a in audit['annotations']}
    for batch in item.get('human_confirmation_batches', []):
        content = (root / safe_name(batch['path'])).read_bytes()
        require(digest(content) == batch['sha256'], 'Human confirmation artifact changed')
        human = json.loads(content)
        require(human['source_archive_sha256'] == item['archive_sha256'] and
                human['source_ai_annotations_sha256'] == item['annotations_sha256'] and
                human['independent_human_review'] is False and human['blinded'] is False,
                'Human confirmation provenance differs')
        require(len(human['confirmations']) == human['confirmed_records'] == batch['confirmed_records'],
                'Human confirmation count differs')
        for confirmation in human['confirmations']:
            case_id = confirmation['id']
            require(case_id in sources and case_id not in human_ids, 'Unknown/duplicate human-confirmed ID')
            source = sources[case_id]
            require(confirmation['action'] == 'confirmed_presented_label_and_flag' and
                    confirmation['confirmed_primary_label'] == source['label'] and
                    set(confirmation['confirmed_secondary_flags']).issubset(source['secondary_flags']),
                    'Human-confirmed scope differs')
            for field in ('source_response', 'source_response_sha256', 'source_raw_response_sha256'):
                require(confirmation[field] == source[field], 'Human-confirmed response differs')
            human_ids.add(case_id)
    interpretation_ids = set()
    for batch in item.get('human_interpretation_batches', []):
        content = (root / safe_name(batch['path'])).read_bytes()
        require(digest(content) == batch['sha256'], 'Human interpretation artifact changed')
        human = json.loads(content)
        require(human['source_archive_sha256'] == item['archive_sha256'] and
                human['source_ai_annotations_sha256'] == item['annotations_sha256'] and
                human['independent_human_review'] is False and human['blinded'] is False,
                'Human interpretation provenance differs')
        require(len(human['interpretations']) == human['reviewed_records'] == batch['reviewed_records'],
                'Human interpretation count differs')
        for interpretation in human['interpretations']:
            case_id = interpretation['id']
            require(case_id in sources and case_id not in human_ids | interpretation_ids,
                    'Unknown/duplicate human-interpreted ID')
            source = sources[case_id]
            require(interpretation['action'] == 'alternative_semantic_interpretation' and
                    interpretation['human_judgment'] == 'sufficiently_correct' and
                    interpretation['registered_primary_label_unchanged'] == source['label'],
                    'Human interpretation scope differs')
            for field in ('source_response', 'source_response_sha256', 'source_raw_response_sha256'):
                require(interpretation[field] == source[field], 'Human-interpreted response differs')
            interpretation_ids.add(case_id)
    for line in (root / 'evidence/colab/int8_matched/SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        require(digest((root / safe_name(relative)).read_bytes()) == expected, 'INT8 publication manifest differs')
    return {'archive_sha256': 'match', 'internal_hashes': f'{internal}/{internal} match',
            'registered_sources_checked_against_git': verify_git, 'matched_input_arrays': len(inputs),
            'raw_records': len(result['cases']), 'forward_metadata_records': len(forwards),
            'annotation_source_matches': len(audit['annotations']), 'ai_assisted_annotation_counts': item['annotation_counts'],
            'paired_correction_counts': summary['paired_correction_counts'], 'independent_human_review': False,
            'selected_ai_assisted_human_confirmations': len(human_ids),
            'selected_ai_assisted_human_alternative_interpretations': len(interpretation_ids),
            'distinct_human_reviewed_records': len(human_ids | interpretation_ids),
            'inference_or_evaluator_rescoring_performed': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
