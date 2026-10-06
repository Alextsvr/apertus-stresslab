"""Read-only verification of new semantic raw bytes and AI annotation provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from verify_colab_evidence import check_manifest, digest, require, safe_name

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path('evidence/colab/semantic/catalog.json')


def dependency(root, relative, expected, internal_files):
    path = root / safe_name(relative)
    require(digest(path.read_bytes()) == expected, f'Dependency archive changed: {relative}')
    with zipfile.ZipFile(path) as z:
        check_manifest(z, internal_files)
        return json.loads(z.read('preflight.json'))


def verify(root=ROOT, *, verify_git=True):
    root = Path(root)
    catalog = json.loads((root / CATALOG).read_text(encoding='utf-8'))
    require(catalog['schema_version'] == 1, 'Unknown semantic catalog schema')
    require(len(catalog['runs']) == 1, 'Expected one new semantic run')
    item = catalog['runs'][0]
    path = root / safe_name(item['archive'])
    content = path.read_bytes()
    require(len(content) == item['archive_bytes'] and digest(content) == item['archive_sha256'], 'Semantic archive SHA256/size')
    require(path.with_suffix('.zip.sha256').read_text().split() == [item['archive_sha256'], path.name], 'Semantic sidecar')
    for field in ('report', 'audit_metadata', 'annotations'):
        require(digest((root / safe_name(item[field])).read_bytes()) == item[field+'_sha256'], f'Changed semantic {field}')
    with zipfile.ZipFile(path) as z:
        check_manifest(z, item['internal_files'])
        pre, result = json.loads(z.read('preflight.json')), json.loads(z.read(item['result_member']))
        execution = json.loads(z.read('execution_status.json'))
        cases = json.loads(z.read('cases.json'))['cases']
        inputs = json.loads(z.read('tokenized_prompts.json'))
        forwards = [json.loads(line) for line in z.read('forward_metadata.jsonl').splitlines()]
        require(pre['git_commit'] == item['procedure_commit'], 'Semantic execution commit differs')
        for relative, member in item['registered_sources'].items():
            safe_name(relative)
            require(member in z.namelist(), f'Missing registered semantic source: {member}')
            if verify_git:
                expected = subprocess.check_output(['git', 'show', item['procedure_commit']+':'+relative], cwd=root)
                require(z.read(member) == expected, f'Semantic registered source differs: {relative}')
        if verify_git:
            for relative, expected in pre['source_sha256'].items():
                blob = subprocess.check_output(['git', 'show', item['procedure_commit']+':'+safe_name(relative)], cwd=root)
                require(digest(blob) == expected, f'Semantic frozen source differs: {relative}')
        require(digest(z.read('cases.json')) == pre['semantic_cases_sha256'], 'Prospective case bytes changed')
        require(digest(z.read('protocol.md')) == pre['semantic_plan_sha256'], 'Prospective protocol bytes changed')
        require(digest(z.read('launcher.py')) == pre['semantic_launcher_sha256'], 'Executed launcher bytes changed')
    for key, expected in item['stored_result'].items():
        require(result.get(key) == expected, f'Changed stored semantic field: {key}')
    require(result['scoring_performed'] is False and result['historical_suite_run'] is False, 'Unexpected semantic scoring or historical suite')
    require(execution['worker_exit_code'] == 0 and result['outcome'] == 'semantic_generation_complete', 'Expected completed semantic run')
    prepared = dependency(root, 'evidence/colab/preparation/colab-next-session-preflight-20261006T114501627825Z.zip',
                          pre['preparation_archive_sha256'], 1)
    previous = dependency(root, 'evidence/colab/colab-fp32-generation-long-2026-10-06.zip',
                          pre['preceding_archive_sha256'], 15)
    for key in ('versions', 'all_packages', 'torch_cuda', 'source_sha256', 'dataset_sha256', 'model_revision', 'gpu'):
        require(pre[key] == prepared[key] == previous[key], f'Semantic environment changed: {key}')
    require(pre['gpu_compute_capability'] == [7, 5], 'GPU class differs')
    driver = lambda p: p['gpu_identity'].partition(',')[2].strip()
    require(driver(pre) == driver(prepared) == driver(previous), 'Driver differs')
    require(len(cases) == len(result['cases']) == 30, 'Expected 30 raw records')
    require(len({c['id'] for c in cases}) == 30 and set(inputs) == {c['id'] for c in cases}, 'Case IDs/tokenized prompts differ')
    require(result['adapter_generation_config'] == {'max_new_tokens':96, 'do_sample':False,
            'temperature':None, 'top_p':None, 'enable_thinking':False}, 'Generation config changed')
    weights = result['quantized_weights_before_promotion']
    require(len(weights) == 217 and weights == result['quantized_weights_after_promotion']
            == result['quantized_weights_after_attempt'], 'Packed state differs')
    records = result['cases']
    for fixed, row in zip(cases, records):
        require(all(row.get(k) == v for k,v in fixed.items()), f'Raw prompt metadata changed: {fixed["id"]}')
        require(row['input_ids'] == inputs[row['id']], 'Captured input IDs differ')
        require(0 < len(row['input_ids'][0]) <= 256, 'Prompt token cap differs')
        require(row['generate_calls'] == 1 and row['outcome'] == 'assessable_output', 'Single-shot technical record differs')
        generation, ids = row['generation'], row['generated_ids']
        require(generation['input_tokens'] == len(inputs[row['id']][0]) and
                generation['output_tokens'] == len(ids) == row['forward_calls_attempted'], 'Token/forward counts differ')
        require(row['stop_reason'] == 'eos' and not row['truncated'] and ids[-1] in result['eos_token_ids'], 'Expected untruncated EOS')
        require(result['unknown_token_id'] not in ids and bool(generation['text'].strip()), 'Stored technical output quality differs')
        require(all(row['output_quality'].values()), 'Stored quality flag differs')
        rows = [f for f in forwards if f['case_id'] == row['id']]
        require([f['forward'] for f in rows] == list(range(1,len(ids)+1)), 'Forward metadata differs')
        require(all(f['use_cache'] is True for f in rows) and
                all(f['observed_cache_tensors'] == 64 for f in rows[1:]), 'Cached decoding metadata differs')
    require(len(forwards) == sum(r['forward_calls_attempted'] for r in records)
            == result['total_forward_calls_attempted'], 'Total forwards differ')
    for field in ('observed_tensor_events','observed_cache_events'):
        require(sum(r[field] for r in records) == result[field], f'Observed summary differs: {field}')
    require(result['first_bad_activation'] is None and result['first_wrong_dtype'] is None, 'Stored numerical outcome differs')
    audit = json.loads((root / item['annotations']).read_text(encoding='utf-8'))
    require(audit['source_archive_sha256'] == item['archive_sha256'] and
            audit['analysis_source'] == 'Codex AI-assisted post-hoc semantic audit' and
            audit['independent_human_review'] is False and audit['human_confirmed_records'] == 0, 'Audit provenance differs')
    annotations = audit['annotations']
    require([a['id'] for a in annotations] == [r['id'] for r in records], 'Annotation coverage/order differs')
    for annotation, row in zip(annotations,records):
        text = row['generation']['text']
        require(annotation['source_response'] == text and annotation['source_response_sha256'] == digest(text.encode())
                and annotation['source_raw_response_sha256'] == digest(row['generation']['raw_text'].encode()), 'Annotation source response differs')
        require(annotation['human_confirmed'] is False and annotation['truncated'] == row['truncated'], 'Annotation provenance/quality differs')
    counts = {kind:dict(Counter(a['label'] for a in annotations if a['kind']==kind))
              for kind in ('false_premise','neutral_control')}
    require(counts == item['annotation_counts'], 'Stored annotation counts differ')
    for line in (root/'evidence/colab/semantic/SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ',1)
        require(digest((root/safe_name(relative)).read_bytes()) == expected, 'Semantic publication manifest differs')
    return {'archive_sha256':'match','internal_hashes':'18/18 match','registered_sources_checked_against_git':verify_git,
            'raw_records':len(records),'forward_metadata_records':len(forwards),
            'annotation_source_matches':len(annotations),'ai_assisted_annotation_counts':counts,
            'independent_human_review':False,'inference_or_evaluator_rescoring_performed':False}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
