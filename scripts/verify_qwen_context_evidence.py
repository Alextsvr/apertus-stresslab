"""Verify preserved Qwen evidence and counts from fixed AI labels; never classify text."""
import json
from pathlib import Path
import subprocess
import zipfile
from verify_colab_evidence import check_manifest,digest,require,safe_name
from verify_context_controls_evidence import check_annotations,check_triplets,paired_summary
from qwen_comparison_summary import cross_model_summary

ROOT=Path(__file__).resolve().parents[1]
CATALOG=Path('evidence/colab/qwen_context/catalog.json')
MODEL='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'


def check_cross_pairs(pairs,qwen,apertus):
    require([p['id'] for p in pairs]==[a['id'] for a in qwen], 'Cross-model coverage/order differs')
    old={a['id']:a for a in apertus}
    require(len(old)==len(apertus) and set(old)=={a['id'] for a in qwen}, 'Cross-model ID coverage differs')
    for pair,new in zip(pairs,qwen):
        prior=old[new['id']]
        for key in ('id','kind','base_pair_id','scenario_id','context_triplet_id','context_condition','direction_state','variant','asserted_relation'):
            require(pair[key]==new[key]==prior[key], 'Cross-model pairing metadata differs')
        for key in ('instruction','context','question','subject','object','metric','subject_value','object_value'):
            require(new[key]==prior[key], 'Cross-model user text/ground truth differs')
        for model,a in (('qwen',new),('apertus',prior)):
            require(pair[model]=={'label':a['label'],'response':a['source_response'],
                'response_sha256':a['source_response_sha256'],'secondary_flags':a['secondary_flags'],
                'truncated':a['truncated']}, 'Cross-model annotation source differs')


def verify(root=ROOT, *, verify_git=True):
    root = Path(root)
    catalog = json.loads((root/CATALOG).read_text(encoding='utf-8'))
    require(catalog['schema_version']==1 and len(catalog['runs'])==1, 'Unknown context catalog')
    item = catalog['runs'][0]
    path = root/safe_name(item['archive'])
    content = path.read_bytes()
    require(digest(content)==item['archive_sha256'] and len(content)==item['archive_bytes'], 'Context archive SHA256/size')
    require(path.with_suffix('.zip.sha256').read_text().split()==[item['archive_sha256'],path.name], 'Context sidecar differs')
    for field in ('report','annotations','within_comparison','cross_comparison','audit_metadata','receipt'):
        require(digest((root/safe_name(item[field])).read_bytes())==item[field+'_sha256'], f'Changed context {field}')
    with zipfile.ZipFile(path) as z:
        check_manifest(z,item['internal_files'])
        pre = json.loads(z.read('preflight.json'))
        result = json.loads(z.read('qwen_run.json'))
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
    require(reference['revision']==pre['model_revision'] and reference['model']==pre['model_id'] and reference['cases_sha256']==pre['context_cases_sha256']
            and reference['model_weights_loaded'] is False and reference['model_forward_calls']==0
            and reference['registered_before_model_responses'] is True and reference['rubric_sha256']==pre['context_rubric_sha256'], 'Token registration provenance differs')
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
    require(execution['worker_exit_code']==0 and result['outcome']=='qwen_context_complete', 'Context completion differs')
    require(result['seed_per_prompt']==42 and result['max_new_tokens']==96 and result['max_input_tokens']==256, 'Generation settings differ')
    require(result['adapter_generation_config']=={'max_new_tokens':96,'do_sample':False,'temperature':None,
             'top_p':None,'enable_thinking':None}, 'Adapter config differs')
    require(result['context_control_precision_settings']=={'fp32_precision':'ieee','allow_fp16_reduced_precision_reduction':False},
            'Numerical settings differ')
    baseline = root/safe_name(item['baseline_archive'])
    require(digest(baseline.read_bytes())==pre['apertus_context_reference_archive_sha256']==item['baseline_archive_sha256'], 'Baseline archive changed')
    with zipfile.ZipFile(baseline) as old:
        check_manifest(old,20)
        old_pre = json.loads(old.read('preflight.json'))
        old_run = json.loads(old.read('context_run.json'))
    for key in ('versions','all_packages','torch_cuda','source_sha256','dataset_sha256','gpu'):
        require(pre[key]==old_pre[key], f'Context environment differs: {key}')
    require(pre['environment_reference_model_revision']==old_pre['model_revision'], 'Environment model reference differs')
    require(pre['model_id']==MODEL and pre['model_revision']==REVISION, 'Qwen model identity differs')
    driver = lambda p:p['gpu_identity'].partition(',')[2].strip()
    require(driver(pre)==driver(old_pre) and pre['gpu_compute_capability']==[7,5], 'GPU class/driver differs')
    weights = result['quantized_weights_before_promotion']
    require(len(weights)==196 and weights==result['quantized_weights_after_promotion']==result['quantized_weights_after_attempt'], 'Packed state changed')
    records = result['cases']
    require(len(cases)==len(records)==result['completed_records']==result['planned_records']==108 and result['unattempted_records']==0,
            'Context raw coverage differs')
    require(len({r['id'] for r in records})==108 and set(inputs)=={r['id'] for r in records}, 'Context IDs/input coverage differs')
    for fixed,row in zip(cases,records):
        require(all(row.get(k)==v for k,v in fixed.items()), 'Context prompt metadata changed')
        require(row['input_ids']==inputs[row['id']] and len(row['input_ids'])==1 and 0<len(row['input_ids'][0])<=256, 'Context input IDs differ')
        require(row['generate_calls']==1 and row['outcome']=='assessable_output', 'Single-shot technical contract differs')
        require(row['supplied_generate_kwargs']=={'max_new_tokens':96,'do_sample':False,'repetition_penalty':1.0}, 'Generate kwargs changed')
        ids = row['generated_ids']
        require(row['generation']['output_tokens']==len(ids)==row['forward_calls_attempted'] and 0<len(ids)<=96
                and row['generation']['input_tokens']==len(row['input_ids'][0]), 'Token/forward counts differ')
        require(row['stop_reason']=='eos' and not row['truncated'] and ids[-1] in result['eos_token_ids']
                and result['unknown_token_id'] not in ids and row['generation']['text'].strip(), 'Context technical output differs')
        require(all(row['output_quality'].values()), 'Technical quality flag differs')
        captured = [f for f in forwards if f['case_id']==row['id']]
        require([f['forward'] for f in captured]==list(range(1,len(ids)+1)) and all(f['use_cache'] is True for f in captured)
                and captured[0]['input_shape']==[1,len(row['input_ids'][0])]
                and all(f['observed_cache_tensors']==56 and f['input_shape']==[1,1] for f in captured[1:]), 'Forward/cache metadata differs')
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
    comparison = json.loads((root/item['within_comparison']).read_text(encoding='utf-8'))
    check_triplets(comparison['triplets'],annotations,records)
    summary = paired_summary(comparison['triplets'])
    require(comparison['summary']==summary and item['annotation_counts']==summary['annotation_counts'], 'Qwen within-model counts differ')
    old_audit_bytes=(root/safe_name(item['apertus_annotations'])).read_bytes()
    require(digest(old_audit_bytes)==item['apertus_annotations_sha256']==
            'cc9518f5f90c39531dbdf1c50824b9b5615a883f3eeefb3081e338bc8cb381ef', 'Historical Apertus annotations changed')
    old_annotations=json.loads(old_audit_bytes)['annotations']
    cross=json.loads((root/item['cross_comparison']).read_text(encoding='utf-8'))
    require(cross['qwen_archive_sha256']==item['archive_sha256'] and
            cross['apertus_archive_sha256']==item['baseline_archive_sha256'] and
            cross['qwen_annotations_sha256']==item['annotations_sha256'] and
            cross['apertus_annotations_sha256']==item['apertus_annotations_sha256'], 'Cross-model provenance differs')
    check_cross_pairs(cross['pairs'],annotations,old_annotations)
    cross_summary=cross_model_summary(cross['pairs'])
    require(cross['summary']==cross_summary,'Cross-model stored counts differ')
    receipt=json.loads((root/item['receipt']).read_text(encoding='utf-8'))
    integrity=json.loads((root/item['audit_metadata']).read_text(encoding='utf-8'))
    require(receipt['archive_sha256']==item['archive_sha256'] and receipt['procedure_commit']==item['procedure_commit'] and
            receipt['model_responses_read_before_anchor'] is False and receipt['semantic_annotations_created_before_anchor'] is False and
            receipt['sender_checksum_received_and_matched'] is True, 'Receipt provenance differs')
    require(integrity['archive_sha256']==item['archive_sha256'] and integrity['procedure_commit']==item['procedure_commit'] and
            integrity['hashes_saved_before_semantic_annotations'] is True and integrity['responses_read_before_anchor'] is False and
            integrity['sender_checksum_received_and_matched'] is True and integrity['semantic_evaluator_invoked'] is False and
            integrity['human_confirmed_records']==0 and integrity['inference_or_rescoring_performed_during_audit'] is False, 'Audit provenance differs')
    require(receipt['registered_sources']==item['registered_sources'] and receipt['internal_files']==item['internal_files'], 'Receipt source scope differs')
    with zipfile.ZipFile(path) as z:
        require(receipt['internal_member_sha256']=={n:digest(z.read(n)) for n in z.namelist() if n!='SHA256SUMS'}, 'Receipt member hashes differ')
    for field,actual in (('records',len(records)),('forward_calls',len(forwards)),
                        ('tensor_events',result['observed_tensor_events']),('cache_events',result['observed_cache_events'])):
        require(integrity[field]==actual, 'Integrity count differs')
    effective=result['effective_generation_config']
    require(effective['repetition_penalty']==1.0 and effective['do_sample'] is False and effective['num_beams']==1 and
            effective['max_new_tokens']==96 and effective['use_cache'] is True and
            effective['eos_token_id']==[151645,151643] and effective['pad_token_id']==151643, 'Effective Qwen decoding differs')
    require(result['checkpoint_generation_config']['repetition_penalty']==1.05 and result['unknown_token_id'] is None,
            'Checkpoint/special token contract differs')
    expected_names={'lm_head'}|{f'model.layers.{i}' for i in range(28)}|{f'model.layers.{i}.mlp.down_proj' for i in range(28)}
    require(set(result['observed_module_names'])==expected_names,'Observed Qwen boundary scope differs')
    for case in cases:
        text=f"{case['instruction']}\n\nContext: {case['context']}\n\nQuestion: {case['question']}"
        require(digest(text.encode())==reference['user_text_sha256'][case['id']], 'Exact user text changed')
    require(integrity['token_input_range']==[min(r['generation']['input_tokens'] for r in records),max(r['generation']['input_tokens'] for r in records)] and
            integrity['token_output_range']==[min(r['generation']['output_tokens'] for r in records),max(r['generation']['output_tokens'] for r in records)] and
            integrity['peak_gpu_memory_gib']==max(r['generation']['peak_gpu_memory_gib'] for r in records) and
            integrity['instrumented_generation_latency_sum_s']==sum(r['generation']['latency_s'] for r in records), 'Reported technical ranges differ')
    for line in (root/'evidence/colab/qwen_context/SHA256SUMS').read_text().splitlines():
        expected,relative=line.split('  ',1)
        require(digest((root/safe_name(relative)).read_bytes())==expected, 'Qwen publication manifest differs')
    return {'archive_sha256':'match','internal_hashes':'22/22 match','registered_sources_checked_against_git':verify_git,
            'raw_records':len(records),'forward_metadata_records':len(forwards),'annotation_source_matches':len(annotations),
            'registered_input_arrays_matched':len(inputs),'equal_length_date_attribute_triplets':36,
            'ai_assisted_annotation_counts':summary['annotation_counts'],
            'combined_corrections':summary['combined_corrections'],
            'cross_model_acceptance_differences':{c:v['explicit_acceptance_qwen_minus_apertus']['net_left_minus_right']
                for c,v in cross_summary['by_condition'].items()},
            'human_confirmed_qwen_records':0,'independent_human_review':False,
            'inference_or_evaluator_rescoring_performed':False}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
