"""Prospective INT8/FP16 condition on the existing 30 prompts; no evaluator or rerun."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from colab_quantization import ROOT, MODEL, REVISION, check, claim, git, require, save, seal, sha
from colab_4bit_diagnostic import NonFiniteObserved, tensors, stats
from colab_fp32_probe import UnexpectedPrecision, check_probe_environment
from colab_fp32_long import GenerateCapture, cache_values
from colab_layer_replay import precision_settings

STUDY = 'colab-int8-matched-prompts-2026-10-06'
DEST = ROOT / 'results' / STUDY
PLAN = ROOT / 'experiments' / STUDY / 'protocol.md'
CASES_PATH = ROOT / 'experiments/colab-fp32-semantic-2026-10-06/cases.json'
CASES_SHA = '664553fa44d934688b3e052c8a98af37657370b6c831273615daae1adb21f24d'
PRIOR = ROOT / 'evidence/colab/semantic/colab-fp32-semantic-2026-10-06.zip'
PRIOR_SHA = 'd0ca5248c7b3879356c30d7935e52761308efbd93d2eefda33ee8bf8632a932a'
CAP = 96
INPUT_CAP = 256
HELPERS = ('colab_quantization.py', 'colab_4bit_diagnostic.py', 'colab_layer_replay.py',
           'colab_fp32_probe.py', 'colab_fp32_long.py', 'verify_evidence.py',
           'colab_int8_matched_cell.py', 'colab_notebook_process.py')


def read_archive(path, digest):
    require(path.is_file() and sha(path) == digest, f'Dependency archive missing or changed: {path.name}')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        entries = [line.split('  ', 1) for line in z.read('SHA256SUMS').decode().splitlines()]
        require(len(names) == len(set(names)) and len(entries) == len({n for _, n in entries})
                and {n for _, n in entries} == set(names) - {'SHA256SUMS'}, 'Incomplete dependency manifest')
        for expected, name in entries:
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, f'Dependency member changed: {name}')
        return {name: z.read(name) for name in names}


def prompt_for(case):
    return f"{case['instruction']}\n\nContext: {case['context']}\n\nQuestion: {case['question']}"


def load_cases():
    require(sha(CASES_PATH) == CASES_SHA, 'Prospective prompts changed')
    cases = json.loads(CASES_PATH.read_text(encoding='utf-8'))['cases']
    require(len(cases) == len({c['id'] for c in cases}) == 30, 'Expected 30 unique cases')
    require([c['kind'] for c in cases[:6]] == ['neutral_control'] * 6
            and all(c['kind'] == 'false_premise' for c in cases[6:]), 'Case types/order changed')
    return cases


def preflight():
    prior = read_archive(PRIOR, PRIOR_SHA)
    previous = json.loads(prior['preflight.json'])
    result = json.loads(prior['semantic_run.json'])
    require(result['outcome'] == 'semantic_generation_complete' and result['completed_records'] == 30,
            'Completed NF4/FP32 baseline required')
    require(prior['cases.json'] == CASES_PATH.read_bytes(), 'Matched-prompt case bytes differ')
    current = check()
    import torch
    current['gpu_compute_capability'] = list(torch.cuda.get_device_capability(0))
    if DEST.is_dir():
        save(DEST / 'preflight.json', current)
    current['baseline_hardware_comparison'] = check_probe_environment(previous, current)
    require(not current['baseline_hardware_comparison']['driver_changed'], 'Driver changed; amend prospective plan')
    cases = load_cases()
    expected_inputs = json.loads(prior['tokenized_prompts.json'])
    require(set(expected_inputs) == {c['id'] for c in cases}, 'Baseline prompt IDs incomplete')
    for path in (PLAN, CASES_PATH, Path(__file__), ROOT / 'scripts/colab_int8_matched_cell.py'):
        require(bool(git('ls-files', '--', path.relative_to(ROOT).as_posix())), f'Uncommitted input: {path.name}')
    current.update(condition='INT8_FP16', matched_plan_sha256=sha(PLAN), matched_cases_sha256=CASES_SHA,
                   matched_launcher_sha256=sha(Path(__file__)), baseline_archive_sha256=PRIOR_SHA,
                   planned_records=len(cases))
    return current, expected_inputs


def int8_identity(model, linear8bit_class):
    """Record stored INT8 bytes; backend state can legitimately initialize at first forward."""
    identity = {}
    for name, module in model.named_modules():
        if isinstance(module, linear8bit_class):
            weight = module.weight.detach().cpu().contiguous()
            identity[name] = {'shape': list(weight.shape), 'dtype': str(weight.dtype),
                              'weight_sha256': hashlib.sha256(weight.numpy().tobytes()).hexdigest()}
    require(bool(identity), 'No INT8 linear modules found')
    return identity


def matched_inputs(processor, cases, baseline_inputs):
    expected_inputs = {}
    for case in cases:
        encoded = processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_for(case)}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
        ids = encoded['input_ids'].tolist()
        require(len(ids) == 1 and 0 < len(ids[0]) <= INPUT_CAP, f"Input cap exceeded: {case['id']}")
        expected_inputs[case['id']] = ids
    require(expected_inputs == baseline_inputs, 'Templated input IDs differ from NF4/FP32 baseline; zero generation allowed')
    return expected_inputs


class Observer:
    """Check complete selected tensors; retain summaries/last events, not all activations."""
    def __init__(self):
        self.step = 0
        self.events = 0
        self.cache_events = 0
        self.first_bad = None
        self.first_wrong_dtype = None
        self.dtype_counts = {}
        self.trace = deque(maxlen=100)

    def inspect(self, name, stage, value):
        import torch
        for path, tensor in tensors(value):
            event = {'step': self.step, 'module': name, 'stage': stage, 'path': path, **stats(tensor)}
            self.events += 1
            self.cache_events += int(name == 'kv_cache')
            self.trace.append(event)
            self.dtype_counts[str(tensor.dtype)] = self.dtype_counts.get(str(tensor.dtype), 0) + 1
            if tensor.dtype not in {torch.float16, torch.float32}:
                self.first_wrong_dtype = event
                raise UnexpectedPrecision(f'Expected FP16/FP32 at {name}: {stage} {path}')
            if event['nan'] or event['positive_inf'] or event['negative_inf']:
                self.first_bad = event
                raise NonFiniteObserved(f'Non-finite tensor at {name}: {stage} {path}')


def output_quality(text, ids, unk, eos_ids):
    return {'nonempty_cleaned_response': bool(text.strip()),
            'no_unknown_token_selected': unk is None or unk not in ids,
            'recognized_stop': bool(ids) and (ids[-1] in eos_ids or len(ids) == CAP)}


def execute_cases(adapter, observer, report, persist, cases, expected_inputs):
    from stresslab.schemas import GenerationConfig
    model, processor = adapter._model, adapter._processor
    config = GenerationConfig(max_new_tokens=CAP, do_sample=False, enable_thinking=False)
    report['adapter_generation_config'] = config.model_dump(mode='json')
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    unk = processor.tokenizer.unk_token_id
    report.update(eos_token_ids=eos_ids, unknown_token_id=unk)
    for case in cases:
        observer.step = 0
        events_before, cache_before = observer.events, observer.cache_events
        row = {**case, 'outcome': 'generation_started', 'forward_calls_attempted': 0}
        report['cases'].append(row)
        persist()
        print(f"Generating once: {case['id']} ({len(report['cases'])}/{len(cases)})", flush=True)
        original = model.generate
        had_method, prior_method = 'generate' in model.__dict__, model.__dict__.get('generate')
        capture = GenerateCapture(original)
        model.generate = capture
        try:
            result = adapter.generate(prompt_for(case), config, seed=42)
            row['generation'] = result.model_dump(mode='json')
            require(capture.calls == 1 and capture.output_ids is not None, 'Missing single normal generate call')
            require(capture.input_ids == expected_inputs[case['id']], 'Template/tokenized prompt changed')
            require(len(capture.output_ids) == 1, 'Unexpected generation batch')
            ids, all_ids = capture.input_ids[0], capture.output_ids[0]
            require(all_ids[:len(ids)] == ids, 'Generated prefix changed')
            new_ids = all_ids[len(ids):]
            row.update(input_ids=capture.input_ids, generated_ids=new_ids, supplied_generate_kwargs=capture.kwargs)
            require(0 < len(new_ids) <= CAP and observer.step == len(new_ids), 'Forward/token count differs')
            require(result.input_tokens == len(ids) and result.output_tokens == len(new_ids), 'Adapter token counts differ')
            row['stop_reason'] = 'eos' if new_ids[-1] in eos_ids else ('token_cap' if len(new_ids) == CAP else 'other')
            row['output_quality'] = output_quality(result.text, new_ids, unk, eos_ids)
            row['truncated'] = row['stop_reason'] == 'token_cap'
            row['outcome'] = 'assessable_output' if all(row['output_quality'].values()) else 'output_quality_stop'
            persist()
            if row['outcome'] != 'assessable_output':
                return 'output_quality_stop'
        finally:
            if had_method:
                model.generate = prior_method
            else:
                del model.generate
            row.update(forward_calls_attempted=observer.step, generate_calls=capture.calls,
                       observed_tensor_events=observer.events-events_before,
                       observed_cache_events=observer.cache_events-cache_before)
            if capture.input_ids is not None:
                row['input_ids'] = capture.input_ids
            save(DEST / 'last_events.json', list(observer.trace))
            persist()
    return 'int8_generation_complete'


def worker():
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    import bitsandbytes as bnb
    from transformers import set_seed
    from stresslab.models import ApertusAdapter

    observer, handles, settings, model = Observer(), [], None, None
    report = {'outcome': 'runtime_error', 'detail': None, 'cases': [], 'planned_records': 30,
              'scoring_performed': False, 'historical_suite_run': False, 'seed_per_prompt': 42,
              'condition': 'INT8_FP16', 'fp32_promotion_performed': False, 'baseline_archive_sha256': PRIOR_SHA,
              'max_new_tokens': CAP, 'max_input_tokens': INPUT_CAP, 'total_forward_calls_attempted': 0,
              'generation_path': 'frozen ApertusAdapter.generate -> model.generate'}
    persist = lambda: save(DEST / 'int8_run.json', report)
    progress = (DEST / 'forward_metadata.jsonl').open('w', encoding='utf-8')
    try:
        current, baseline_inputs = preflight()
        save(DEST / 'preflight.json', current)
        cases = load_cases()
        snapshot = Path(Path('/content/apertus-pinned-snapshot.txt').read_text().strip())
        require(snapshot.name == REVISION, 'Cached model revision changed')
        shards = set(json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map'].values())
        require(len(shards) == 6 and all((snapshot / s).is_file() for s in shards), 'Pinned cache incomplete')
        set_seed(42)
        print('Loading cached INT8/FP16 model once; no FP32 promotion', flush=True)
        adapter = ApertusAdapter(model_id=MODEL, revision=REVISION, dtype='float16', quantization='8bit',
                                 device_map='cuda:0', cpu_offload=False)
        adapter.load()
        model, processor = adapter._model, adapter._processor
        report['initial_model_loading_info'] = adapter.info().model_dump(mode='json')
        report['parameter_elements_by_dtype'] = {dtype: sum(p.numel() for p in model.parameters() if str(p.dtype) == dtype)
                                                   for dtype in sorted({str(p.dtype) for p in model.parameters()})}
        require(all(p.device.type == 'cuda' and p.device.index == 0 for p in model.parameters()), 'Parameters off GPU')
        require(all(not p.is_floating_point() or p.dtype in {torch.float16, torch.float32}
                    for p in model.parameters()), 'Unexpected floating parameter dtype')
        before = int8_identity(model, bnb.nn.Linear8bitLt)
        report['int8_weights_before_generation'] = before
        require(all(w['dtype'] == 'torch.int8' for w in before.values()), 'INT8 weights not packed')
        require('model.language_model.layers.2.mlp.down_proj' in before, 'Expected target not INT8')
        report['quantized_module_count'] = len(before)
        settings = precision_settings(torch)
        report['original_precision_settings'] = settings
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report['int8_precision_settings'] = precision_settings(torch)
        report['checkpoint_generation_config'] = model.generation_config.to_dict()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report['post_load_memory'] = {'allocated_gib': round(torch.cuda.memory_allocated()/2**30, 3),
                                          'cuda_free_gib': round(free/2**30, 3), 'cuda_total_gib': round(total/2**30, 3)}
        require(free >= 2**30, 'Need >=1 GiB free CUDA memory after INT8 loading; no fallback')
        expected_inputs = matched_inputs(processor, cases, baseline_inputs)
        save(DEST / 'tokenized_prompts.json', expected_inputs)
        selected = [(name, module) for name, module in model.named_modules()
                    if name == 'lm_head' or name.endswith('.mlp.down_proj')
                    or (name.startswith('model.language_model.layers.') and name.rsplit('.', 1)[-1].isdigit())]
        require(any(name == 'model.language_model.layers.2.mlp.down_proj' for name, _ in selected),
                'Expected diagnostic boundary missing')
        report['observed_module_names'] = [name for name, _ in selected]

        def root_before(mod, args, kwargs):
            require(observer.step < CAP, 'Forward cap exceeded; no retry')
            observer.step += 1
            report['total_forward_calls_attempted'] += 1
            cache = kwargs.get('past_key_values')
            observer.inspect('root_forward', 'input', {'activation': args[:1], **kwargs})
            observer.inspect('kv_cache', 'input', cache_values(cache))
            metadata = {'case_id': report['cases'][-1]['id'], 'forward': observer.step,
                        'use_cache': kwargs.get('use_cache'),
                        'cache_class': None if cache is None else type(cache).__name__,
                        'observed_cache_tensors': len(cache_values(cache)),
                        'input_shape': list(kwargs['input_ids'].shape) if kwargs.get('input_ids') is not None else None}
            progress.write(json.dumps(metadata) + '\n')
            progress.flush()
            require(kwargs.get('use_cache') is True, 'Normal generation cache disabled')
            if observer.step > 1:
                require(len(cache_values(cache)) == 64, 'Expected all 32 layers of KV cache')

        def root_after(mod, args, output):
            observer.inspect('root_forward', 'output', output)
            observer.inspect('kv_cache', 'output', cache_values(getattr(output, 'past_key_values', None)))

        handles.append(model.register_forward_pre_hook(root_before, with_kwargs=True))
        handles.append(model.register_forward_hook(root_after))
        for name, module in selected:
            def before_hook(mod, args, kwargs, name=name):
                observer.inspect(name, 'input', {'activation': args[:1], **kwargs})
            def after_hook(mod, args, output, name=name):
                observer.inspect(name, 'output', output)
            handles.append(module.register_forward_pre_hook(before_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(after_hook))
        persist()
        report['outcome'] = execute_cases(adapter, observer, report, persist, cases, expected_inputs)
    except NonFiniteObserved as exc:
        report.update(outcome='non_finite_detected', detail=str(exc))
    except UnexpectedPrecision as exc:
        report.update(outcome='unexpected_precision', detail=str(exc))
    except Exception as exc:
        report.update(outcome='runtime_error', detail=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        for handle in handles:
            handle.remove()
        progress.close()
        if model is not None and 'int8_weights_before_generation' in report:
            try:
                after = int8_identity(model, bnb.nn.Linear8bitLt)
                report['int8_weights_after_attempt'] = after
                report['stored_int8_weight_bytes_unchanged'] = after == report['int8_weights_before_generation']
            except Exception as exc:
                report.update(outcome='runtime_error', fingerprint_error=f'{type(exc).__name__}: {exc}')
        if settings is not None:
            torch.backends.cuda.matmul.fp32_precision = settings['fp32_precision']
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings['allow_fp16_reduced_precision_reduction']
        report.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      first_bad_activation=observer.first_bad, first_wrong_dtype=observer.first_wrong_dtype,
                      observed_tensor_events=observer.events, observed_cache_events=observer.cache_events,
                      observed_dtype_counts=observer.dtype_counts,
                      completed_records=sum(c['outcome'] == 'assessable_output' for c in report['cases']),
                      unattempted_records=30-len(report['cases']))
        save(DEST / 'last_events.json', list(observer.trace))
        persist()
    print('INT8_MATCHED_OUTCOME:', report['outcome'], flush=True)
    return int(report['outcome'] != 'int8_generation_complete')


def run():
    require(not DEST.with_suffix('.zip').exists(), 'Attempt archive exists; preserve it, do not retry')
    claim(DEST)
    code = None
    try:
        shutil.copyfile(PLAN, DEST / 'protocol.md')
        shutil.copyfile(CASES_PATH, DEST / 'cases.json')
        shutil.copyfile(Path(__file__), DEST / 'launcher.py')
        for name in HELPERS:
            shutil.copyfile(ROOT / 'scripts' / name, DEST / name)
        with (DEST / 'int8_run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        save(DEST / 'execution_status.json', {'worker_exit_code': code,
                                             'finished_at_utc': datetime.now(timezone.utc).isoformat()})
        print('Evidence archive:', seal(DEST), flush=True)
    return 1 if code is None else code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(preflight()[0], indent=2))
    else:
        sys.exit(worker() if args.worker else run())
