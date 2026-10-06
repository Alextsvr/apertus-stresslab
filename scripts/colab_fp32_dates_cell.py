"""Explicit preparation/check/cache/run; only run mode generates responses."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

from google.colab import files, userdata

mode = globals().get('MODE')
assert mode in {'prepare', 'check', 'cache', 'run'}, "Specify MODE='prepare', 'check', 'cache' or 'run'"
root = Path('/content/apertus-stresslab')
python = Path('/content/apertus-env/bin/python')
assert root.is_dir(), 'Clone the pinned project first'
sys.path.insert(0, str(root / 'scripts'))
from colab_notebook_process import stream
destination = root / 'results/colab-fp32-date-ablation-2026-10-06'
assert not destination.exists() and not destination.with_suffix('.zip').exists(), 'Attempt exists; do not repeat'
if mode == 'prepare' and not python.is_file():
    environment = python.parent.parent
    assert not environment.exists(), 'Incomplete environment exists; preserve it for review'
    stream([sys.executable, '-c', "import sys,torch; assert sys.version_info[:3]==(3,13,15), 'Base Python changed'; assert torch.__version__=='2.11.0+cu130', 'Base PyTorch changed'"], check=True)
    stream([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(environment)], check=True)
    pins = ['torch==2.11.0+cu130', 'accelerate==1.15.0', 'bitsandbytes==0.50.2',
            'tokenizers==0.22.2', 'numpy==2.1.3', 'huggingface_hub==1.31.0',
            'safetensors==0.8.0', 'pydantic==2.13.5', 'rich==13.9.4']
    stream([sys.executable, '-m', 'pip', '--python', str(python), 'install', *pins], check=True)
    stream([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-deps',
            'git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4'], check=True)
    stream([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-deps', '-e', str(root)], check=True)
assert python.is_file(), 'Use MODE=prepare to prepare an empty environment'

if mode in {'prepare', 'check'}:
    checked = subprocess.run([str(python), 'scripts/colab_fp32_dates.py', '--check'], cwd=root,
                             capture_output=True, text=True)
    if checked.returncode:
        print(checked.stdout, checked.stderr, flush=True)
        checked.check_returncode()
    preflight = json.loads(checked.stdout)
    print(json.dumps({k:preflight[k] for k in ('condition','versions','gpu','gpu_identity',
          'free_vram_gib','available_ram_gib','free_disk_gib','baseline_hardware_comparison','weights_loaded')},indent=2))
    pointer = Path('/content/apertus-pinned-snapshot.txt')
    cache_ready = False
    if pointer.is_file():
        snapshot = Path(pointer.read_text().strip())
        index = snapshot / 'model.safetensors.index.json'
        if snapshot.name == 'a411d838600baf0e3635a3daf66fb7c55fc97bb6' and index.is_file():
            shards = set(json.loads(index.read_text())['weight_map'].values())
            cache_ready = len(shards) == 6 and all((snapshot/s).is_file() and (snapshot/s).stat().st_size > 0 for s in shards)
    print('CACHE_PRESENT:', cache_ready)
    print('DATE_PREFLIGHT_OK: no model loaded or inference')
else:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

if mode == 'cache':
    checked = subprocess.run([str(python), 'scripts/colab_fp32_dates.py', '--check'], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if checked.returncode:
        print(checked.stderr, flush=True)
        checked.check_returncode()
    print('Preflight OK; downloading pinned cache only', flush=True)
    env = os.environ.copy()
    env.pop('HF_HUB_OFFLINE', None)
    code = r'''
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
model='swiss-ai/Apertus-v1.5-8B'
revision='a411d838600baf0e3635a3daf66fb7c55fc97bb6'
index=json.loads(Path(hf_hub_download(model,'model.safetensors.index.json',revision=revision)).read_text())
shards=sorted(set(index['weight_map'].values()))
assert len(shards)==6
snapshot=Path(snapshot_download(repo_id=model,revision=revision,
    allow_patterns=shards+['*.json','*.txt','*.jinja','*.model'],max_workers=2))
assert all((snapshot/name).is_file() for name in shards)
Path('/content/apertus-pinned-snapshot.txt').write_text(str(snapshot))
print('CACHE_READY: 6/6 shards; no model loaded or inference')
'''
    stream([str(python), '-u', '-c', code], cwd=root, env=env, check=True)
elif mode == 'run':
    assert Path('/content/apertus-pinned-snapshot.txt').is_file(), 'Complete cache preparation first'
    code = stream([str(python), '-u', 'scripts/colab_fp32_dates.py', '--run'], cwd=root)
    print('Код завершения:', code)
    report_path = destination / 'date_ablation_run.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        print(json.dumps({key: report.get(key) for key in (
            'outcome', 'detail', 'planned_records', 'completed_records', 'unattempted_records',
            'total_forward_calls_attempted', 'observed_tensor_events', 'first_bad_activation',
            'first_wrong_dtype', 'packed_state_unchanged_after_attempt', 'scoring_performed')}, indent=2))
    for suffix in ('.zip', '.zip.sha256'):
        artifact = destination.with_suffix(suffix)
        if artifact.is_file():
            if suffix == '.zip':
                print('Archive SHA256:', hashlib.sha256(artifact.read_bytes()).hexdigest())
            files.download(str(artifact))
