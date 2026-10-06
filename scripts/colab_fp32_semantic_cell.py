"""Notebook-only helper. Explicit MODE='cache' or MODE='run'; never install packages."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

from google.colab import files, userdata

mode = globals().get('MODE')
assert mode in {'cache', 'run'}, "Specify MODE='cache' or MODE='run'"
root = Path('/content/apertus-stresslab')
python = Path('/content/apertus-env/bin/python')
assert root.is_dir() and python.is_file(), 'Use the currently prepared Colab session'
sys.path.insert(0, str(root / 'scripts'))
from colab_notebook_process import stream
destination = root / 'results/colab-fp32-semantic-2026-10-06'
assert not destination.exists() and not destination.with_suffix('.zip').exists(), 'Attempt exists; do not repeat'
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

if mode == 'cache':
    checked = subprocess.run([str(python), 'scripts/colab_fp32_semantic.py', '--check'], cwd=root,
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
else:
    assert Path('/content/apertus-pinned-snapshot.txt').is_file(), 'Complete cache preparation first'
    code = stream([str(python), '-u', 'scripts/colab_fp32_semantic.py', '--run'], cwd=root)
    print('Код завершения:', code)
    report_path = destination / 'semantic_run.json'
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
