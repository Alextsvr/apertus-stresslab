# Reproduction and evidence guide

## Inspect the existing result first

The published evidence is the original Phase 3B run `results/2026-10-02_111432`. Four files are included:

| File | Contents |
|---|---|
| `results.jsonl` | All 30 prompts, responses, frozen checks/evidence/statuses and generation metadata |
| `summary.json` | Original aggregate counts and planned breakdowns |
| `failures.jsonl` | The 8 original non-PASS records, copied by the runner |
| `metadata.json` | Model revision, command, environment and device/offload configuration |

Run from the repository root with Python 3.11+:

```text
python scripts/verify_evidence.py
```

This uses only the Python standard library. It reads files and counts existing labels; it does not invoke
the evaluator, regenerate answers or write results. SHA256 comparisons are against the original
[post-run manifest](audits/phase3b-2026-10-02_111432/SHA256SUMS). Git attributes disable newline conversion
for the curated raw run and audit artifacts so Windows checkouts preserve their bytes.

The [audit-time verification record](audits/phase3b-2026-10-02_111432/verification.json) is historical:
its document hashes describe the documents at that time, before later publication edits. It is not a
manifest of the entire current repository. The raw hashes remain the integrity anchors.

## Run software tests without inference

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
```

Core tests do not require model weights or a Hugging Face account. Echo/mocked test runs write only to
temporary directories. The published run is read as evidence, never rescored. Test failures must not be
fixed by silently changing the frozen experiment's evaluator or dataset.

## Create a portable evidence archive

```text
python scripts/package_evidence.py
```

Output: `dist/apertus-stresslab-phase3b-evidence.zip` and a neighbouring `.sha256` file. The package includes
README, license, Python configuration, dependency specifications, source, tests, scripts, documentation,
synthetic cases and the four curated raw files. It excludes other runs, caches, environments, credentials
and model weights. The archive contains `PACKAGE_SHA256SUMS` for its payload and `PACKAGE_PROVENANCE.json`
with the available Git HEAD/tag and dirty-state information; it does not include `.git` history.

After extraction, `python scripts/verify_evidence.py` works without Git or third-party packages. The archive
is a portable evidence snapshot, not a model distribution or a replacement for the repository's Git history.
The older `scripts/make_bundle.py` remains a source-only Colab helper and is not the evidence packager.

## Optional future inference: recorded Phase 3B configuration

The commands below are reproduction instructions, **not part of integrity verification**. They perform
new inference and require an appropriate dedicated environment. The project owner's working laptop and
Docker environment are not scheduled for another model run. The higher-precision comparison is deferred.

The original environment recorded Python 3.12, PyTorch `2.11.0+cu128`, transformers `5.14.0.dev0` from
Swiss AI fork commit `3797303dda74844e3d1f8977ff5518bb91f818b4`, accelerate `1.15.0`, bitsandbytes `0.50.2`
and tokenizers `0.22.2`. Consult raw metadata for the complete environment. The dependency files describe
installation requirements, not a complete environment lock; newer compatible versions may behave differently.

In a suitable environment, install the matching CUDA PyTorch build, then `requirements-inference.txt`.
Access to the gated model must be established using your own account and credentials outside the repository.
Do not put a token in a command saved to version control. No weights or credentials are bundled.

Recorded command (PowerShell, after installation and authentication):

```powershell
.\.venv\Scripts\python.exe -m stresslab run --suite false_premise_heldout `
  --adapter apertus `
  --model-id swiss-ai/Apertus-v1.5-8B `
  --revision a411d838600baf0e3635a3daf66fb7c55fc97bb6 `
  --dtype bfloat16 `
  --quantization 4bit `
  --cpu-offload `
  --gpu-max-memory-gib 7.0 `
  --seed 42 `
  --max-new-tokens 96 `
  --enable-thinking false
```

One new run means 30 calls, one per prompt. Keep its output in a new directory and report it as a new
experiment. Never overwrite or rescore the published Phase 3B run. Preserve errors and truncation flags;
do not select only the original failure cases or merge partial attempts as a complete run. A changed
precision/hardware configuration requires a finalized protocol before inference. Greedy decoding and a
fixed seed alone do not establish bitwise reproducibility across environments.
