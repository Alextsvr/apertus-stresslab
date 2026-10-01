# Apertus StressLab

> A reproducible red-teaming toolkit that stress-tests Apertus for hallucinations, inconsistency, and robustness failures, with evidence, severity scoring, and replayable test cases.

Built for **Hack Apertus 2026 — Track 1A: Red-Teaming Apertus**.

## Why

A model failure is only useful to the people improving the model if it is **observable, backed by evidence, reproducible and replayable**. StressLab stores every prompt, response, seed, generation setting, model revision and runtime environment, so anyone can rerun a finding and check it. Evaluation logic will stay deterministic and auditable; no commercial LLM API is used at any point.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | One real Apertus inference, stored with full metadata | Code ready; first real run pending (see below) |
| 1 | Schemas, model adapter, runner, JSONL storage, metadata, CLI, smoke cases, tests | Done |
| 2–9 | Grounding suite, mutations, consistency/robustness, reproducibility, scoring, dashboard, experiments, report | Not started |

**No experimental results exist yet.** Nothing in this repository should be read as a finding about Apertus until a real run is committed.

## Model

| | |
|---|---|
| Model | [`swiss-ai/Apertus-v1.5-8B`](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) (official, Apache-2.0) |
| Size | ~9B parameters (text + image + audio encoders), BF16 weights ≈ 18 GB |
| Access | **Gated**: free Hugging Face account + accept the license/AUP on the model page + a read token |
| Library | swiss-ai `transformers` fork pinned at `3797303dda74844e3d1f8977ff5518bb91f818b4` (required by the model card; mainline transformers does not load it yet) |
| Thinking mode | Optional; controlled with `--enable-thinking true/false` and recorded |

Rough hardware guide for inference with transformers:

| Setup | Mode | Notes |
|---|---|---|
| NVIDIA GPU ≥ 24 GB VRAM | `--quantization none` (bf16) | Reference setting |
| NVIDIA GPU 12–16 GB (e.g. Colab T4) | `--quantization 8bit` | ~10 GB weights; recorded as quantized |
| NVIDIA GPU 8 GB | `--quantization 4bit --cpu-offload` | text model mostly on GPU, overflow in RAM; slower, recorded |
| CPU only | not practical | needs ~20 GB free RAM and minutes per answer |

## Install (Windows PowerShell)

Requires Python 3.11 or 3.12 and Git.

```powershell
cd C:\Extra\apertus-stresslab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt     # core + pytest (light, no torch)
python -m pytest                    # unit tests, no model download
```

Check your hardware (writes `results\env_check.txt`, read-only probe):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_env.ps1
Get-Content results\env_check.txt
```

### Inference dependencies (only if you have a suitable NVIDIA GPU)

```powershell
# 1) CUDA build of PyTorch FIRST (otherwise pip installs a CPU-only build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# 2) Apertus transformers fork + accelerate + bitsandbytes
pip install -r requirements-inference.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Hugging Face access (free, no billing)

1. Accept the terms on https://huggingface.co/swiss-ai/Apertus-v1.5-8B.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Provide it via the environment, never in source files:

```powershell
$env:HF_TOKEN = "<paste-your-token>"     # current session only
# or once, stored in your user profile by huggingface_hub:
hf auth login
```

Model weights are cached in `%USERPROFILE%\.cache\huggingface` (~18 GB). Set `$env:HF_HOME` to move it.

## Run

```powershell
# Pipeline check without a model (fake "echo" adapter, clearly marked as not real)
python -m stresslab infer --adapter echo --prompt "Hello"

# Phase 0: first REAL Apertus inference (pick --quantization for your GPU)
python -m stresslab infer --enable-thinking false --prompt "Context: Maria Keller was born in Bern in 1981. Question: Where and when was Maria Keller born? Answer using only the context."

# Smoke suite (data/test_cases/smoke.jsonl, 4 cases)
python -m stresslab run --suite smoke --enable-thinking false

# Inspect a run / print environment
python -m stresslab show results\2026-10-01_153000
python -m stresslab env
```

### Small GPUs (8 GB): 4-bit + explicit CPU offload

```powershell
python -m stresslab infer --quantization 4bit --cpu-offload --gpu-max-memory-gib 7.0 --enable-thinking false --prompt "..."
```

`--cpu-offload` builds an explicit device map from a weightless model skeleton (no `device_map="auto"`):
the image/audio tokenizers (never used for text prompts) are parked in CPU RAM, 4-bit decoder layers get GPU
priority, and whatever does not fit under `--gpu-max-memory-gib` (minus headroom) stays **unquantized in system
RAM** and is copied to the GPU on every forward pass. Disk offload is never used. The plan is printed before
loading and stored in `metadata.json` (`model.extra.offload_plan`, `model.extra.hf_device_map`). This is not
full-GPU inference and is slower; runs with offload are labelled as such.

Useful options: `--quantization {none,8bit,4bit}`, `--cpu-offload`, `--gpu-max-memory-gib 7.0`, `--gpu-reserve-gib 1.0`, `--cpu-max-memory-gib 24`, `--seed 42`, `--max-new-tokens 256`, `--do-sample --temperature 0.7 --top-p 0.9`, `--revision <commit-sha>`, `--results-dir <path>`.

### No suitable GPU? Use Google Colab (free)

1. `python scripts/make_bundle.py` → creates `dist\apertus-stresslab-bundle.zip`.
2. Open `notebooks/colab_smoke_test.ipynb` in Colab (File → Upload notebook), set the runtime to **T4 GPU**, add `HF_TOKEN` in Colab *Secrets*.
3. Run all cells; it uploads the bundle, installs, runs one inference + the smoke suite in 8-bit, and downloads `stresslab-results.zip`.
4. Unzip into `C:\Extra\apertus-stresslab\results\`.

## Output

Each run gets its own directory; existing runs are never overwritten (a `_2`, `_3` suffix is added on collision).

```
results/
  2026-10-01_153000/
    metadata.json    # run id, command, seed, generation config, model id + resolved revision,
                     # dtype, quantization, device, Python/OS/torch/transformers/GPU, git commit
    results.jsonl    # one record per test: prompt, response, raw response, usage, status, ...
```

Phase 1 does not evaluate answers: every record has `"status": "UNSCORED"` (or `"ERROR"` with the error message). `results/` is git-ignored; curated runs that back a published finding will be added explicitly.

## Project layout

```
src/stresslab/   cli.py, runner.py, models.py (adapters), schemas.py, storage.py,
                 cases.py, environment.py, config.py
data/test_cases/ smoke.jsonl
tests/           pytest (no model, no network)
notebooks/       colab_smoke_test.ipynb
scripts/         check_env.ps1, make_bundle.py
docs/            methodology.md, technical_report.md
```

## Known limitations

- The first real Apertus run has not been executed yet (needs a GPU or Colab plus the gated-model token).
- The `ApertusAdapter` follows the official model card but could not be exercised end-to-end during development; expect a small fix-up on first contact with the fork.
- 8-bit/4-bit quantization changes the model's numerics. Findings from quantized runs must be labelled as such and, where possible, confirmed in bf16.
- No evaluators, mutations, reproducibility statistics or severity yet (Phases 2–6).
- Greedy decoding is the default; GPU kernels can still introduce small nondeterminism.

## License

Apache-2.0 (see `LICENSE`). Apertus itself is released by the Swiss AI Initiative under Apache-2.0 with an Acceptable Use Policy.
