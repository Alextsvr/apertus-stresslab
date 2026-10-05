# Prospective Colab T4 quantization comparison (2026-10-05)

Status: prospective protocol, prepared before either new condition. The Git commit
containing this document and its launcher is the checkpoint; the launcher records
its exact commit and protocol SHA256 before loading weights. This is a separate
configuration-sensitivity study on already observed prompts, not Phase 3B repeated
as new held-out evidence and not an unquantized/BF16 validation.

## Question and fixed design

Do explicit false-comparison acceptances persist under LLM.int8(), compared with
NF4 on the same free Colab Tesla T4? Both configurations still use quantization.
Differences cannot be attributed to bit width alone: the quantization algorithms
and their arithmetic differ. FP16 is requested for unquantized floating modules
and NF4 computation; LLM.int8 uses its own mixed INT8/FP16 computation and default
outlier threshold of 6.0. Model-declared FP32 modules remain FP32.

| Setting | Fixed value |
|---|---|
| Budget | $0; free Colab only, no paid upgrade or local model execution |
| Hardware | One Tesla T4, compute capability 7.5, about 15 GiB usable VRAM |
| Model | `swiss-ai/Apertus-v1.5-8B` |
| Model revision | `a411d838600baf0e3635a3daf66fb7c55fc97bb6` |
| Order | One complete 4-bit condition, then one complete 8-bit condition |
| 4-bit | bitsandbytes NF4, double quantization, FP16 compute |
| 8-bit | bitsandbytes LLM.int8, no retained FP16 weight copy |
| Placement | Explicit `cuda:0` for the full model; no CPU/disk offload |
| Cases | All 30 cases in stored dataset order: 24 false-premise variants, 6 controls |
| Decoding | Seed 42 per case, greedy, thinking false, max 96 new tokens |
| Calls | One generation per prompt per condition: 60 total planned calls |
| Scoring | Existing frozen evaluator once for each newly generated response |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130; driver observed during setup: 580.82.07 |
| Transformers | 5.14.0.dev0, Swiss AI fork `3797303dda74844e3d1f8977ff5518bb91f818b4` |
| Other pinned packages | accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.22.2, NumPy 2.1.3 |

The wrapper uses the unchanged adapter, runner, prompt construction and evaluator
from published code commit `cd4e1438937c23ae7ff4919645d1ae00c3b29912`. Its small
adapter subclass checks GPU placement before inference and prints progress only.
It does not alter generation, scoring or the dataset. All installed package
names/versions and GPU UUID/driver are recorded before each condition; the wrapper
rejects changes between them. Other dependencies are recorded, not fully pinned.

The original preregistration remains `phase3b-preregistered` at
`ed64ba8cb382073d652b24e647a898c94f8ce186`. Original run
`results/2026-10-02_111432` is read-only: no regeneration, rescoring or hash changes.
Its frozen false-premise counts remain **16 PASS / 6 POTENTIAL / 2 DETECTED**,
with **6/6 controls PASS**. New conditions are never pooled into these counts.
Dataset SHA256: `62f1b7434a450da17ab3e64a09986d0ec555d1aa86981236baa27c606ee4d459`.

## Completed setup observations, before model inference

The user reported T4/15 GiB VRAM, 12.7 GiB RAM (11.2 GiB initially available),
70.2 GiB initially free disk and successful CUDA tests for small NF4 and INT8
linear layers. These are infrastructure checks, not model responses. Access to
the exact revision's config and safetensors index passed; the index reports six
weight shards and 18,396,023,436 tensor bytes (17.13 GiB before quantization).

A local, weightless meta-device skeleton using the pinned fork estimated 7.513
GiB (NF4) and 10.698 GiB (INT8) for full-GPU weights, including model-declared
FP32 components and the unquantized output head. No model weights were loaded
on the user's laptop. Estimates exclude loading peaks, KV cache, activations,
CUDA context and quantization workspaces; successful loading is not guaranteed.

The original installation's optional audio extra pulled pyctcdecode 0.5.0 and
NumPy 1.26.4 into the Colab environment. Before inference, pyctcdecode was removed
and NumPy restored to 2.1.3. Model/processor imports and both tiny GPU checks then
passed. This setup repair did not change frozen research code or produce answers.

## Gates, attempts and evidence

1. Before inference, require clean tracked files, unchanged `src/` and `data/`
   relative to the published commit, matching four original raw hashes, expected
   package/fork versions, one T4, at least 13 GiB free VRAM, 8 GiB available RAM
   and 22 GiB free disk before each condition.
   These conservative resource gates do not guarantee loading will fit.
2. First run `scripts/colab_quantization.py --check` in the dedicated Colab
   environment. It reads provenance/resources only. Authenticate through Colab
   Secrets as `HF_TOKEN`; never put a token in code, logs or Git. After the check,
   `--run` performs the fixed pair, with each model in a fresh subprocess so the
   previous model's GPU memory is released. Same Colab runtime/GPU for both.
3. The study directory is `results/colab-t4-quantization-2026-10-05`, with separate
   subdirectories and normal runner metadata/results for each condition. Existing
   attempts are rejected, not deleted, overwritten, resumed or silently repeated.
4. On model-load failure, retain early metadata and logs. Per-case exceptions use
   the frozen runner's behavior: record ERROR and continue remaining cases once.
   If a condition ends with any ERROR, a missing case or a fatal error, do not
   start the next condition. Do not select successful cases, change precision,
   lower the suite size, enable offload or choose an alternative model mid-study.
   Any later attempt requires a dated protocol amendment and separate output.
5. Store preflight, complete package inventory, launcher/protocol copies, per-run
   environment, device map, responses, token counts, errors and truncation flags.
   The parent hashes all surviving study files and creates a study-only ZIP before
   semantic inspection; it prints progress, not answer text. Download the ZIP and
   checksum immediately. A killed/disconnected VM may prevent final packaging:
   preserve surviving partial files and label the study incomplete. There is no
   guarantee of durable storage in a free Colab session.

## Analysis decided before generation

Primary: report the paired frozen outcomes for all 24 false-premise prompts, raw
counts per condition and exact transitions/IDs; A/B variants, more/fewer direction
and the 12 paired scenarios remain visible. Report all six controls separately.
Count errors, missing outputs and truncations explicitly; none are PASS by default.
The original Phase 3B condition is historical context, with different hardware,
BF16 computation, CPU offload, Python and CUDA; it is not a matched causal control.

Secondary: review all 30 new answers in each completed condition with the existing
semantic rubric, separately from frozen scores. Preserve verbatim evidence and
AI/human provenance; AI annotations are not independent human adjudication. Do not
inspect only previously failing cases. Hash raw files before this audit; do not
modify the frozen evaluator in response to disagreements.

Persistent false comparisons in INT8 show persistence in this second quantized
configuration, not proof of behavior in unquantized BF16. Their disappearance is
a descriptive configuration difference, not proof NF4 alone caused the original
failures. One greedy output per prompt, known synthetic prompts, paired variants,
fixed condition order and one GPU/session do not establish an independent failure
rate, general model reliability, statistical significance or repeat-run stability.

This protocol leaves the earlier unquantized precision-check draft deferred.

## Pre-inference amendment 1 (2026-10-05): CUDA memory accounting

The initial protocol/launcher checkpoint was
`d89cf3d43141911af7990f64f3fce53b750e826f`. Its resource check rejected the user's
T4 before any weight loading or model inference. The launcher incorrectly required
CUDA-reported total memory >=14.9 GiB, in addition to the intended free-memory gate.
Its combined error message did not distinguish the two conditions.

The user supplied a diagnostic showing nvidia-smi total 15,360 MiB and no listed
compute processes, but CUDA total 14.563 GiB and free 14.461 GiB. Available RAM was
9.415 GiB and disk 69.726 GiB. The intended >=13 GiB free VRAM gate was satisfied.
Remove the erroneous nominal-total assumption, retain the same GPU identity and
>=13 GiB free VRAM requirement, and record CUDA total separately. A failed check
now reports both actual values. RAM/disk gates and every experimental condition
remain unchanged. No study attempt has been generated or resumed by this amendment.
The Git commit containing this amendment supersedes the initial checkpoint for
new inference and is captured in the study's preflight and per-condition metadata.
