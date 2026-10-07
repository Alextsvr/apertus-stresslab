# Prospective Qwen comparison on the completed Apertus context-control set

Registered after inspecting all Apertus outputs and selected human confirmations,
before any Qwen response. Model choice and this question were informed by those
observations. This is an exploratory cross-model extension, not a new held-out
validation, a blind audit, or a causal test of architecture/training/quantization.

## Fixed inputs and reference

Use exactly the same 108 ordered rows and instruction/context/question strings
from `../colab-fp32-context-controls-2026-10-07/cases.json` (SHA256
`7caba65af8db5ec1edaeaec1132cc13f72c620c6f069a4c5776b868a5cd0fb42`):
36 controls first, then 72 false-premise prompts; six entity pairs, twelve numeric
scenarios, 36 adjacent context triplets, all six condition orders balanced.
The bare/date/non-temporal-fact contexts and numbers are unchanged. Their original
`previously_observed: false` metadata describes their first Apertus registration;
they have now been observed on Apertus. It is not a claim of novelty for this run.

Reuse the frozen six-category rubric (SHA256
`6eccd6174008fe4ea51e88fa5560e288ebd12ac0d4b72ebbdaf1158635faca03`).
The successful Apertus reference ZIP is
`evidence/colab/context_controls/colab-fp32-context-controls-2026-10-07.zip`, SHA256
`8af72b78e03b415e89ca615bdf9d3723b4d24a2b77e38b174048fcd0698676ca`.
Its immutable AI annotations have SHA256
`cc9518f5f90c39531dbdf1c50824b9b5615a883f3eeefb3081e338bc8cb381ef`.
The separate four selected human confirmations (two primary labels, two flag-only)
retain their original scope. Do not promote them to a full human audit.
No rerun, rescore, or edit of that reference or Phase 3B is permitted. Original
Phase 3B preregistration remains `phase3b-preregistered` / `ed64ba8`.

## Model, template, numerics and environment

`Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, public Apache-2.0 checkpoint.
Model/config/tokenizer/index hashes and four shard names are in
`model_manifest.json`. Only metadata and the tokenizer were used locally on CPU;
no Qwen weights were downloaded or loaded and no Qwen forward was performed.
All 108 exact input arrays and user-text hashes are registered in
`tokenization_reference.json`. Use the native Qwen template with one user string;
it inserts its own default identity system message. No extra manual system text,
reasoning switch, padding, or revised prompts. All 36 Qwen triplets independently
satisfy dates = attributes = bare + 20 tokens; inputs range 98–129 tokens.
This length equality is within Qwen, not equality to Apertus input IDs/lengths.
Runtime must match every registered array before the first forward.

Use a free Colab Tesla T4 (compute capability 7.5), CUDA 13.0, driver 580.82.07.
GPU UUID may differ. Core versions and full package inventory must match the
Apertus reference: Python 3.13.15, PyTorch 2.11.0+cu130, Transformers Swiss fork
3797303dda74844e3d1f8977ff5518bb91f818b4 (5.14.0.dev0), Accelerate 1.15.0,
bitsandbytes 0.50.2, tokenizers 0.22.2, NumPy 2.1.3. Fail on mismatch; amend the
protocol before inference if a new environment is needed. No paid resources,
local inference, CPU offload, or remote custom model code.

Load once using NF4 double quantization with initial FP16 compute; promote all
floating parameters/registered buffers and all 196 Linear4bit compute dtypes to
FP32 before any forward. Packed weights and quantization state must remain
unchanged through promotion and generation. Do not compare their fingerprint to
Apertus weights. Require >=13 GiB free CUDA memory, >=8 GiB available RAM and
>=22 GiB free disk before loading, >=1 GiB free CUDA after promotion. No fallback.
Use IEEE FP32 matmul and disable reduced-precision FP16 reduction as in Apertus.

One ordinary `AutoModelForCausalLM.generate` call per row, seed 42 reset per row,
greedy (`do_sample=false`), cap 96 new tokens, cache enabled, one beam, explicit
`repetition_penalty=1.0`. The Qwen checkpoint defaults to 1.05; override it to
match Apertus's effective 1.0. Apertus stored nulls resolve through the pinned
library's `GenerationConfig._get_default_generation_params()` and
`_prepare_generation_config`, not through the serialized null value itself.
Qwen's inactive sampling defaults are retained and recorded. Native EOS IDs
151645 and 151643, pad 151643; no unknown token. Record actual generation config,
provided kwargs, input/output IDs, clean/raw text, timing, memory and stops.

## Observation, stopping and evidence

Check full floating tensors for FP32 and finiteness at root inputs/outputs,
all 28 decoder layer inputs/outputs, every MLP down-projection, LM head and all
56 recognized KV tensors on cached steps. This checks selected boundaries,
not every intermediate operation. Preserve first bad/dtype event and last 100
events, per-forward metadata and per-row progress. Technical output checks do
not judge semantic correctness. Keep token-cap outputs with truncation flags
and continue. Empty output, unknown token, unrecognized stop, non-finite tensor,
wrong dtype, OOM or other runtime exception stops the attempt and seals partial
evidence. No retries, substitutions, silent resume, or overwriting. A changed
plan or follow-up attempt needs a new prospective record and destination.

Preparation/check/cache modes perform no model forward. Cache downloads the
pinned public checkpoint without needing HF_TOKEN. Run mode alone loads/generates.
Seal protocol, manifest, unchanged cases/rubric, token reference, launcher/helpers,
preflight, reports and logs into a ZIP with internal SHA256SUMS and outer SHA256.
The archive excludes weights, cache and credentials. Anchor/check returned ZIP
hashes before creating annotations; disclose any earlier reading of responses.

## Prespecified analysis after return

Primary descriptive question: does Qwen explicitly accept at least one false
comparison among the 72 false-premise prompts? Count all six frozen categories,
by condition (24 each) and overall, plus control correctness (12 each).
Separately flag a false comparison printed alongside the correctly attributed
stored values. Zero observed acceptances is not proof of model-specificity.

Compare the same case IDs to the existing Apertus AI annotations: within each
condition, publish the paired acceptance 2x2 table, all six-category transitions,
correction totals/differences, secondary flags and control outcomes, with
Qwen-minus-Apertus signs fixed. Within Qwen, publish paired date-vs-attribute
acceptance (primary context contrast), then date-vs-bare and attribute-vs-bare;
report explicit+implicit corrections separately. Show six entity-block results,
A/B and more/fewer splits, ambiguous/unassessable sensitivity, missing/truncated
counts and available paired denominators; do not impute failed/unattempted rows
as successes. Technical failure means an incomplete comparison, not model safety.
No threshold-driven stopping, extra seeds, cherry-picking, p-values that treat
correlated prompts as independent, or model ranking from six entity blocks.

Any first semantic labels are AI-assisted and unblinded, separate from frozen
automated Phase 3B results and actual selected human confirmations. Preserve the
original audit and record annotation disagreements separately. Optional human
review targets concrete disputed/new failure examples, with exact limited scope;
there is no requirement to manually review every response.

Different architecture, training, tokenizer, default system text, native chat
template and quantized kernels confound cross-model attribution. Same user text,
decoding settings and within-model length controls do not isolate those causes.
Do not claim universal robustness, isolated quantization effects or independent
replication from this small, previously observed, single-generation prompt set.
