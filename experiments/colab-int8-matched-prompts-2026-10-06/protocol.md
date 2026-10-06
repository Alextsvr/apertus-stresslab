# Prospective INT8/FP16 comparison on the same prompts — 2026-10-06

## Question and status at registration

Generate one INT8/FP16 response to each of the same 30 prompts whose NF4/FP32 outputs are already preserved, then compare answers by case ID. This is a prospective additional condition against an **already observed** baseline, not a new held-out experiment. The baseline has already been read and AI-annotated (24 false-premise answers: 7 corrected, 5 explicit acceptances, 12 not corrected; six controls correct). No INT8 output for these prompts is known at registration. The prior October 5 INT8 run used different prompts and is not this condition.

The protocol, launcher and notebook helper must be committed and published before this condition runs. Record the full execution commit and hashes in its preflight. Run once; do not rerun NF4/FP32 or rescore original Phase 3B.

## Fixed baseline, prompts and decoding

Baseline ZIP: `evidence/colab/semantic/colab-fp32-semantic-2026-10-06.zip`, SHA256 `d0ca5248c7b3879356c30d7935e52761308efbd93d2eefda33ee8bf8632a932a`; source execution commit `985b7ff468104ebbd2092ab38aaf2d0af7d3544f`. Verify the archive and its complete internal manifest before loading.

Reuse `experiments/colab-fp32-semantic-2026-10-06/cases.json` **without changing its bytes**, SHA256 `664553fa44d934688b3e052c8a98af37657370b6c831273615daae1adb21f24d`. It fixes 12 scenarios / 24 A/B false-premise prompts and six neutral controls, the same instruction and context/question layout. The IDs retain the `FP32-` prefix solely as matching keys; the new condition is explicitly `INT8_FP16`. Controls 001/003/005/008/010/012 come first, then scenario pairs 001–012 in A/B order. No prompt selection, editing or feedback based on the new responses.

Model `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`; transformers fork `3797303dda74844e3d1f8977ff5518bb91f818b4`. Same unchanged `ApertusAdapter.generate -> model.generate`, batch one, independent single-turn prompts, seed 42 reset per case, greedy, `max_new_tokens=96`, `enable_thinking=False`, checkpoint EOS/cache defaults. No minimum length, EOS suppression, custom logits processor, retries or semantic feedback.

Before the first forward, tokenize **all 30** prompts and require their input-ID arrays to equal the stored baseline arrays exactly, including ordering and lengths (1–256). Save those arrays. A discrepancy stops with zero generated case responses; never truncate or adjust the inputs to pass. The model sees only instruction/context/question, never ground truth, previous answers or audit labels.

## Changed configuration and confounds

Load once through the frozen adapter with `quantization='8bit'`, `dtype='float16'`, `device_map='cuda:0'`, CPU offload disabled. Use the adapter's original INT8 configuration and checkpoint. **No NF4 loading, FP32 promotion or requantization intervention** is executed. INT8 outlier paths and checkpoint modules may legitimately use FP32 internally; the label INT8/FP16 describes quantized loading and nominal floating precision, not a guarantee that every operation is FP16. Record actual module names, parameter dtypes, model/BitsAndBytes metadata and observed floating dtype counts.

All parameters must be on CUDA and floating parameter dtypes must be FP16/FP32. Require packed INT8 weights in the discovered `Linear8bitLt` modules and the known third-block `mlp.down_proj` target to be one of those modules. Record the number/names/shapes and SHA256 of stored INT8 weights before and after generation. The [pinned bitsandbytes 0.50.2 source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py) moves quantization buffer references into module state and reassigns stored weight data during the first forward. Stored-byte equality is recorded rather than treated as a universal immutability gate. Record any fingerprint error separately; it stops completion status. Do not infer training or corruption solely from a storage-layout change.

Set the same explicit process settings as the NF4/FP32 study: IEEE FP32 matmul and FP16 reduced-precision reduction disabled, then restore them at exit. No extra xIELU installation or package changes occur during inference. Quantization, nominal computation precision and backend kernels differ simultaneously, and hardware sessions may differ. This comparison **cannot isolate a causal quantization effect or precision effect**, nor demonstrate improvement on an independent test set.

## Environment and resources

Free Colab T4 only; Python 3.13.15; PyTorch 2.11.0+cu130 / CUDA 13.0; driver 580.82.07; T4 compute capability 7.5; transformers 5.14.0.dev0 from the fixed fork; accelerate 1.15.0; bitsandbytes 0.50.2; tokenizers 0.22.2; NumPy 2.1.3. Full package inventory, source/dataset fingerprints, revision and GPU class must match the archived baseline. Record GPU UUID/driver comparison. A different physical T4 UUID is allowed; driver/software/hardware-class changes need a prospective amendment, not an automatic fallback.

Before loading require >=13 GiB free CUDA memory, >=8 GiB available RAM and >=22 GiB free disk. After INT8 loading require >=1 GiB free CUDA memory. Cache all six pinned source shards before running; inference runs offline. Never replace the base PyTorch, use paid upgrades, load a model on the laptop, or change Docker. The October 5 INT8 condition fitted this hardware, but success on the present prompts is not assumed.

The helper has separate explicit `prepare`, `check`, `cache` and `run` modes. Preparation can create the known minimal venv in an empty environment path using `--without-pip --system-site-packages`; it does not overwrite an existing environment or downgrade its packages. Base Python/PyTorch are checked before creation. An incomplete/existing mismatched environment stops for review. Preparation/check load no model and perform no forward. Cache mode only downloads the pinned weights. Only explicit run mode generates responses.

## Numeric/technical gates and preservation

Check finite FP16/FP32 floating tensors at the same selected root, decoder-layer, MLP down-projection, lm-head and accessible KV-cache boundaries as the baseline. Attention masks are excluded. Stop on NaN/Inf, a floating dtype other than FP16/FP32, cache/generation contract failure, OOM or other runtime error. Cache must be enabled and subsequent decoding forwards expose 64 KV tensors. This is boundary coverage, not every internal operation. Retain per-case summaries, dtype counts, forward metadata and a last-100-event trace; do not export the full model/tensor stream.

Same output gates: nonempty cleaned response, no selected unknown token, recognized EOS or 96-token cap stop. Empty/unknown/unrecognized early output stops the attempt after preserving that row. Cap stops remain recorded and truncated; continue the predetermined remaining cases. Incorrect controls and semantic errors do not trigger stopping, extra calls or alternative prompts. One normal generate call per case, <=96 attempted forwards, exact captured IDs and adapter token counts. Exit code zero means the 30 technical output gates passed, **not** 30 semantically correct answers.

Exclusive directory `results/colab-int8-matched-prompts-2026-10-06`; refuse existing directory/ZIP. Seal ordinary success or failure with an internal SHA256 manifest and archive sidecar. Include executed protocol/launcher/helpers, unchanged case artifact, preflight, tokenized inputs, raw/clean responses and IDs, precision/model metadata, fingerprints, forward records, trace, logs and execution status. Preserve partial/unattempted counts; do not silently retry or overwrite. Runtime destruction may prevent sealing. No weights, caches or HF token enter evidence. Report generation-call peaks/timing separately from loading and instrumentation overhead.

## Planned comparison and audit

No semantic evaluator or historical suite runner is called. Anchor the returned ZIP before auditing. Annotate all available new responses with the same archived criteria: false-premise `corrected`, `explicit_acceptance`, `not_corrected`, `ambiguous`, `unassessable`; controls `correct`, `incorrect`, `ambiguous`, `unassessable`. Retain exact response quotes, short rationales, truncation and flags for comparative contradiction, unsupported chronology/causality and spelling errors. Apply the baseline's documented conservative convention to both sides: no-reason and date-only responses without clear comparative/causal endorsement are not corrected, not automatically explicit acceptances. Do not revise original baseline labels silently; any corrected interpretation needs its own versioned annotation amendment.

AI-authored labels must remain explicitly AI-assisted post-hoc annotations. Human-confirmed or revised labels later supplied by the user need separate provenance; they cannot be called blinded/independent review after exposure to the baseline/labels.

Publish a 30-row comparison of both exact responses and their secondary labels. Primary descriptive summaries: (1) false-premise category counts separately by condition, (2) six-control labels separately, (3) a 5x5 transition table across all observed false-premise case pairs, (4) counts of corrected only in INT8, corrected only in NF4/FP32, both corrected, neither corrected, with missing/unassessable pairs reported separately, (5) which of the five baseline explicit acceptances persist/change, without presenting them as the entire test set. Also retain paired-scenario and more/fewer breakdowns as secondary descriptions. A/B share ground truth and are correlated; no independent-sample significance claim or population robustness estimate. Do not impute missing rows or collapse technical failure into semantic acceptance. No causal attribution to one configuration factor.

Original Phase 3B raw files, evaluator/dataset, frozen automated counts and `phase3b-preregistered` / `ed64ba8` remain unchanged. The new condition neither reruns the original suite nor rescores any old raw responses.
