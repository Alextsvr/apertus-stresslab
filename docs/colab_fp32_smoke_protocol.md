# Prospective NF4 / FP32 normal-generation smoke test — 2026-10-06

Publish this protocol, launcher and relevant CPU tests before any model execution. The successful preceding short probe (commit `c8ec8053b2b84c01087c7a393ebb4ebe7537327c`, archive SHA256 `96c0c4553e5dd96f28982ac9b4a6504f30807e6b4f870a97198678382e554a2e`) produced `hello` and EOS using two manually selected tokens. It did not exercise the normal adapter/Transformers generation machinery. This new exploratory technical smoke test addresses that specific gap, without any held-out study prompts or evaluator execution.

## Frozen inputs and decoding

Run these prompts once each, in order, stopping the entire attempt at the first unsuccessful result:

| ID | Prompt | Expected cleaned response |
| --- | --- | --- |
| TECH-01 | `Reply with exactly one word: hello.` | `hello` |
| TECH-02 | `Reply with exactly these three words: red green blue.` | `red green blue` |
| TECH-03 | `Reply with exactly these numbers separated by spaces: 1 2 3 4 5 6 7 8.` | `1 2 3 4 5 6 7 8` |

Use the **unchanged `ApertusAdapter.generate`** method with `max_new_tokens=32`, `do_sample=False`, `enable_thinking=False`, seed 42 separately for each prompt. Load/promote the model once for this attempt. Do not manually select tokens, replace generation processors, override cache settings, change the checkpoint generation configuration or add warm-up forwards. The adapter supplies only its usual decoding kwargs; record them and the full checkpoint generation configuration. Cache behavior follows those defaults, rather than the preceding manual probe's explicit `use_cache=False`.

A transparent temporary wrapper around the loaded model's `generate` method records prompt/output token IDs, delegates exactly once with identical arguments, and returns the same object. It changes no source or numerical outputs and is removed after each prompt. Require TECH-01's input IDs to equal those of the preceding short probe. No prompt substitution, fallback, retries or continuation from a partial attempt.

## Environment and precision intervention

Use the existing free Colab T4 environment and six cached shards of `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`. No paid resources or additional dependency installation. Before loading, verify the preceding successful ZIP's exact SHA256 and complete 12-entry internal manifest. Restore only that verified archive into the content-addressed dependency directory if necessary; do not overwrite earlier attempts. Require the same pinned core versions, full package inventory, CUDA version, frozen source/dataset hashes and GPU model as the preceding successful probe, plus compute capability 7.5. A different physical GPU UUID or driver is allowed and explicitly recorded, as in the preceding amended functional probe. No exact same-board causal inference is planned.

The existing resource preflight requires at least 13 GiB free CUDA memory, 8 GiB available RAM and 22 GiB free disk. Use a fresh worker process and full GPU placement without CPU/disk offload. Load NF4 double quantization from FP16 exactly as before. Fingerprint all 217 packed quantized weights and states and require equality with the successful short probe. Promote already loaded non-quantized floating parameters and registered buffers to FP32 using the unchanged `colab_fp32_probe.promote` helper; set quantized linear computation to FP32. No re-quantization or reloading source weights in wider precision. Require all parameters on GPU, FP32 floating parameters/buffers, unchanged packed state after promotion and again after this attempt, and at least 1 GiB free CUDA memory after promotion. Request IEEE FP32 matrix precision and disable reduced-precision FP16 reduction, restoring those settings at exit. Initial adapter loading metadata describes FP16 loading, separately from the recorded FP32 intervention.

## Observations and stop rules

Observe floating inputs/outputs at the language-model and output-head module boundaries and the outer model forward output. Also inspect accessible `Cache.layers[*].keys/values` at outer forward entry/exit, without mutating them; record actual cache class, accessible cache-tensor count, input shape and supplied `use_cache` per forward. These hooks do not observe every internal operation or arbitrary unrecognized cache storage. Exclude attention-mask keys, whose negative infinity can be intentional. Log every observed event and retain the last 100. Stop immediately on NaN/Inf, an observed floating dtype other than FP32, more than 32 forwards for one prompt, an unexpected adapter interface/decoding override, runtime error, resource failure or fingerprint mismatch. No attempt to sanitize outputs, reduce precision, offload or retry.

For each returned answer, preserve the adapter result (cleaned/raw text, token counts, latency, peak memory), actual generated IDs, EOS IDs and four separate descriptive checks: nonempty cleaned response; no unknown token selected; terminal EOS within the 32-token cap; and exact expected answer after case-folding and normalization of whitespace. Extra punctuation or prose fails the exact-answer check. If any check fails, preserve that answer and stop before the next prompt. These are technical instruction/output checks, not a false-premise evaluator or general semantic audit. `smoke_complete` requires all three returned answers to pass all four checks and all numerical/provenance gates; code zero is reserved for this outcome.

## Preservation and interpretation

Use the exclusive directory `results/colab-fp32-generation-smoke-2026-10-06`. Retain partial reports and traces, seal a ZIP, internal SHA256 manifest and ZIP checksum after handled errors, and return nonzero on any incomplete/failed attempt. A terminated runtime may prevent sealing. Archive only this attempt's texts, observations, metadata and launcher/protocol/helper copies, never model weights or credentials. Never delete or overwrite an existing attempt. Record total attempted forwards and each prompt's forwarded/captured generation calls. Peak memory is measured per adapter call because the frozen adapter resets the peak counter; do not interpret the final allocator reading as an attempt-wide peak.

Success would establish only that this intervention can complete these three short technical requests through the normal generation path in the recorded environment. It would not establish stability for longer contexts, 96-token outputs, all study prompts, other hardware, or semantic robustness. The original Phase 3B frozen automated result remains 24 false-premise prompts (16 PASS / 6 POTENTIAL_FAILURE / 2 DETECTED_FAILURE) and 6 PASS controls: 30 records, 22/6/2 overall. Its evaluator, dataset, raw results and `phase3b-preregistered` / `ed64ba8` checkpoint remain unchanged; no rerun/rescore is authorized by this smoke test. Further experiments need another prospective plan.

Source references: [frozen project adapter](https://github.com/Alextsvr/apertus-stresslab/blob/c8ec8053b2b84c01087c7a393ebb4ebe7537327c/src/stresslab/models.py), [pinned Apertus model](https://github.com/swiss-ai/transformers/blob/3797303dda74844e3d1f8977ff5518bb91f818b4/src/transformers/models/apertus1p5/modeling_apertus1p5.py).
