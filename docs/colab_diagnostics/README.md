# Colab T4 numerical diagnostics — post-run publication, 2026-10-06

**A finite FP32 projection result outside FP16's representable range explains one directly observed NF4/FP16 overflow. Promoting the loaded model's floating parameters, buffers and quantized computation to FP32 produced valid short answers and one complete 96-token cached technical continuation on a free Colab T4.** This is a numerical diagnostic result, not evidence that false-premise reasoning is repaired or that the model is generally stable.

This page publishes eight preserved attempts, including both preflight stops. Original ZIP bytes, archived launcher/protocol versions, internal manifests, stored outputs and tensor observations remain intact. Per-attempt reports and integrity metadata are byte-exact copies of the local post-run audits. Statements in those reports about what was then local, pending or not yet executed describe their creation-time state; this dated overview describes the completed chain. All new post-run interpretation is AI-assisted, not independent human adjudication.

## Chain of evidence

| Attempt | Registered procedure commit | Observed result | Report |
| --- | --- | --- | --- |
| October 5 NF4/INT8 comparison | `fdbb94c` | All 30 NF4/FP16 answers collapsed to 96 unknown tokens; INT8/FP16 returned assessable text | [Comparison](colab-t4-2026-10-05/post_run_report.md) |
| NF4/FP16 boundary diagnostic | `562fc48` | First observed non-finite boundary: third decoder block `mlp.down_proj`, one positive infinity | [Diagnostic](colab-4bit-diagnostic-2026-10-06/post_run_report.md) |
| Isolated projection replay | `81bc476` | Original and ordinary FP16 outputs match; FP32 reference finite at 117986.203125, casting it to FP16 reproduces infinity | [Replay](colab-layer-replay-2026-10-06/post_run_report.md) |
| FP32 attempt 1 | `956b6d5` | Preceding-archive gate stop; zero forwards, no model load | [Attempt 1](colab-fp32-activation-probe-2026-10-06/post_run_report.md) |
| FP32 attempt 2 | `3ed0ae3` | Cross-session GPU-identity gate stop; zero forwards, no model load | [Attempt 2](colab-fp32-activation-probe-2026-10-06-attempt2/post_run_report.md) |
| FP32 attempt 3 | `c8ec805` | Manual `hello` + EOS, two forwards, 2070 finite FP32 observations | [Short probe](colab-fp32-activation-probe-2026-10-06-attempt3/post_run_report.md) |
| Normal adapter smoke | `b864d89` | Three exact answers with EOS, 22 forwards, 25394 finite FP32 observations, active KV cache | [Smoke](colab-fp32-generation-smoke-2026-10-06/post_run_report.md) |
| Normal adapter 96-token probe | `86c3069` | 96 actual tokens/forwards, 95 cached decoding forwards, 111584 finite FP32 observations | [96-token probe](colab-fp32-generation-long-2026-10-06/post_run_report.md) |

The replay's coordinate `[0, 0, 2012]` was positive infinity in the original bitsandbytes FP16 output and ordinary FP16 linear output, 117986.203125 in the FP32 reference using common represented operands, and positive infinity after converting that FP32 result to FP16. FP16's largest finite value is 65504. The bitsandbytes FP32 path produced 117986.359375, also finite. No bitsandbytes-specific defect is needed to explain this coordinate's representation overflow. This does not prove general kernel correctness or identify the cause of every October 5 unknown-token response.

The FP32 intervention did not re-quantize or reload wider-precision source weights. It promoted already loaded floating parameters/buffers and set the existing 4-bit linear modules' computation to FP32. All 217 packed weight/state fingerprints matched before promotion, after promotion and after both normal-generation checks. Initial loading metadata therefore still says FP16, while the separate intervention/trace describes FP32 execution. The original and intervention runs can use different backend kernels/rounding.

## Completed 96-token technical result

The fixed prompt requested integers 1 through 200 separated by spaces. The model returned the correct sequence 1 through 35 plus a raw trailing space, reaching exactly 96 output tokens and stopping at the intentional cap. Its 91-token prompt required one prefill, followed by 95 one-token decoding forwards with `DynamicCache`. All 12224 observed accessible KV-cache events were finite FP32; cache length reached 186 positions. There was no unknown token or EOS in the continuation. The generation-call peak was **10.773 GiB**, and instrumented generation took **67.342 seconds**.

This measures one real 96-token cached continuation, not merely a configured upper bound. It does not test long contexts, arbitrary prompts, other hardware/software, semantic robustness, repeated-run stability or the original study suite. Hooks cover selected boundaries and accessible cache storage, not every internal operation. The peak excludes an attempt-wide loading/promotion peak because the frozen adapter resets its counter. Instrumented latency is not an uninstrumented performance benchmark.

The FP32 probes used the same recorded T4 UUID/driver as each other, but a different physical board from the original FP16 replay. The allowed cross-session UUID change is explicit in the prospective amendment; no same-board contemporaneous FP16 control or exact cross-session causal comparison is claimed.

## Separate stored study counts

| Configuration/run | 24 false-premise prompts: PASS / POTENTIAL / DETECTED | Six controls | Overall |
| --- | --- | --- | --- |
| Original Phase 3B: NF4/BF16 + CPU offload | 16 / 6 / 2 | 6 PASS | 22 / 6 / 2 |
| October 5 Colab NF4/FP16 | 0 / 24 / 0 | 6 POTENTIAL | 0 / 30 / 0 |
| October 5 Colab INT8/FP16 | 16 / 5 / 3 | 6 PASS | 22 / 5 / 3 |

These are **stored frozen labels**, not newly evaluated responses. The October 5 experiment reused already observed prompts and is not new held-out evidence. NF4's empty cleaned outputs invalidate a meaningful semantic comparison between its condition and INT8; its 30 POTENTIAL labels are not 30 semantic acceptances. INT8's separate AI-authored audit found 14 corrections, three explicit acceptances, five responses without correction and two ambiguous responses, with all six controls correct. Those secondary annotations do not replace stored counts and have no independent human sign-off.

All later diagnostics used technical prompts, no study evaluator and no study-suite inference. Original Phase 3B raw hashes/counts, source/dataset, annotations and `phase3b-preregistered` / `ed64ba8` remain unchanged. No unquantized/higher-precision semantic confirmation is claimed by these technical probes.

## Evidence and model-free verification

Download the original ZIPs and checksum sidecars from [evidence/colab](../../evidence/colab/). [The catalog](../../evidence/colab/catalog.json) records every full SHA256, file size, procedure commit, report hash and source mapping. Complete tensor streams and replay arrays are inside their original ZIPs, avoiding duplicate large extracted logs in the repository. Model weights and credentials are excluded. October 5's ZIP checksum was captured locally after receipt; no external sender checksum was received for that one archive. All seven later sender checksums matched.

With Python 3.11+ and a Git clone containing the procedure history:

```text
python scripts/verify_colab_evidence.py
```

The stdlib verifier checks all eight ZIP hashes/sizes, checksum sidecars, complete internal manifests, exact archived source/protocol blobs against their recorded commits, available preflight source fingerprints, report/metadata hashes and stored October 5 counts. It also verifies original Phase 3B with the existing model-free verifier. It never imports an inference adapter, runs a model, invokes a semantic evaluator or updates stored scores. Hashes anchor the received bytes, not pre-capture immutability or universal scientific validity.

## Next Colab session

Use [the unified notebook](../../notebooks/colab_evidence_and_setup.ipynb) to review/verify evidence on CPU, and optionally prepare the recorded free T4 environment and model cache. Preparation flags default to false; there is no automatic generation, replay of historical attempts or full-suite cell. The bootstrap is syntax-checked locally, not GPU-validated in a new Colab session. A future changed base runtime/package inventory must be recorded and considered in a new prospective plan before inference. The old `colab_smoke_test.ipynb` is historical and is not the current setup route.

Any next semantic experiment needs its own prospective prompts/configuration/quality gates and separate outputs. The present publication executes no further inference or rescoring.
