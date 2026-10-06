# Paired NF4 projection replay: post-run findings

Date: 2026-10-06. Prepared by Codex; AI-assisted technical audit, not independent human review.

**Final-output FP16 range overflow is established at the observed projection coordinate in this technical probe.** With the same captured finite activation and the same common reconstructed weight values, the FP32 reference is `117986.203125`, above FP16's maximum finite value `65504`. Both the original bitsandbytes FP16 projection and the ordinary FP16 reference return `+Inf` there. Casting the saved FP32 result to FP16 independently reproduces the infinity.

## Evidence and provenance

The prospective plan and launcher were committed and published before execution at `81bc476c0d765667f82c22c8b9e1beaaf2b9745b`. The supplied `colab-layer-replay-2026-10-06.zip` is 6,226,343 bytes with SHA256 `ef98d1fe60a71adf4abe2082a9a4974d8ec8ba4eb97e04e596a1d6cd83b2ae23`. The separately supplied sender checksum matches. All 16 internal file hashes match and the manifest covers the complete 17-entry archive. Protocol, launcher and all three archived helpers match the committed blobs; all 16 frozen source hashes match.

The ZIP, checksum and extracted payload are preserved under `results/colab-layer-replay-2026-10-06` and its adjacent archives. Saved input and all five output arrays were loaded without pickle and their data fingerprints verified. This report and `integrity.json` are separate derivatives. The original Phase 3B evaluator, dataset, raw results, labels and preregistration remain unchanged; no study was rerun or rescored.

The preceding diagnostic archive hash matches the recorded anchor `50f77cd2ba0aa919f53ae4a466a39067576f099aea326c3d8277dce65bdd109a`. Package inventory, GPU identity, core versions, CUDA version, frozen source/dataset and model revision match the preceding probe. Input token IDs match exactly, and the 99-event trace is byte-identical to that probe's trace. Packed target weights and quantization-state fingerprints before/after replay match; the captured input fingerprint also remains unchanged.

## Fixed execution and observed results

One model prefix forward was attempted on `Reply with exactly one word: hello.`, with thinking disabled and no cache. It stopped immediately after `model.language_model.layers.2.mlp.down_proj`, the third decoder block's projection. Three isolated matrix multiplications and one result cast followed; no final logits, generated text, runner or evaluator were produced/invoked. The model remained NF4 with double quantization, FP16 compute and full GPU placement on the Tesla T4. Peak allocated GPU memory was 8.349 GiB. Process code was 0; recorded outcome was `paired_replay_complete`, finishing at 08:59:58 UTC.

The captured input was finite FP16, shape `[1, 70, 21504]`, range `[-0.022552490234375, 12824]`. Common reconstructed weights were finite FP16, shape `[4096, 21504]`, range `[-4.3125, 7.34375]`. Both ordinary references use these identical representable operands, casting them exactly to FP32 for the second reference. The original checkpoint without quantization was not loaded or compared.

At output coordinate `[0, 0, 2012]`:

| Computation | Recorded value | Whole-tensor NaN / +Inf / -Inf |
|---|---:|---|
| Original bitsandbytes FP16 | +Inf | 0 / 1 / 0 |
| Ordinary FP16, reduced-precision reduction disabled | +Inf | 0 / 1 / 0 |
| Ordinary FP32 with common reconstructed weights | 117986.203125 | 0 / 0 / 0 |
| That FP32 result cast to FP16 | +Inf | 0 / 1 / 0 |
| bitsandbytes with FP32 input/output | 117986.359375 | 0 / 0 / 0 |

Exactly one FP32-reference output exceeds the finite FP16 range, at the same coordinate as the original infinity. The original bitsandbytes FP16 tensor is byte-identical to the ordinary FP16-reference tensor, including all 286,720 elements. The archived cast tensor is byte-identical to an independently computed NumPy FP32-to-FP16 cast. These findings establish that final representation in FP16 cannot retain this projection result.

The FP16 reference and the FP32-then-FP16 cast are not byte-identical overall: 24,536 jointly finite entries differ because the numerical paths round differently. Their non-finite coordinates/signs coincide. The maximum finite-entry absolute difference from the common FP32 reference is 8.203125 and the FP16-reference finite-entry RMSE is approximately 0.01551; these finite-entry summaries exclude the overflowing element and do not quantify its error.

The separate bitsandbytes FP32 path is finite throughout; its maximum absolute difference from the common FP32 reference is 0.80078125 and RMSE approximately 0.002597. It uses the same packed weights/state, but dtype changes can change reconstruction rounding and kernel dispatch. It is supporting evidence, not an exact arithmetic oracle or an isolated test of accumulation precision alone.

## Interpretation and limits

For this captured layer and input, a bitsandbytes-specific bug is not needed to explain the observed infinity: ordinary FP16 arithmetic reproduces the entire original output, and an explicit FP32-to-FP16 cast reproduces the overflow. This does not establish that every bitsandbytes kernel is correct, or identify why this model/configuration produces such a large activation. It does not isolate the contribution of quantization relative to unquantized weights.

The result strengthens a numerical explanation for the October 5 NF4/FP16 generation collapse. It still does not trace that historical run's repeated `<unk>` tokens causally: those prompts were not replayed and this probe stopped before logits or generation. Historical raw counts and the separate INT8 semantic review remain unchanged.

An isolated finite FP32 projection is not proof of repaired generation. A later stabilization probe must retain a sufficient numerical range for the projection output and its downstream path, not merely compute internally in FP32 then cast this large result back to FP16. The pinned `Linear4bit` wrapper returns the original input dtype, so changing only its compute setting with FP16 inputs would still narrow the result. Any such model-level test needs a new prospective plan on technical prompts before another study is considered.
