# NF4/FP16 numerical diagnostic: post-run findings

Date: 2026-10-06. Prepared by Codex; AI-assisted technical analysis, not an independent human audit.

**A numerical failure was observed in one forward call on a new technical prompt.** The first observed non-finite tensor appeared at the output of `model.language_model.layers.2.mlp.down_proj` (the third decoder block): one positive infinity, no NaN and no negative infinity, in a `[1, 70, 4096]` FP16 tensor on `cuda:0`.

## Integrity and frozen plan

Prospective plan and launcher: commit `562fc48518b163a276e39a3f431acd1a1c7192c9`, published before execution. The downloaded ZIP is 22,295 bytes, SHA256 `50f77cd2ba0aa919f53ae4a466a39067576f099aea326c3d8277dce65bdd109a`. The separately supplied Colab checksum matches. All eight internal hashes match and the manifest covers the full payload. Plan and launcher match that commit byte-for-byte; all 16 frozen source hashes and the dataset hash match.

The ZIP, sender checksum and nine extracted files are preserved under `results/colab-4bit-diagnostic-2026-10-06` and its adjacent archives. JSON files in this audit directory are byte-exact evidence copies; this report and `integrity.json` are separate derivatives. No old raw file, evaluator, dataset or historical label was changed. The existing Phase 3B raw verification still reports 4/4 matching hashes and 22 PASS / 6 POTENTIAL / 2 DETECTED across 30 records.

## Observed execution

The fixed prompt was `Reply with exactly one word: hello.` Its processor chat template produced 70 input tokens. Model revision was `a411d838600baf0e3635a3daf66fb7c55fc97bb6`. NF4 double quantization and FP16 compute were used on a Tesla T4, with all model parameters on GPU and no CPU offload. Model metadata resolved to the requested revision. The new session's full package inventory is recorded in `preflight.json`; matching core versions do not establish an identical environment to October 5.

Checks found no NaN/Inf in 639 floating parameters or 868 accessible quantization-state tensors. This establishes finiteness of those checked stored values, not correctness of quantization or finiteness of every internally dequantized value.

All 99 observed tensor events were retained: the first 98 were finite; the final event contained one `+Inf` and triggered the planned stop. The decisive module boundaries were:

| Third block boundary | Finite minimum | Finite maximum | NaN / +Inf / -Inf |
|---|---:|---:|---|
| `up_proj` output | -3.1953125 | 28.296875 | 0 / 0 / 0 |
| xIELU `act_fn` output | -0.022552490234375 | 12824 | 0 / 0 / 0 |
| `down_proj` input | -0.022552490234375 | 12824 | 0 / 0 / 0 |
| `down_proj` output | -5344 | 34560 | 0 / 1 / 0 |

The last row's range describes only finite elements; it excludes the positive infinity. The xIELU output was finite in this probe. Large finite activation growth preceded the failing projection, but hooks do not reveal the internal operation that first overflowed or otherwise produced infinity.

The process finished at 08:42:55 UTC with code 0: this means successful completion of the diagnostic and sealing of evidence, including the detected failure. It does not mean successful model generation. Exactly one forward call was attempted, stopped at this boundary. No final logits, top tokens, unknown-token scores or generated text were produced. Neither the runner nor the evaluator was invoked.

## Interpretation and limits

Numerical failure is now directly observed for this NF4/FP16 configuration on this new prompt. The earlier October 5 NF4 condition's 30 empty cleaned answers and repeated `<unk>` output remain preserved generation failures, not 30 semantic false-premise acceptances. A numerical explanation for that earlier collapse is supported as a hypothesis, but is not proven: the study prompts were not replayed, its intermediate values were not recorded, and this diagnostic stopped before logits or autoregressive generation.

This probe cannot separate FP16 range overflow, quantization effects, dequantization behavior or kernel behavior. It does not establish that changing dtype or disabling quantization repairs the model. It does not alter the historical Phase 3B observations or invalidate the assessable October 5 INT8 answers.

A useful follow-up would compare the failing projection's arithmetic at different compute precisions with the same captured input and quantized weights, under a new prospective plan. That comparison has not been performed. Do not silently repeat this attempt or the 30-case study.
