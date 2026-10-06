# NF4 with FP32 activations: short probe, attempt 3

This is a post-run, AI-assisted technical audit of evidence supplied by the project owner. It is not a human semantic audit, a benchmark result, or a rescore of Phase 3B. The prospectively committed procedure was followed by a successful two-forward technical probe. No inference or evaluator execution was performed while preparing this report.

## Registered procedure and provenance

- Procedure commit: `c8ec8053b2b84c01087c7a393ebb4ebe7537327c`; protocol: `docs/colab_fp32_probe_protocol.md`.
- Model: `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`.
- Supplied archive: `colab-fp32-activation-probe-2026-10-06-attempt3.zip`, 189,981 bytes.
- Archive SHA256: `96c0c4553e5dd96f28982ac9b4a6504f30807e6b4f870a97198678382e554a2e`.
- Sender checksum and all 12 internal manifest hashes match. All manifest entries are accounted for. The launcher, protocol, four helper files and 16 frozen source hashes match the recorded commit.
- Preceding paired replay archive: SHA256 `ef98d1fe60a71adf4abe2082a9a4974d8ec8ba4eb97e04e596a1d6cd83b2ae23`; locally verified. Software inventories, frozen sources, model revision, dataset hash and the 70-token prompt encoding match the replay.
- Earlier unsuccessful attempts remain separate: attempt 1 (`1fc0b20ba6df520028e8b0e3898a7087baf5122d850937b8b33a84f3f88136e1`) stopped at the dependency-archive gate; attempt 2 (`a0b21d6664df7bf4f225cac504517a7dd2e6489ac0dfe70e14b2db3f6c13157c`) stopped at the cross-session hardware-identity gate. Both attempted zero model forwards. They are retained, not replaced by this successful attempt.

## Intervention and environment

The model was first loaded with the existing NF4/FP16 procedure. Before inference, non-quantized floating parameters and registered buffers were promoted to FP32; all 217 quantized linear modules were configured for FP32 computation. The fingerprints of every packed NF4 weight and quantization state are identical before promotion, after promotion and after continuation. The previously implicated layer's fingerprint also matches the paired replay. Promoting already loaded FP16 values preserves those represented values; it does not restore precision lost on loading or remove quantization error.

The `model_loading_info` field records the initial FP16 loading configuration. The subsequent promotion record and tensor observations describe the FP32 execution in this probe. FP32 IEEE precision was requested and reduced-precision FP16 reduction disabled.

Recorded environment: Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 5.14.0.dev0 from the pinned Swiss-AI fork, Accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.22.2 and NumPy 2.1.3. The GPU was a Tesla T4 with compute capability 7.5 and driver 580.82.07.

The GPU UUID changed from `GPU-577c04ea-9cd1-bbc6-7826-547a9c671dfa` in the replay to `GPU-282e8aac-f790-b267-994f-ee18450a0f14` in this attempt. The GPU class and driver remained the same. Attempt 3 prospectively permitted and recorded this cross-session change. This is not a same-physical-GPU paired comparison.

## Observed result

| Quantity | Stored result |
| --- | --- |
| Prompt | `Reply with exactly one word: hello.` |
| Method | Manual greedy argmax, no generation processors, `use_cache=False` |
| Output cap | 8 new tokens |
| Model forward calls | 2 |
| Generated token IDs | `[29706, 68]` |
| Raw response | `hello<\|assistant_end\|>` |
| Cleaned response | `hello` |
| Stop reason | EOS |
| Worker exit code / outcome | 0 / `short_probe_complete` |
| Observed floating-tensor events | 2,070, all FP32, zero NaN/Inf events |
| Peak allocated GPU memory | 11.424 GiB |
| Allocated / free GPU memory after promotion | 10.389 / 2.984 GiB |
| CUDA-visible total memory | 14.563 GiB |

All four stored descriptive output checks passed: the cleaned response was nonempty, no unknown token was selected, EOS occurred within the cap, and the cleaned response matched `hello` case-insensitively. These checks are not the frozen false-premise evaluator.

At `model.language_model.layers.2.mlp.down_proj`, both observed outputs were finite FP32. Their maximum was 118004.5078125, which exceeds the FP16 finite limit of 65504 without clipping or becoming Inf. This is consistent with the preceding replay's finding that a large, finite result cannot be represented in FP16. The current number is not an exact replay of the old operands: promotion also changed upstream computation precision.

The selected token was `hello` on step 1 and assistant-end on step 2. Unknown-token ranks were 1138 and 811 respectively; it was never selected. The finite minimum seen in the final logits includes the model's intentional finite padding value for the unused vocabulary tail, not an observed infinity.

## Interpretation and limits

The NF4 model with FP32 floating parameters, buffers and computation successfully completed this short technical prompt on the tested free Colab T4 environment. The original overflow boundary was finite, every observed floating boundary was FP32 and finite, and packed quantized weights and state remained unchanged. Together with the paired replay, this supports a practical precision-based workaround for the observed numerical failure on this prompt.

The observation covers one prompt, two forwards and two output tokens. It does not establish stability for 30 study prompts, longer contexts or longer generations, and does not test the adapter's normal `generate` path or its generation processors. Hooks observe selected module boundaries rather than every internal operation. There was no same-session, same-physical-GPU FP16 control in this attempt. Changing computation dtype can also change bitsandbytes kernel selection and dequantization rounding; unchanged packed state does not imply identical runtime operands or kernels. The successful probe does not by itself prove the cause of all earlier `<unk>` outputs or demonstrate semantic robustness.

## Separation from frozen Phase 3B

The original preregistration checkpoint remains `phase3b-preregistered` / `ed64ba8cb382073d652b24e647a898c94f8ce186`. The evaluator, dataset and original raw results were not changed or executed again.

Original stored results remain 30 records: 24 false-premise prompts with 16 PASS, 6 POTENTIAL_FAILURE and 2 DETECTED_FAILURE, plus 6/6 PASS controls. Overall counts remain 22 PASS, 6 POTENTIAL_FAILURE and 2 DETECTED_FAILURE. This diagnostic adds no records or scores to that frozen result.

The supplied ZIP, sender checksum and all archive entries are preserved byte-for-byte under `results/colab-fp32-activation-probe-2026-10-06-attempt3`. This report and verification metadata are held separately under `docs/audits/colab-fp32-activation-probe-2026-10-06-attempt3`. Preservation and interpretation are local; they do not constitute a new full experiment or publication.
