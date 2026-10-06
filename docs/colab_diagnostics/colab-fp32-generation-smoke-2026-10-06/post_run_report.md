# NF4 / FP32 normal-generation smoke test: post-run report

The prospectively registered three-prompt technical smoke test completed through the unchanged `ApertusAdapter.generate` and the loaded model's normal `generate` implementation. All three requested answers were returned exactly and stopped on EOS. All observed floating module-boundary and accessible KV-cache tensors were finite FP32. This is an AI-assisted technical audit of evidence supplied by the project owner, not a human semantic audit or a false-premise benchmark.

## Provenance and verification

- Procedure commit: `b864d896245bc956c24b438c6dc92602935a87a1`; protocol: `docs/colab_fp32_smoke_protocol.md`.
- Model: `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`.
- Supplied archive: `colab-fp32-generation-smoke-2026-10-06.zip`, 559,108 bytes.
- Archive SHA256: `e50e058df196207d0c9712f7ea4950a0168a39989b5596f800622a849d880b4a`.
- The sender checksum and all 13 internal manifest hashes match, with complete coverage. The launcher, protocol and five helper files match the recorded commit byte-for-byte. All 16 frozen source hashes match that commit.
- The preceding successful short-probe archive SHA256 is `96c0c4553e5dd96f28982ac9b4a6504f30807e6b4f870a97198678382e554a2e`; its locally preserved bytes match the dependency anchor. Core versions, full package inventory, CUDA version, model revision, source/dataset hashes and GPU identity match the preceding probe. TECH-01's 70 input token IDs also match.
- Preflight was recorded at `2026-10-06T10:49:56.860435+00:00`; the worker finished at `2026-10-06T10:52:26.960679+00:00`, followed by parent sealing at `10:52:28.684895+00:00`.

## Intervention and generation path

The existing NF4 double-quantized model was loaded once from FP16 on the T4 without offload. Before any forward, loaded floating parameters and registered floating buffers were promoted to FP32 and the existing 4-bit linear modules configured for FP32 computation. Packed quantized weights and quantization states for all 217 modules are identical before promotion, after promotion and after the attempt, and equal those in the preceding short probe. This preserves the represented loaded values; it does not recover precision lost on initial FP16 loading or remove quantization error.

IEEE FP32 matrix precision was requested and reduced-precision FP16 reduction disabled. The initial loading metadata still records FP16; the separate promotion and observation records describe the FP32 execution. Memory after promotion was 10.389 GiB allocated and 2.984 GiB CUDA-free out of 14.563 GiB CUDA-visible total.

Each prompt used seed 42, thinking disabled, greedy decoding and a 32-new-token cap. The frozen adapter supplied only `max_new_tokens=32` and `do_sample=False` to the normal model generation method. A transparent observational wrapper delegated once per prompt and preserved the returned IDs without changing the arguments or output. No manual token selection, warm-up forwards, alternate prompts, changed generation configuration, fallback or retry was used. The checkpoint generation configuration is preserved in `smoke.json`; its nullable fields do not by themselves describe all runtime-resolved defaults.

Actual forward metadata resolves the cache behavior: **`use_cache=True` with `DynamicCache`** on all 22 forwards. Each prompt began with a full prefill of 70, 73 or 88 tokens, followed by one-token cached decoding. Subsequent forwards exposed 64 cache tensors (key/value storage for 32 layers). This exercises a different path from the preceding manual continuation, which explicitly disabled caching.

## Observed outputs

| Case | Cleaned answer | Input tokens | Output tokens, including EOS | Forwards | Generation latency (s) | Per-call peak GPU memory (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| TECH-01 | `hello` | 70 | 2 | 2 | 3.729 | 10.765 |
| TECH-02 | `red green blue` | 73 | 4 | 4 | 3.058 | 10.765 |
| TECH-03 | `1 2 3 4 5 6 7 8` | 88 | 16 | 16 | 11.303 | 10.771 |

All three raw answers end in `<|assistant_end|>` (token 68). Each prompt invoked normal `generate` exactly once. All four registered output checks passed independently for each returned answer: nonempty cleaned response, no unknown token selected, terminal EOS within the cap, and exact expected response after case/whitespace normalization. These are descriptive technical output checks, not evaluator PASS labels.

There were **22 attempted forwards: 3 prefills and 19 cached decoding forwards**. The full log contains **25,394 floating-tensor events, all FP32 on CUDA with zero NaN/Inf**, including **2,624 accessible KV-cache events**. Event counts by case are 2,262, 4,588 and 18,544. The maximum observed cache sequence lengths were 71, 76 and 103. The final longest cache tensor shape was `[1, 8, 103, 128]`; its keys and values were finite FP32.

The previously implicated third-block `mlp.down_proj` produced finite FP32 outputs in all 22 observed calls. The TECH-01 prefill retained a maximum of 118004.5078125, exceeding FP16's finite limit of 65504 without clipping or becoming infinity. Its cached second forward processed only one new input token; this is not an identical full-prefix arithmetic replay of the earlier non-cached probe.

The maximum of the three recorded generation-call peaks was **10.771 GiB**. The frozen adapter resets peak memory at each call; this is not an attempt-wide peak and cannot be used to infer the peak during load/promotion. Final allocated memory was 10.397 GiB. The three generation latencies total 18.090 seconds and exclude model loading, promotion and preflight.

## Environment and interpretation limits

Recorded environment: Python 3.13.15; PyTorch 2.11.0+cu130 and CUDA 13.0; pinned Swiss-AI Transformers 5.14.0.dev0; Accelerate 1.15.0; bitsandbytes 0.50.2; tokenizers 0.22.2; NumPy 2.1.3. Hardware was Tesla T4, compute capability 7.5, driver 580.82.07, UUID `GPU-282e8aac-f790-b267-994f-ee18450a0f14`. UUID and driver were unchanged from the immediately preceding successful FP32 short probe. That earlier probe used a different physical board from the original FP16 layer replay, so this chain is not a same-board FP16-versus-FP32 causal experiment.

This result establishes that the tested NF4/FP32 intervention can produce these three short technical answers through the normal adapter and cached generation path in the recorded free Colab environment. It extends the earlier manual two-token probe and supports this configuration as a working candidate for further prospectively specified checks.

It does not establish stability for arbitrary prompts, long contexts, 96-token outputs, the full study suite, other hardware or other software versions. The longest actual answer was 16 tokens including EOS; the 32-token cap was not reached. No full token-cap stress test occurred. Hooks cover selected boundaries and recognized cache storage, not every internal operation or processor. No contemporaneous FP16 control was run. These technical echo/sequence requests do not establish comparative reasoning quality or semantic robustness. Changes in computation dtype may change backend kernels and dequantization rounding despite identical packed state.

## Separation from original Phase 3B and preservation

The original preregistration remains `phase3b-preregistered` / `ed64ba8cb382073d652b24e647a898c94f8ce186`. Original stored results remain 30 records: 24 false-premise prompts with 16 PASS, 6 POTENTIAL_FAILURE and 2 DETECTED_FAILURE, plus six PASS controls (22/6/2 overall). No original evaluator, dataset, annotation or raw result was modified, rerun or rescored. The three technical prompts add no records or scores to Phase 3B.

The sender ZIP/checksum and every supplied archive entry are preserved byte-for-byte under `results/colab-fp32-generation-smoke-2026-10-06`. This report, copied observations, verification metadata and a separate audit manifest are saved under `docs/audits/colab-fp32-generation-smoke-2026-10-06`. The post-run audit performed no local inference or scoring. This preservation is local; the registered protocol remains the publicly committed pre-run artifact. No further experiment is executed or authorized by this report.
