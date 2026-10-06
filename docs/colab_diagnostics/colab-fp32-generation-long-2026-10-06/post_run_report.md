# NF4 / FP32 96-token cached generation: post-run report

The prospectively specified single technical prompt reached exactly 96 generated tokens through the unchanged `ApertusAdapter.generate` and normal model generation path. All 96 observed forwards, including 95 cached decoding forwards, completed with finite FP32 boundary/cache observations. The response was the correct sequence from 1 through 35, followed by a raw trailing space, and stopped at the registered token cap. This is an AI-assisted technical audit of supplied evidence, not a human semantic audit or a study-suite result.

## Provenance

- Procedure commit: `86c30699c412ac7e9ae78f1eb025de66057634a2`; protocol: `docs/colab_fp32_long_protocol.md`.
- Model: `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`.
- Supplied archive: `colab-fp32-generation-long-2026-10-06.zip`, 1,899,546 bytes.
- SHA256: `3ef82701b2049d42527c1b601176e52f796b679e2060578dea69bc9539045afc`.
- Sender checksum and all 15 internal hashes match, with complete manifest coverage. The launcher, protocol, six helper files and all 16 frozen source hashes match the recorded commit. The notebook helper is included among these files.
- Preceding smoke archive SHA256: `e50e058df196207d0c9712f7ea4950a0168a39989b5596f800622a849d880b4a`; the local copy matches. Core versions, full package inventory, CUDA, frozen source/dataset hashes, model revision and physical GPU identity match the preceding smoke.
- Preflight: `2026-10-06T11:05:29.838926+00:00`; worker completion: `11:08:52.014587+00:00`; parent sealing: `11:08:54.183253+00:00`.

## Registered prompt and intervention

Prompt: `Write the integers from 1 to 200 in ascending order, separated by a single space. Output only the numbers, without commentary.`

The loaded pinned tokenizer represented the complete requested target in 691 tokens, exceeding the 96-token cap. One normal adapter call used seed 42, thinking disabled, greedy decoding and `max_new_tokens=96`. Only the normal `max_new_tokens` and `do_sample` overrides were supplied. EOS was neither suppressed nor forced, and no minimum length, custom decoding processor, manual token selection, warm-up or retry was used. A transparent observational wrapper recorded actual prompt/output IDs and returned the original generation object unchanged.

The model was loaded once with the preceding NF4 double-quantization/FP16 loading configuration and full T4 placement. Before any forward, loaded floating parameters and registered buffers were promoted to FP32 and all quantized linear modules configured for FP32 computation, using the unchanged helper. All 217 packed weight/state fingerprints match the preceding smoke and remain identical before promotion, after promotion and after this attempt. The intervention preserves the loaded representable values; it does not recover loading precision or eliminate quantization error. Initial adapter metadata describes FP16 loading separately from actual FP32 execution. IEEE FP32 matrix precision was requested and reduced-precision FP16 reduction disabled.

## Observed result

| Quantity | Recorded result |
| --- | --- |
| Worker outcome / exit code | `long_probe_complete` / 0 |
| Normal adapter generation calls | 1 |
| Input tokens | 91 |
| Actual new output tokens | 96 |
| Forward calls | 96: one prefill and 95 cached decoding forwards |
| Stop reason | `token_cap` |
| Cleaned response | Integers 1 through 35, separated by spaces |
| Raw response ending | `33 34 35 `, including the trailing space |
| Generation-call latency | 67.342 seconds |
| Generation-call peak GPU memory | 10.773 GiB |
| Floating tensor observations | 111,584, all FP32 on CUDA, zero NaN/Inf events |
| Accessible KV-cache observations | 12,224, all finite FP32 |
| Largest observed cache sequence length | 186 |

All six registered checks passed independently: nonempty cleaned text; no unknown token selected; exactly 96 actual token IDs; no EOS selected; cleaned text is a prefix of the requested sequence; and all 96 forwards with 95 observed cached decoding forwards. Stopping before 200 is the intended cap-limited result, not a claim that all requested integers were generated. These are technical checks, not evaluator PASS labels.

Actual forward metadata records `use_cache=True` and `DynamicCache`. The first forward used shape `[1, 91]`; all subsequent forwards used `[1, 1]` with 64 accessible cache tensors. At each forward exit, all 32 layers' key/value caches had the expected sequence length `90 + forward_index`; the final shape was `[1, 8, 186, 128]`. The last generated token is selected after the final forward and is not fed back into the cache, explaining a cache length of 186 rather than 187. The saved forward-progress record confirms 96 completed forwards and the same event count/peak memory.

The previously implicated third-block `mlp.down_proj` had 96 finite FP32 observed outputs. Its prefill maximum was 118004.5078125, exceeding FP16's finite limit of 65504 without clipping or becoming infinity. This reflects the wider upstream computation and is not an identical-operands replay of the earlier FP16 experiment.

Post-promotion memory was 10.389 GiB allocated and 2.984 GiB CUDA-free out of 14.563 GiB CUDA-visible total; final allocated memory was 10.397 GiB. The frozen adapter resets peak memory at the start of generation. The 10.773 GiB figure therefore covers the generation call, not an attempt-wide loading/promotion peak. The 67.342-second latency includes the instrumented generation and excludes setup/loading/promotion; it is not an uninstrumented performance benchmark.

## Environment and limits

Recorded environment: Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0, pinned Swiss-AI Transformers 5.14.0.dev0, Accelerate 1.15.0, bitsandbytes 0.50.2, tokenizers 0.22.2 and NumPy 2.1.3. GPU: Tesla T4, compute capability 7.5, driver 580.82.07, UUID `GPU-282e8aac-f790-b267-994f-ee18450a0f14`. UUID/driver were unchanged from the immediately preceding FP32 smoke. This board differs from the original FP16 paired replay's board; no contemporaneous same-board FP16 control was performed here.

This result establishes an actual 96-token cached technical continuation for this one prompt under the recorded intervention/environment. It extends the short/manual and three-prompt normal-generation checks and supports the NF4/FP32 configuration as a working candidate for further prospectively specified experiments.

It does not establish robustness across arbitrary prompts, long contexts, other devices/software or semantic performance on the study suite. The prompt had 91 input tokens and the cache reached only 186 positions; 96-token output coverage is not long-context coverage. Hooks observe selected boundaries and recognized cache storage, not every internal operation or generation processor. Backend/kernel selection and dequantization rounding may differ with computation dtype despite identical packed state. The technical sequence task supplies no estimate of false-premise error rates and does not by itself prove the cause of every earlier unknown-token response.

The notebook's trailing dictionary is the return value displayed by `runpy.run_path`, separate from the sealed worker report. It is incidental UI output, not an additional inference, failure or retry. The supplied original output and archive are not rewritten to hide it.

## Separation and preservation

Original Phase 3B remains 30 records: 24 false-premise prompts with 16 PASS, 6 POTENTIAL_FAILURE and 2 DETECTED_FAILURE, and six PASS controls (22/6/2 overall). Its preregistration remains `phase3b-preregistered` / `ed64ba8cb382073d652b24e647a898c94f8ce186`. No original evaluator, dataset, annotation or raw result was changed, rerun or rescored. This technical continuation contributes no study-suite records or scores.

The ZIP, sender checksum and all supplied entries are preserved byte-for-byte under `results/colab-fp32-generation-long-2026-10-06`. The separate local audit folder `docs/audits/colab-fp32-generation-long-2026-10-06` contains this report, copied observation/provenance JSON and tensor stream, integrity metadata and its own SHA256 manifest. Preparing this audit performed no inference or scoring and published no new post-run result. No further experiment is executed by this report.
