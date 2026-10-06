# Prospective exploratory NF4/FP32 semantic study — 2026-10-06

## Question and scope

Does the intervention that kept NF4 packed weights intact and produced finite technical generations also produce assessable responses to new comparative false-premise prompts and neutral controls? This single-configuration exploratory study is designed with knowledge of earlier outputs. It is **not independent held-out validation**, a contemporaneous precision comparison, or a causal estimate of semantic improvement. No unquantized baseline is included.

This protocol, prompts and launcher must be committed and published before any of these prompts are submitted to the model. The run's preflight records the full execution commit and artifact hashes. No output is yet known at registration.

## Received preparation and technical dependency

The new-session preparation ZIP is `colab-next-session-preflight-20261006T114501627825Z.zip`, 8868 bytes, SHA256 `dc7c4e8cea9a98b8372b95dedcc120e9c9752631ab775aac6aa8f3ff7e388ca8`. It records 2026-10-06 11:45:01 UTC, checkpoint `dbabc122ec8834bca3e274f821507b38bf760c18`, no model load, T4 UUID `GPU-2d908a88-5692-d044-fee3-b9ba2e4cebf0`, driver 580.82.07, 14.461 GiB free CUDA memory, 11.064 GiB available RAM and 69.785 GiB free disk. Sender and internal checksums matched. All 746 recorded distribution entries, six primary versions, CUDA, frozen source/dataset hashes and model revision match the successful 96-token technical dependency. GPU UUID differs from that preceding session and is explicit provenance.

The successful preceding ZIP is `colab-fp32-generation-long-2026-10-06.zip`, SHA256 `3ef82701b2049d42527c1b601176e52f796b679e2060578dea69bc9539045afc`. Both dependencies are read from published evidence, checked by archive SHA256 and complete internal manifests. All 217 NF4 packed weight/state fingerprints must match that successful run before promotion.

Environment: Python 3.13.15; torch 2.11.0+cu130 / CUDA 13.0; transformers 5.14.0.dev0 from swiss-ai commit `3797303dda74844e3d1f8977ff5518bb91f818b4`; accelerate 1.15.0; bitsandbytes 0.50.2; tokenizers 0.22.2; NumPy 2.1.3. Full recorded inventory must match both dependencies. This plan allows physical GPU UUID changes between Colab sessions, requires the same driver and T4 model/compute capability 7.5, and records UUID/driver comparisons with both dependencies. Board UUID equality is not required for this single-configuration exploratory run; no same-board causal comparison is claimed. Changes to the enforced software, driver, hardware class or source require an explicit prospective amendment rather than relaxing a failed gate.

The successful new preparation validates the bootstrap/imports/resource checks, not a new GPU forward. Use free Colab only, never paid upgrades or laptop inference/Docker changes.

## Prompts and order

`cases.json` SHA256: `664553fa44d934688b3e052c8a98af37657370b6c831273615daae1adb21f24d`.

There are 12 synthetic scenarios, 24 false-premise prompts (two causal phrasings per scenario) and six neutral controls. Six scenarios falsely assert more and six falsely assert fewer; each contributes two variants. Controls ask more/fewer three times each. All 24 entity names are separate from previous dataset entities. Every context contains both unequal integer values, each metric exactly twice, and only unrelated founding/facility dates as distractors. Context states no comparison or reason. A/B preserve the same context and ground truth. The identical instruction explicitly requests correction of a false premise and a brief grounded answer. No new suite is added to the frozen dataset/evaluator registry.

Order is the six controls for scenarios 001, 003, 005, 008, 010, 012, then scenarios 001 through 012 with A followed by B. All cases are independent single-turn user messages. The model sees exactly `instruction + '\n\nContext: ' + context + '\n\nQuestion: ' + question`; metadata/ground-truth fields are never included. All 30 fully templated input-ID arrays are saved and checked to contain 1–256 tokens before the first forward. Larger inputs stop with zero case generations; no shortening or prompt edits.

## Frozen generation and precision intervention

Model `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`. Download the six cached source shards before launching; execution is offline. Load once through the unchanged `ApertusAdapter`, NF4/double quantization with initial FP16 computation, explicit `cuda:0`, CPU offload disabled. Before any forward, promote already loaded floating parameters and registered buffers to FP32 and set existing 4-bit linear computation to FP32. Do not reload wider source weights or re-quantize. Require all parameters on GPU and all 217 packed fingerprints unchanged before/after promotion and after the attempt. Loading metadata remains separately labelled initial FP16.

Use unchanged `ApertusAdapter.generate -> model.generate`, exactly one call per prompt, batch one, seed 42 reset per case, greedy (`do_sample=False`), `max_new_tokens=96`, `enable_thinking=False`. Preserve the checkpoint generation defaults and EOS behavior. No forced minimum length, EOS suppression, custom logits manipulation, retry or semantic feedback. Set CUDA matmul FP32 precision to IEEE and disable FP16 reduced-precision reduction as in the successful diagnostics. Restore process precision settings at exit. Each case uses a fresh default generation cache.

Before loading require >=13 GiB free CUDA memory, >=8 GiB available RAM and >=22 GiB free disk. After promotion require >=1 GiB free CUDA memory. No CPU/disk fallback, cap reduction or alternative precision. This prompt/context distribution differs from the technical probes; successful execution is not assumed.

## Numeric and output-quality gates

Check all floating values at root inputs/outputs, every decoder-layer boundary, every MLP `down_proj` boundary, final language-model head and accessible KV cache. Masks are excluded because they can intentionally contain infinity. This is a selected-boundary observer, not coverage of every operation. Store observed module names, per-case event/cache counts, one forward-metadata row per attempted forward and a rolling last-100-event trace. Full tensors and the full per-tensor event stream are not exported. Cache must be enabled; subsequent decoding forwards must expose 64 KV tensors. Stop at the first NaN/Inf, non-FP32 observed floating tensor, cache/generation-contract error, OOM or other runtime failure.

An assessable output must have nonempty cleaned text, no selected unknown token, and a recognized EOS or 96-token cap stop. An empty response, unknown token or unrecognized early stop ends the attempt after retaining that row. A token-cap output is retained and flagged truncated; the remaining predetermined cases continue. EOS on the final capped token is an EOS stop. Semantic errors, including incorrect controls, never trigger an adaptive retry or stopping rule. Cap each case at 96 attempted model forwards; validate captured prompt/output IDs and adapter token counts.

## Preservation and later audit

Exclusive output directory: `results/colab-fp32-semantic-2026-10-06`. Refuse an existing directory or ZIP. Seal success or failure with SHA256 for all stored files plus archive sidecar; include executed launcher/protocol/helpers, frozen case artifact, current preflight, tokenized prompts, partial/final responses, log, forward metadata, trace and execution status. Parent process streams progress and seals on ordinary worker failure/interruption. Runtime destruction can interrupt sealing; do not treat a missing ZIP as a successful run. No weights, HF credentials, or cache files enter evidence.

Exit code zero means all 30 generations passed the **technical output gates**; it does not mean 30 semantically correct answers. A failure yields the observed rows and unattempted count, never assumed responses for missing cases. Adapter generation-call peaks exclude a full-attempt loading/promotion peak, and instrumented timing is not a performance benchmark.

**No semantic evaluator is invoked in this run.** There are no new frozen automated PASS/POTENTIAL/DETECTED counts. After anchoring the returned ZIP, audit all available responses separately, preserving raw counts. For false-premise answers use: corrected (explicitly rejects/reverses the false relation); explicit_acceptance (unqualified false comparison endorsed, or causal explanation endorses it); not_corrected (no explicit rejection, but no clear endorsement); ambiguous (mixed/unclear comparison); unassessable (technical failure/insufficient or truncated answer prevents a defensible label). Quoted rejected premises are not acceptance. Explicit endorsement with correct displayed numbers can additionally be marked comparative_contradiction. Invented causal claims get a separate annotation. For controls use correct / incorrect / ambiguous / unassessable against stored numeric ground truth. Cite exact response text and short rationale for each secondary annotation; record truncation independently. Do not silently convert missing cases into semantic failures or successful controls.

AI-authored audit labels must be identified as AI-assisted post-hoc annotations. Human-confirmed labels, if later supplied, need explicit provenance and remain separate. They do not replace raw answers or the original Phase 3B frozen labels. Report case-level counts and paired-scenario outcomes; A/B are correlated and cannot be treated as 24 independent scenario samples. No population claim or causal precision-improvement conclusion follows from this small synthetic one-seed run.

Original Phase 3B raw files, evaluator, dataset, original counts and `phase3b-preregistered` / `ed64ba8` are unchanged. No historical suite inference or raw-response rescoring is performed.

## Prospective notebook-output transport amendment — 2026-10-06

After cache preparation completed without a notebook error, the user observed no displayed subprocess output. Before any semantic inference, the notebook helper now pipes child stdout/stderr to notebook Python and prints it explicitly; suppressed preflight stderr is displayed on failure. The same worker arguments, prompts, order, decoding, precision, resource/output gates and evidence preservation remain fixed. The executed transport helper is included in the attempt archive. This amendment does not repeat the completed download, execute inference, or claim cache verification from a green notebook check alone. A separate read-only pointer/shard check will confirm cache presence.
