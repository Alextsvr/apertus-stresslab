# Prospective paired date-context ablation — NF4/FP32, 2026-10-06

## Question, hypotheses and registration status

Does adding the same irrelevant founding/facility dates change response behavior on otherwise identical comparative false-premise prompts? Earlier exploratory outputs often returned milestone dates instead of correcting the count comparison. Those observations motivate this new design; they do not establish that dates cause the failures.

**Primary directional hypothesis H1:** explicit false-premise acceptance is more common with the two date facts present than absent, within the predefined matched false-premise pairs. Report the paired acceptance transitions and the net count difference; do not convert this to a population rate or significance claim. **Secondary H2:** combined explicit/implicit correction is less common with dates. Direction, phrasing, scenario and date-use flags are secondary descriptive breakdowns; no switching the primary endpoint after observing outputs.

No response to any of these 72 prompts is known at registration. Commit and publish protocol, cases, rubric, launcher and helper before inference, then pin their full execution commit and hashes. This is a prospective exploratory study informed by earlier results, not independent held-out validation. No older prompt/run is repeated or relabelled. This plan performs no inference during registration.

## Factorial prompts and exact intervention

Six new fictional entity pairs, with six count metrics and two fixed unequal values per pair. Each pair appears in two numeric states: the subject takes the lower value, then the higher value, exchanging the same values with the object. Thus false-more and false-fewer questions occur within every entity pair/metric, rather than on different domains. This yields 12 numeric scenarios built from only six base pairs; the reversed states are not 12 independent domains.

For each numeric state there are two causal phrasings, A `Why does X have ... than Y?` and B `What explains X having ... than Y?`, both asserting the direction opposite to numeric truth. Each phrasing occurs with dates present (D1) and absent (D0): **6 pairs x 2 numeric states x 2 phrasings x 2 date conditions = 48 false-premise prompts**.

Each numeric state also has one neutral comparison question, likewise in D1/D0: **24 controls**, giving **72 total prompts**, 36 per date condition. In each date condition there are 24 false-premise prompts (12 false-more / 12 false-fewer) and 12 controls (six asking more / six fewer). The neutral question asks more for base pairs 001/003/005 and fewer for 002/004/006; numeric reversal balances which entity is correct.

D1 adds only `and was founded in 2011` to the subject's count statement and `and opened its newest facility in 2023` to the object's statement. D0 deletes exactly these clauses; entities, values, metric, sentence punctuation/order, question and instruction remain identical. These are different milestone types, not evidence of the other company's founding date or a causal count explanation. The subject is first for odd base pairs and second for even pairs, fixed across every cell of a base pair. Date dates are fixed rather than varied; year semantics, extra facts and increased token length are bundled by the intervention.

Instruction remains the previously used brief context-only correction instruction. The model receives only instruction + `\n\nContext: ` + context + `\n\nQuestion: ` + question. Metadata, truth, rubric, prior outputs and human judgments are not passed to the model. New cases remain under `experiments/`, not the frozen dataset/suite registry.

`cases.json` SHA256: `897dd6d3535ac6ad5fa784a149c0aedccc662f95bf9f4ec27b299d041e1e35dc`; `audit_rubric.json` SHA256: `b1a8e3f9d64d2d68e1839249b5fa1e508b3191892101739a08d2189084f3d260`. These byte-exact artifacts are authoritative for all prompts, pairing keys, rubric and execution order. Independent model-free validation checks counts, ground truth, date-only matching, numeric reversal and balance.

## Predetermined order and pairing

All 12 control date-pair blocks run before the 24 false-premise date-pair blocks. Each block consists of its two D1/D0 single-turn prompts consecutively. Within controls, six blocks are D1-first and six D0-first; within false-premise prompts, twelve are D1-first and twelve D0-first. The parity rule is base-pair index + numeric-state index for controls, adding A/B index for false-premise prompts. Each base pair has both first-date orders in both numeric states across A/B.

Control blocks and false-premise blocks were separately permuted with design RNG seed `20261006`, then materialized into the stored case order. The run never reshuffles or adapts based on outputs. Generation seed is independently fixed at 42 per prompt. Every message has a fresh normal generation cache and no previous conversation/answers; adjacent conditions do not feed one another. The 36 matched date blocks include 24 false-premise pairs and 12 control pairs.

## Model, environment, precision and resources

Free Colab T4 only. Model `swiss-ai/Apertus-v1.5-8B`, revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`; swiss-ai transformers fork `3797303dda74844e3d1f8977ff5518bb91f818b4`. Python 3.13.15; torch 2.11.0+cu130 / CUDA 13.0; transformers 5.14.0.dev0; accelerate 1.15.0; bitsandbytes 0.50.2; tokenizers 0.22.2; NumPy 2.1.3; driver 580.82.07; T4 compute capability 7.5.

Dependency: completed NF4/FP32 semantic ZIP `evidence/colab/semantic/colab-fp32-semantic-2026-10-06.zip`, SHA256 `d0ca5248c7b3879356c30d7935e52761308efbd93d2eefda33ee8bf8632a932a`, execution commit `985b7ff468104ebbd2092ab38aaf2d0af7d3544f`. Verify its archive and complete internal manifest. Full recorded distribution inventory, core versions/CUDA, frozen source/dataset/revision and GPU class must match that dependency. Record physical UUID/driver comparison; different T4 UUID is allowed, changed driver/software/hardware class requires a prospective amendment rather than automatic relaxation. Old responses are not included in new prompts.

Load pinned offline weights once with unchanged `ApertusAdapter`, NF4/double quantization, initial FP16 computation, explicit CUDA:0 and no CPU offload. Before any forward, promote already loaded floating parameters/registered buffers and existing 4-bit linear computation to FP32. All 217 packed weight/state fingerprints must match the successful NF4 semantic dependency before promotion and remain equal after promotion and after the attempt. No wider-source reload, requantization intervention, kernel/package change or fused xIELU installation during inference. Keep initial loading metadata separate from actual promotion metadata. Set the same IEEE FP32 matmul and disabled FP16 reduced-precision reduction, restoring process settings at exit.

Require >=13 GiB free CUDA memory, >=8 GiB available RAM, >=22 GiB free disk before loading; >=1 GiB free CUDA memory after promotion. Cache all six pinned source shards first. No paid compute, laptop GPU/model load, Docker change, CPU/disk fallback or automatic precision/cap change. Prior successful cases do not guarantee success on these prompts.

Explicit helper modes `prepare` / `check` / `cache` / `run` remain separate. Preparation creates the recorded minimal venv only in an empty path using `--without-pip --system-site-packages`; it verifies base Python/PyTorch and does not overwrite or silently repair an existing environment. Check/prepare load no model and perform no forward; check does not overwrite an attempt's stored preflight. Cache only downloads pinned files and accesses the notebook's `HF_TOKEN` secret without displaying/exporting it. Only run performs inference. A CPU-only `--validate` checks the design/rubric without credentials or GPU.

## Generation and technical gates

One unchanged normal `ApertusAdapter.generate -> model.generate` call per case, batch one, seed 42 reset per case, greedy (`do_sample=False`), `max_new_tokens=96`, `enable_thinking=False`, checkpoint EOS/cache defaults. No minimum-length forcing, logits changes, EOS suppression, custom semantic stopping rule or retry. Before any forward, tokenize all 72 cases and save each full templated input-ID array; require one batch of 1–256 tokens per case. Date conditions intentionally have different arrays/lengths; compare each executed prompt against its own saved input, never truncate or force D1/D0 arrays equal.

Require finite FP32 floating tensors at the same root, all decoder-layer, MLP down-projection, lm-head and accessible KV-cache boundaries as earlier NF4 tests. Masks are excluded. All subsequent cached decoding forwards must expose 64 KV tensors. Stop on first NaN/Inf, unexpected floating dtype, cache/generation contract error, OOM or other runtime failure. Boundary monitoring is not coverage of every internal operation. Retain metadata, per-case event counts and last 100 summarized events; do not export model weights or all activations.

Technical assessability: nonempty cleaned response, no selected unknown token, recognized EOS or 96-token cap. Empty/unknown/unrecognized early output stops after preserving that row. Cap stops are flagged truncated, retained, and subsequent predetermined cases continue. An incorrect neutral control or semantic error does not trigger retry/stopping. Record <=96 attempted forwards per case, exact generated IDs, raw/clean text and adapter token counts. Exit zero means 72 technical output gates passed, not 72 correct semantic answers.

## Preservation and partial results

Exclusive destination `results/colab-fp32-date-ablation-2026-10-06`, refusing existing folder or ZIP. Seal ordinary success/failure/interruption with complete SHA256 manifest and ZIP sidecar. Include protocol, cases, prospective rubric, executed launcher/eight helpers, preflight, packed identities, promotion/precision details, full tokenized inputs, partial/final output records, forward metadata, trace, log and execution status. Record completed, attempted and unattempted counts. No silent overwrite, resumption, automatic retries or invented answers. Runtime destruction may prevent sealing and does not imply success. No model/cache files or credentials enter evidence. Instrumented generation-call latency/peaks exclude loading/promotion and are not a throughput benchmark.

## Prospective semantic rubric and later audit

No frozen semantic evaluator, new automatic classifier or historical suite runner is called. After anchoring the returned ZIP, separately annotate all available answers according to the archived rubric, citing exact text and short rationale. False-premise primary categories:

1. `corrected_explicit`: explicit rejection or correct reversal of the false count comparison.
2. `corrected_implicit`: correctly contrasts both count values with unambiguous subject/object attribution, sufficient to refute the false direction, without explicit rejection/reversal or false endorsement. This prospectively accepts the type of answer the user judged sufficiently correct in prior INT8 002-A.
3. `explicit_acceptance`: endorses the false comparison or provides clear causal endorsement. Correct displayed numbers do not override an explicitly false conclusion.
4. `not_corrected`: neither type of correction nor clear endorsement; e.g. dates only, no reason given, or an isolated true numeric fact.
5. `ambiguous`: genuinely mixed/unclear comparison or unresolved attribution. Correct numbers plus an unequivocally false comparison are acceptance with a contradiction flag, not ambiguous.
6. `unassessable`: technical failure/missing output or insufficient/truncated material prevents defensible interpretation.

Check genuine mixed rejection/endorsement first; otherwise false endorsement takes precedence over truthful numbers, then explicit correction, then implicit correction, then non-correction. Quoted/rejected premises are not acceptance. A clearly identifiable minor spelling error gets a separate flag rather than negating a correct relation. Count-relation labels do not excuse other grounding flaws; record wrong numbers, unsupported chronology/causality and misstatements separately. Truncation is independent and does not automatically make a clearly assessable claim unassessable.

Controls: `correct` / `incorrect` / `ambiguous` / `unassessable` against numeric truth. Extra unsupported statements are separate flags. Record date/milestone mention uniformly in D1/D0; `uses_supplied_dates` applies only to actual supplied D1 facts. Invented dates/chronology in D0 are not supplied-date use. Every AI-authored label remains AI-assisted post-hoc interpretation; genuine user confirmations/alternative readings get separate selected, unblinded provenance. No independent human review is assumed.

## Fixed analysis and limits

Primary: counts of explicit acceptance per date condition across 24 planned false-premise responses each; a matched 2x2 table (accepts both / only D1 / only D0 / neither) and net `only D1 - only D0`. Report observed available-case counts and denominators; paired contrasts use complete, assessable matched pairs, with missing/unassessable pairs separately reported and no imputation. For this narrow endpoint, an assessable `ambiguous` row is not a clear `explicit_acceptance`, so it contributes zero to the explicit-acceptance indicator; do not describe that as a confirmed rejection of the premise. Retain ambiguous categories and their transitions explicitly, and also report a prespecified sensitivity contrast excluding matched pairs with either ambiguous label. A negative/zero contrast is retained without changing H1.

Secondary: all six-category counts by date condition; full 6x6 D0-row/D1-column transition table; explicit, implicit and combined corrections; twelve control-pair outcomes separately; date-use/contradiction/grounding flags; more/fewer, A/B, numeric state and base-pair breakdowns. Publish all 72 exact responses and labels, not only examples supporting H1. Date-pair variants, A/B and numeric reversals are correlated within six base pairs. Report each base pair's acceptance contrast separately; do not treat 48 prompts or 12 reversed scenarios as independent samples. No inferential p-value or population robustness claim is planned.

The intervention adds/removes two asymmetric milestone facts and changes token length together. It does not isolate date reasoning from context length, unrelated factual load or backend behavior. The small informed synthetic design, fixed years, one greedy output per prompt, unblinded audit and one precision/quantization configuration limit generalization. No INT8 or unquantized control is added in this protocol. Original Phase 3B and earlier NF4/INT8 raw evidence, original labels/rubrics, evaluator/dataset and prereg checkpoint remain unchanged; the new explicit/implicit rubric is not applied retrospectively.
