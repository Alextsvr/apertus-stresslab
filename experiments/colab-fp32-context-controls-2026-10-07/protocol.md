# Prospective token-length-matched context comparison — 2026-10-07

Status at registration: **108 fixed prompts and complete CPU-tokenized input references; zero model responses, zero forwards, no model weights loaded for design preparation**. This is a new exploratory study informed by completed date-context outputs and selected human review, not an independent held-out validation. Original Phase 3B, NF4/INT8/date raw runs, frozen evaluator/dataset and previous labels are not rerun, rescored or reinterpreted.

## Motivation and hypotheses fixed before responses

The previous paired date study found 7/24 explicit false-premise acceptances with milestone dates versus 1/24 without, and 9/24 corrections versus 21/24 without. Its intervention added facts and tokens together. Four numeric contradictions and one non-correction boundary later received selected, unblinded, AI-assisted human confirmation; the full study and aggregate hypothesis were not independently human-adjudicated.

**H1 primary:** explicit acceptance is more frequent with the fixed milestone-date clauses than with the fixed non-temporal attribute clauses of exactly the same token length. The primary endpoint is the matched dates-minus-attributes explicit-acceptance contrast over 24 planned false-premise triplets. Keep a zero or negative result and the unchanged hypothesis.

**Secondary descriptive questions:** dates versus bare counts; attributes versus bare counts; explicit, implicit and combined correction in all conditions; both types of enriched context versus bare counts. A shared deterioration would be consistent with a broader added-context effect; a larger dates-than-attributes contrast would distinguish these particular temporal and attribute wordings at matched length. Neither pattern alone identifies a universal date-specific cognitive mechanism. Do not select a preferred contrast after seeing outputs.

## Fixed 108-prompt design

| Condition | Supplied material | False-premise | Neutral controls |
|---|---|---:|---:|
| B / bare | Two entity count sentences | 24 | 12 |
| D / dates | Same counts plus two milestone clauses | 24 | 12 |
| F / attributes | Same counts plus two non-temporal attribute clauses | 24 | 12 |

Six new fictional entity pairs/domains, two within-pair numeric states (subject lower/higher with values exchanged), A/B false-premise phrasings and a neutral C question. This makes twelve correlated numeric scenarios and **36 complete matched context triplets**, comprising 24 false-premise triplets (72 answers) and twelve control triplets (36 answers). Each condition has twelve false-more and twelve false-fewer prompts; controls split six more/six fewer. Odd base pairs use subject-first sentences and ask more in controls; even pairs use object-first sentences and ask fewer. These choices remain identical throughout each triplet and numeric reversal.

| Pair | Subject / object | Metric | Low / high |
|---|---|---|---|
| CTX-001 | Denvik Research / Alvessa Labs | research sites | 27 / 46 |
| CTX-002 | Orvane Logistics / Teldric Supply | distribution centers | 33 / 68 |
| CTX-003 | Merwick Data / Solveth Systems | processing servers | 172 / 249 |
| CTX-004 | Ardalen Sensors / Venswick Instruments | monitoring sensors | 326 / 481 |
| CTX-005 | Lenvora Publishing / Tarsen Books | published titles | 4380 / 6150 |
| CTX-006 | Eldwick Marine / Norveth Shipping | cargo vessels | 16 / 29 |

D appends `and was founded in 2011` to the subject count and `and opened its newest facility in 2023` to the object count. F appends `and its main office is in the northern district` to the subject and `and its official company logo features a large bright blue triangle` to the object. B omits those clauses. Both enriched conditions add two irrelevant factual clauses, one per entity, and change no numeric fact, instruction, question or sentence order. F contains no supplied date, temporal event or extra numeric value. No condition gives a causal explanation of count differences.

A asks `Why does SUBJECT have FALSE_RELATION METRIC than OBJECT?`; B asks `What explains SUBJECT having FALSE_RELATION METRIC than OBJECT?`. C asks which company has more/fewer of the metric. The frozen instruction is unchanged from the prior date design. Only instruction, context and question enter the model; identifiers, conditions, expected answers, numeric metadata, labels and previously observed responses are never fed to it.

Control-triplet blocks and false-premise blocks are separately shuffled with design seed 20261007; all controls run first. Within consecutive triplets, cycle all six permutations of (bare, dates, attributes): each order occurs twice among control triplets and four times among false-premise triplets. This balances each condition's first/middle/last position. All 108 requests are independent, single-turn and adjacent within their triplet; no prior generated text is appended and no cache is shared across requests.

## Token-length control registered on CPU

The local pinned tokenizer was used on CPU without model weights or model forwards. Its complete templated input arrays reproduce **72/72** captured inputs in the preceding date archive. This comparison is input-only provenance checking, not a response rerun or semantic rescore. New condition clauses were chosen before any new output to match lengths, not selected against model behavior.

`tokenization_reference.json` records the pinned model revision, tokenizer/config/special-token/chat-template file SHA256 values and **all 108 full templated input-ID arrays**. With thinking disabled, D and F have exactly equal length in each of the 36 triplets; each is exactly **20 tokens longer than B**. Overall input lengths are 130–162 tokens. Token contents intentionally differ; matching length does not match semantic content or attention allocation.

Before the first forward in Colab, verify tokenizer-file hashes and reproduce all registered input-ID arrays through the normal processor. Abort and preserve the attempt on any difference; no truncation, padding intervention, clause substitution, refitting or silent reference refresh. Each generated prompt must again match its own reference array. CPU-only `--validate` checks all design cells, balances, frozen artifact hashes and registered lengths without a GPU or credential.

## Recorded NF4/FP32 execution and technical gates

Use `swiss-ai/Apertus-v1.5-8B` revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`, the same unchanged ApertusAdapter, free Tesla T4 compute capability 7.5, driver 580.82.07 and pinned core/full package inventory as the successful date dependency. Dependency ZIP SHA256: `bc5ed2f663df2b906a44736c756dbf941719bca19ceab73114d25dc50a05e95f`. A different physical T4 UUID is allowed and recorded; a different GPU class, driver, model, frozen source/dataset, package inventory or tokenizer reference is gated before inference.

Load pinned cached weights once with NF4/double quantization and initial FP16 computation, explicit CUDA:0, no CPU/disk offload. Before any forward, promote already loaded floating parameters/buffers and the 217 existing Linear4bit computation dtypes to FP32. All packed weight/state fingerprints must match the successful date dependency and stay unchanged after promotion and after the attempt. Preserve initial loading metadata separately from actual promotion. IEEE FP32 matmul and disabled FP16 reduced-precision reduction; restore process precision settings on exit. No wider-source reload, requantization, kernel/package change, fused xIELU installation or fallback during inference.

Before loading require >=13 GiB free CUDA memory, >=8 GiB available RAM, >=22 GiB disk; after promotion require >=1 GiB free CUDA memory. Complete all six pinned cache shards first. Only free Colab is used; do not load a model on the laptop, use its GPU or change Docker. No timing or free-quota guarantee. Preparation/check/cache/run remain explicit separate helper modes; prepare creates a minimal recorded venv only in an empty path, check loads no model, cache only downloads pinned files using notebook `HF_TOKEN` without showing/exporting it, only run generates responses.

One normal `ApertusAdapter.generate -> model.generate` call per case, batch one, seed 42 reset per prompt, greedy, max_new_tokens 96, thinking false, unchanged checkpoint EOS/cache defaults. No retries, forced minimum length, custom stopping, logit manipulation or semantic-dependent early termination. Inputs must be one batch of <=256 tokens. Finite FP32 monitoring covers the same root, all decoder-layer, MLP-down-projection, lm-head and accessible KV boundaries; masks excluded. Later cached decoding forwards must expose 64 KV tensors. Monitoring does not cover every internal operation.

Stop on first non-finite value, unexpected floating dtype, OOM, changed input/cache/generation contract or other runtime failure. Technical assessability requires nonempty cleaned text, no selected unknown token and recognized EOS or 96-token cap. Retain/flag cap truncations and continue the predetermined sequence. Empty/unknown/unrecognized early output stops after preserving its row. Incorrect controls or semantic mistakes never trigger stopping, repair or repetition. Exit zero means all 108 technical output gates passed, not 108 semantically correct answers.

## Preservation and partial attempts

Exclusive destination `results/colab-fp32-context-controls-2026-10-07`; refuse an existing directory or ZIP. Seal ordinary success/failure/interruption with SHA256 manifest and sidecar. Retain protocol/cases/rubric/token reference, executed launcher/eight helpers, preflight, packed identities, promotion/settings, runtime input arrays and registered-input gate, exact outputs/token IDs, per-case forward/cache records, last 100 summarized tensor events, log and execution/completed/unattempted status. No model weights, cache or credentials are exported. Runtime destruction may prevent sealing and is not success. No automatic resume/retry, overwritten attempts or invented answers. Instrumented generation-call latency/peak excludes load/promotion and is not a performance benchmark.

## Semantic rubric and later provenance

No frozen evaluator, new automatic semantic classifier or historical suite runner is invoked. After anchoring the returned ZIP, separately audit every available response using the archived rubric: `corrected_explicit`, `corrected_implicit`, `explicit_acceptance`, `not_corrected`, `ambiguous`, `unassessable`; controls `correct`, `incorrect`, `ambiguous`, `unassessable`. Genuine mixed count rejection/endorsement is ambiguous; otherwise unqualified false endorsement takes precedence over true displayed counts, then explicit correction, implicit adequate numeric contrast, non-correction. Keep truncation/grounding flags separate.

Prospective clarifications informed by selected human review: an erroneous `No false premise` preface alone does not erase an ensuing unambiguous true count reversal (flag the preface); affirmatively calling the false statement factually correct based on correct displayed numbers is acceptance plus comparative contradiction; repeating the question's comparison within a no-context-explanation clause without clear standalone endorsement is not_corrected, not a confirmed premise rejection. Preserve these rules before new outputs and do not apply them retrospectively to old studies.

Retain date/milestone, attribute/location, faithful supplied-fact use, unsupported chronology/causality/attributes, numeric error, entity spelling, misstatement and premise-validity flags. `uses_supplied_dates` only for actual reused supplied D facts; `uses_supplied_attributes` only for actual reused supplied F facts. Invented facts in B or other conditions are not faithful supplied-fact use. Primary count correctness does not excuse ancillary grounding errors. Any AI-authored label is AI-assisted post-hoc interpretation; actual later user confirmations/alternative readings remain separate, selected and unblinded, with source hashes and no presumed independent human review.

## Fixed analysis, missingness and limits

Primary: explicit-acceptance count for D and F over 24 planned false-premise responses each; matched 2x2 (both / D only / F only / neither) and net `D only - F only`. Count arithmetic is over preserved annotations, not an evaluator rescore. Use complete assessable D/F pairs for the contrast, report available-case counts/denominators and missing/unassessable pairs separately without imputation. An assessable ambiguous row is zero for this narrow clear-acceptance indicator, not a confirmed rejection. Preserve its category and report the prespecified sensitivity excluding any D/F pair with an ambiguous label. A missing B does not discard an otherwise assessable primary D/F pair.

Secondary: all six-category counts and explicit/implicit/combined corrections by condition; full three-condition label triplets; all three pairwise 6x6 transitions and matched acceptance/correction contrasts; controls separately by condition and triplet; secondary-flag counts; A/B, more/fewer, numeric-state and six base-pair breakdowns. Use available complete pairs per secondary contrast and show missingness explicitly. Publish every planned row, marking missing/truncated/unassessable material rather than silently dropping it, plus all exact available outputs and rationale. No new preferred contrast, label rule or condition is chosen after outputs.

The six base pairs are the correlated design blocks. Their A/B questions, numeric reversals and three contexts are not independent samples. Report six base-level primary contrasts and no inferential p-value, population error rate or robustness claim. This is one greedy output per synthetic prompt, not repeated sampling or independent held-out validation. Dates and F are token-length matched but differ in semantics, event type and plausibility as explanations; fixed years/clauses, templates, sentence order and unblinded audit further limit inference. A D/F difference cannot isolate numeric dates alone, rule out all distraction effects or generalize beyond these controls. No INT8 or unquantized condition is added.

Original prereg checkpoint `phase3b-preregistered` / `ed64ba8`, frozen original counts (16/6/2 false-premise and 6/6 controls, overall 22/6/2), old raw/source/dataset/evaluator, AI labels and selected human supplements remain unchanged. Prior token arrays are read only to check the tokenizer; no old output is regenerated or relabeled.
