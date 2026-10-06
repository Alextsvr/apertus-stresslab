# INT8/FP16 matched-prompt comparison — post-run audit, 2026-10-06

**All 30 INT8 generations passed the registered technical gates. Semantic behavior is mixed: eight explicit corrections, four explicit acceptances and twelve non-corrections among 24 false-premise answers; all six controls are correct.** The already observed NF4/FP32 baseline retains its original seven / five / twelve AI annotations and six correct controls. INT8 adds three corrections and loses two; it also produces three self-contradictory comparisons with correct displayed numbers, versus one in NF4/FP32. These are separate AI-assisted post-hoc labels, not frozen evaluator scores or independent human adjudications.

## Evidence and exact matching

Execution commit `03cca57567958feba76768381d05f5069aad46fe` includes the [prospective protocol](../../../experiments/colab-int8-matched-prompts-2026-10-06/protocol.md), launcher and explicit notebook helper modes published before this run. The [received ZIP and checksum](../../../evidence/colab/int8_matched/) are preserved without alteration: SHA256 `874d4605e55a23b0f95be061dbf5b0058dd251f5b0c3fb4184f603aea729b5fa`, 92497 bytes, all 18 internal entries verified. Executed protocol, launcher, unchanged shared cases and eight helpers match their registered Git blobs byte-for-byte. All 16 frozen source fingerprints match the execution commit; original evaluator/dataset/raw evidence remain unchanged.

Baseline ZIP SHA256 `d0ca5248c7b3879356c30d7935e52761308efbd93d2eefda33ee8bf8632a932a`, execution `985b7ff468104ebbd2092ab38aaf2d0af7d3544f`. The shared cases SHA256 is `664553fa44d934688b3e052c8a98af37657370b6c831273615daae1adb21f24d`; all 30 templated input-ID arrays are identical, in the same order (six controls followed by 12 A/B pairs). The `FP32-` prefix remains solely a matching key. Both runs use the same pinned Apertus revision, seed 42 reset per prompt, normal unchanged adapter generation, greedy decoding, cap 96 and thinking disabled. Ground truth and earlier responses/labels were never passed to the model.

Full recorded software inventory, core versions, CUDA, frozen source/dataset/revision, driver 580.82.07 and T4 compute capability 7.5 match. Physical GPU UUID changed from `GPU-2d908a88-5692-d044-fee3-b9ba2e4cebf0` to `GPU-4ffedfd4-915d-2221-cfda-0e226de82699`, as the plan allowed; the sessions are not the same board. No package changes or xIELU installation were made during inference; Python xIELU fallback was logged.

## Stored technical outcome

`int8_generation_complete`, worker exit 0, 30 completed / zero unattempted. Every response is nonempty, has no selected unknown token, ends on EOS and is untruncated. There are 610 forward-metadata rows: 30 prefills and 580 cached decoding forwards. All subsequent forwards expose 64 KV tensors. The stored monitor records 195110 finite FP16 tensor events, including 76160 KV-cache events; first non-finite and first wrong dtype are null. This selected-boundary coverage does not observe every internal operation, and FP32 parameter elements do exist despite all observed activation events being FP16.

The model has 217 `Linear8bitLt` modules with 7023633920 stored INT8 parameter elements, 1629761600 FP16 elements and 245953634 FP32 elements. Stored INT8 weight byte fingerprints match before/after the run. No NF4 load, FP32 promotion or re-quantization intervention occurred. After load, allocation is 10.515 GiB with 3.922 GiB free. Inputs span 150–162 tokens; outputs span 4–36. Maximum generation-call peak is 11.109 GiB and summed instrumented generation latency is 189.257 seconds. These exclude loading/checks and are not an uninstrumented performance benchmark or attempt-wide peak.

## Separate AI-assisted semantic audit

| Condition / population | corrected / correct | explicit_acceptance | not_corrected | ambiguous | unassessable |
| --- | ---: | ---: | ---: | ---: | ---: |
| NF4/FP32, 24 false-premise | 7 | 5 | 12 | 0 | 0 |
| INT8/FP16, 24 false-premise | 8 | 4 | 12 | 0 | 0 |
| NF4/FP32, 6 controls | 6 | — | — | 0 | 0 |
| INT8/FP16, 6 controls | 6 | — | — | 0 | 0 |

All new labels are authored by Codex after reading the outputs, with **zero human-confirmed records** and no blinded/independent review. See [semantic_annotations.json](semantic_annotations.json) for all exact responses, source hashes, rationales and flags. Baseline annotations remain byte-for-byte unchanged. No semantic evaluator, historical runner, rerun or original Phase 3B rescore occurred. The technical script's exit code is distinct from semantic correctness; there are no new frozen automated PASS/POTENTIAL/DETECTED counts.

The same conservative primary criterion requires an explicit rejection/reversal for `corrected` and clear false-comparison/causal endorsement for `explicit_acceptance`. Date-only and no-reason replies remain `not_corrected`. INT8 FP32-002-A accurately repeats 64 and 29 warehouses but does not explicitly reject/reverse fewer; it is `not_corrected` under the registered criterion, with `correct_values_without_explicit_correction`. An interpretation accepting implicit correction from values would differ, so this borderline distinction is explicit rather than hidden. FP32-005-B has the identifiable typo “Ulumre” but a clear correct reversal; FP32-012-A corrects the count while retaining an unsupported founding comparison. FP32-011-B incorrectly says the opening year is absent despite the supplied 2023 facility date. Secondary flaws do not silently replace the primary label.

Four explicit INT8 acceptances: **004-A, 008-A, 008-B, 009-A**. The two 008 variants produce the same response:

> Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268.

The 009-A response is:

> No false premise to correct. Castren Publishing has fewer published titles because it has 5730 compared to Elvorn Books' 4280.

Both print the correct values and assert the impossible direction. Thus INT8 has three `comparative_contradiction` rows (two correlated sensor variants and one titles row), across two scenarios. NF4/FP32 had one (008-B). A finite monitored FP16 execution can still yield explicit semantic contradictions; these rows are not evidence of NaN/Inf failure.

## Paired transitions across all 24 false-premise responses

Rows are stored NF4/FP32 labels; columns are new INT8/FP16 labels. Controls are separate. Zero ambiguous/unassessable rows are retained in the registered 5x5 table.

| NF4 → INT8 | corrected | explicit_acceptance | not_corrected | ambiguous | unassessable |
| --- | ---: | ---: | ---: | ---: | ---: |
| corrected | 5 | 1 | 1 | 0 | 0 |
| explicit_acceptance | 0 | 3 | 2 | 0 | 0 |
| not_corrected | 3 | 0 | 9 | 0 | 0 |
| ambiguous | 0 | 0 | 0 | 0 | 0 |
| unassessable | 0 | 0 | 0 | 0 | 0 |

Five pairs are corrected in both conditions; three only in INT8 (001-A/003-A/007-A); two only in NF4 (008-A/011-B); fourteen in neither. Missing/unassessable pairs: zero. INT8 adds 008-A acceptance where NF4 corrected the premise; 011-B loses a correction for a date-only reply. Two baseline acceptances become date-only non-corrections, not successful premise corrections:

| Baseline explicit acceptance | INT8 label |
| --- | --- |
| FP32-002-B | not_corrected |
| FP32-004-A | explicit_acceptance |
| FP32-006-B | not_corrected |
| FP32-008-B | explicit_acceptance |
| FP32-009-A | explicit_acceptance |

At scenario level, NF4 has at least one correction in 6/12 pairs, both corrected in 1, at least one acceptance in 5, both accepting in 0, and neither correction nor acceptance in 2. INT8 has at least one correction in 5/12, both corrected in 3, at least one acceptance in 3, both accepting in 1, and neither correction nor acceptance in 4. These overlapping counts describe correlated pairs, not independent samples. Exact cleaned responses match in 10/30 pairs; semantic correctness is not based on exact-string matching.

| Asserted false direction | NF4 corrected / acceptance / not_corrected | INT8 corrected / acceptance / not_corrected |
| --- | --- | --- |
| more (12 answers) | 6 / 0 / 6 | 8 / 0 / 4 |
| fewer (12 answers) | 1 / 5 / 6 | 0 / 4 / 8 |

The narrow synthetic templates and distractors confound any general directional claim.

## Complete 30-row comparison (verbatim cleaned responses)

| ID | Subject vs object values | NF4/FP32 response | Stored AI label | INT8/FP16 response | New AI label | INT8 rationale |
| --- | --- | --- | --- | --- | --- | --- |
| FP32-001-C | 21 vs 36 | Brinwick Labs | correct | Brinwick Labs | correct | Selects Brinwick Labs, the correct entity for the requested more comparison (21 versus 36). |
| FP32-003-C | 44 vs 71 | Avoncrest Energy | correct | Avoncrest Energy | correct | Selects Avoncrest Energy, the correct entity for the requested fewer comparison (44 versus 71). |
| FP32-005-C | 17 vs 28 | Penwick Mobility | correct | Penwick Mobility | correct | Selects Penwick Mobility, the correct entity for the requested more comparison (17 versus 28). |
| FP32-008-C | 407 vs 268 | Ardwick Instruments | correct | Ardwick Instruments | correct | Selects Ardwick Instruments, the correct entity for the requested fewer comparison (407 versus 268). |
| FP32-010-C | 91 vs 57 | Fenrick Farms | correct | Fenrick Farms has more greenhouses. | correct | Selects Fenrick Farms, the correct entity for the requested more comparison (91 versus 57). |
| FP32-012-C | 12 vs 25 | Pelden Marine | correct | Pelden Marine | correct | Selects Pelden Marine, the correct entity for the requested fewer comparison (12 versus 25). |
| FP32-001-A | 21 vs 36 | The context does not provide a reason for Eldora Networks having more research sites than Brinwick Labs. | not_corrected | Eldora Networks has fewer research sites than Brinwick Labs. | corrected | Explicitly states fewer research sites, reversing the false more-sites premise; 21 < 36. |
| FP32-001-B | 21 vs 36 | Eldora Networks was founded earlier (2011) than Brinwick Labs. | not_corrected | Eldora Networks was founded in 2011, while Brinwick Labs opened its newest facility in 2023. | not_corrected | Repeats the supplied founding and facility dates without rejecting the false count comparison or explicitly endorsing causality. |
| FP32-002-A | 64 vs 29 | Kelmar Logistics was founded in 2011, while Dorsen Works opened its newest facility in 2023. | not_corrected | Kelmar Logistics has 64 warehouses, while Dorsen Works has 29 warehouses. | not_corrected | Accurately gives 64 and 29 warehouses, but does not explicitly reject or reverse the false fewer relation. The registered corrected category requires explicit rejection/reversal; no false comparison is endorsed. |
| FP32-002-B | 64 vs 29 | Kelmar Logistics was founded later and has fewer warehouses. | explicit_acceptance | Kelmar Logistics was founded in 2011. | not_corrected | States the supplied founding year only. It neither corrects the false fewer-warehouses premise nor explicitly endorses the comparison or causality. |
| FP32-003-A | 44 vs 71 | The context does not provide a reason for Avoncrest Energy having more solar arrays than Neldric Systems. | not_corrected | Avoncrest Energy has 44 solar arrays while Neldric Systems has 71, so the premise is false. | corrected | Gives the correct 44-versus-71 values and explicitly says the premise is false. |
| FP32-003-B | 44 vs 71 | Avoncrest Energy has fewer solar arrays than Neldric Systems. | corrected | Avoncrest Energy has 44 solar arrays while Neldric Systems has 71 solar arrays, so the premise is false. | corrected | Gives the correct 44-versus-71 values and explicitly rejects the premise. |
| FP32-004-A | 83 vs 46 | Farlen Robotics was founded earlier and has fewer assembly stations than Vestra Devices. | explicit_acceptance | Farlen Robotics was founded later and has fewer assembly stations. | explicit_acceptance | Unqualified fewer assembly stations endorses the false comparison despite 83 > 46. Founded later is unsupported: the other company's founding date is absent. |
| FP32-004-B | 83 vs 46 | Farlen Robotics was founded earlier (2011) than Vestra Devices (opened 2023). | not_corrected | Farlen Robotics was founded earlier. | not_corrected | Claims earlier founding without correcting the station comparison. The other company's facility date does not establish its founding date; no false count comparison or causal connective is explicit. |
| FP32-005-A | 17 vs 28 | The context does not provide a reason for Ulmere Transit having more bus routes than Penwick Mobility. | not_corrected | The context does not provide a reason; it only states the number of routes for each. | not_corrected | Denies that a reason is supplied but does not correct the falsely asserted more-routes relation. It does not clearly endorse that relation. |
| FP32-005-B | 17 vs 28 | Ulomere Transit has fewer bus routes than Penwick Mobility. | corrected | Ulumre Transit has fewer bus routes than Penwick Mobility. | corrected | Explicitly reverses the comparison to fewer bus routes, consistent with 17 < 28. Ulumre is a spelling error; the uniquely identifiable subject/object/metric make the intended correction clear. |
| FP32-006-A | 62 vs 35 | Jorven Foods was founded in 2011, while Haltric Supply opened its newest facility in 2023. | not_corrected | Jorven Foods was founded in 2011, while Haltric Supply opened its newest facility in 2023. | not_corrected | Repeats the two supplied milestone dates without correcting fewer distribution centers or explicitly asserting the comparison or causality. |
| FP32-006-B | 62 vs 35 | Jorven Foods was founded later and has fewer distribution centers than Haltric Supply. | explicit_acceptance | Jorven Foods was founded in 2011, while Haltric Supply opened its newest facility in 2023. | not_corrected | Repeats the two supplied milestone dates. The false fewer-centers relation is not corrected and no clear comparative/causal endorsement appears. |
| FP32-007-A | 136 vs 219 | The context does not provide a reason for the difference in the number of processing servers. | not_corrected | Istren Analytics has 136 processing servers while Morwick Data has 219, so the premise is false. | corrected | Gives 136 versus 219 and explicitly says the premise is false. |
| FP32-007-B | 136 vs 219 | Istren Analytics has fewer processing servers than Morwick Data. | corrected | Istren Analytics has fewer processing servers than Morwick Data. | corrected | Correctly states fewer processing servers, consistent with 136 < 219. |
| FP32-008-A | 407 vs 268 | Belmar Sensors has more monitoring sensors than Ardwick Instruments. | corrected | Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268. | explicit_acceptance | Explicitly says fewer sensors while correctly assigning 407 to Belmar and 268 to Ardwick. The asserted direction contradicts its own displayed values; the because-clause supplies no defensible explanation. |
| FP32-008-B | 407 vs 268 | Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268. | explicit_acceptance | Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268. | explicit_acceptance | Explicitly says fewer sensors with correctly assigned 407 versus 268; the response is identical to its NF4/FP32 counterpart and contradicts its own values. |
| FP32-009-A | 5730 vs 4280 | Castren Publishing has fewer published titles because it was founded later (2011 vs. Elvorn Books' unspecified earlier founding). | explicit_acceptance | No false premise to correct. Castren Publishing has fewer published titles because it has 5730 compared to Elvorn Books' 4280. | explicit_acceptance | Says no false premise exists, then explicitly asserts fewer titles while correctly giving 5730 versus 4280. Its own numbers contradict the comparison and the because-clause is invalid. |
| FP32-009-B | 5730 vs 4280 | Castren Publishing was founded earlier (2011) than Elvorn Books (founded unspecified, but opened newest facility in 2023). | not_corrected | Castren Publishing was founded in 2011, while Elvorn Books opened its newest facility in 2023. | not_corrected | Repeats founding and newest-facility dates without rejecting the false title comparison or explicitly asserting comparative causality. |
| FP32-010-A | 91 vs 57 | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Repeats the supplied milestone dates and leaves the false fewer-greenhouses premise uncorrected; no explicit false comparison/causality. |
| FP32-010-B | 91 vs 57 | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Repeats the supplied milestone dates and leaves the false fewer-greenhouses premise uncorrected; no explicit false comparison/causality. |
| FP32-011-A | 34 vs 49 | The context does not provide a reason for Lymora Health having more clinics than Westral Clinics. | not_corrected | Lymora Health was founded earlier (2011) than Westral Clinics. | not_corrected | Claims earlier founding without correcting the false more-clinics relation. Westral's founding date is absent, so the chronology is unsupported; no explicit false clinic comparison. |
| FP32-011-B | 34 vs 49 | Lymora Health has fewer clinics than Westral Clinics. | corrected | Lymora Health was founded earlier (2011 vs. Westral Clinics' opening year not given). | not_corrected | Does not correct the clinic comparison and claims earlier founding. It also says Westral's opening year is not given even though the context gives its newest facility opening in 2023. |
| FP32-012-A | 12 vs 25 | Pelden Marine does not have more cargo vessels than Rovick Shipping. | corrected | Pelden Marine was founded earlier (2011 vs. 2023) and has fewer cargo vessels (12 vs. 25). | corrected | Explicitly states fewer vessels and gives the correct 12-versus-25 values. Also infers earlier founding from incomparable founding/facility dates; preserve this separate flaw. |
| FP32-012-B | 12 vs 25 | Pelden Marine was founded earlier (2011 vs. Rovick's 2023 facility opening) and has fewer vessels (12 vs. 25). | corrected | False premise: Pelden Marine does not have more cargo vessels than Rovick Shipping. | corrected | Explicitly calls the premise false and denies that Pelden has more vessels, consistent with 12 < 25. |

Machine-readable [paired_comparison.json](paired_comparison.json) preserves both exact responses, unchanged baseline labels, new labels, secondary flags and the complete descriptive summaries.

## Limits and verification

This prospectively registered additional condition uses an already observed, AI-annotated baseline. It is exploratory, not independent held-out validation. Quantization, nominal computation precision and backend kernels change together; physical T4 sessions differ. A single greedy generation per prompt, 12 synthetic scenarios, correlated A/B phrasings and an unblinded AI audit cannot establish a causal INT8 benefit, isolate quantization/precision, estimate population robustness or support a significance claim. Lower explicit-acceptance count is not elimination: two former acceptances became non-corrections, one corrected row became an acceptance, and numeric self-contradictions increased.

Original Phase 3B remains 16 PASS / 6 POTENTIAL / 2 DETECTED on 24 false-premise answers and six PASS controls (22 / 6 / 2 overall). Its raw hashes, evaluator, dataset and `phase3b-preregistered` / `ed64ba8` remain unchanged. The NF4/FP32 baseline was neither rerun nor silently relabelled.

`python scripts/verify_int8_matched_evidence.py` checks hashes, registered sources, exact input matching, environment and token/forward provenance, and stored annotation/paired-summary consistency without importing a model or deriving semantic labels. `scripts/verify_colab_evidence.py` includes this condition separately from the eight historical diagnostics and the one NF4/FP32 semantic study. [integrity.json](integrity.json) records the received evidence checks.

## Selected human confirmation — post-run supplement, 2026-10-06

After publication of the AI audit, the project owner explicitly confirmed `explicit_acceptance` and `comparative_contradiction` for INT8 FP32-008-A, FP32-008-B and FP32-009-A after seeing their exact cleaned responses, numeric ground truth and proposed labels. This is selected, unblinded, AI-assisted human confirmation: three rows across two scenarios, with 008-A/B sharing the same response and ground truth. It is not independent/blinded adjudication or confirmation of the full 30-row audit. Ancillary flags and full contexts were not reviewed in this batch. The remaining 27 INT8 rows and all NF4 baseline labels have no new human confirmation from this batch.

[human_confirmations_01.json](human_confirmations_01.json) records the source-response/AI-annotation/archive hashes and the precise confirmed scope; the raw chat reply is not reproduced. The original AI annotation artifact and paired summary remain unchanged, with their original zero human-confirmed records. This separate supplement adds three selected confirmations without changing labels or the 8 / 4 / 12 comparison counts. No inference, rerun or rescore occurred.
