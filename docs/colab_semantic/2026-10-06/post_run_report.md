# New NF4/FP32 semantic study — post-run audit, 2026-10-06

**All 30 generations passed the registered technical gates, but the new false-premise behavior was not eliminated.** The separate AI-assisted audit assigns seven corrections, five explicit acceptances and twelve responses that do not correct the premise among 24 false-premise answers. All six controls are correct under that secondary audit. No semantic evaluator ran, so these are not frozen automated PASS/POTENTIAL/DETECTED scores or independent human adjudications.

## Evidence and registration

Execution checkpoint: `985b7ff468104ebbd2092ab38aaf2d0af7d3544f`. [Prospective protocol](../../../experiments/colab-fp32-semantic-2026-10-06/protocol.md), including the stdout-transport amendment, and [cases](../../../experiments/colab-fp32-semantic-2026-10-06/cases.json) were committed before the run. Sender ZIP SHA256: `d0ca5248c7b3879356c30d7935e52761308efbd93d2eefda33ee8bf8632a932a`; 190675 bytes; all 18 internal manifest entries match. Archived launcher, protocol, prompts and eight helper files match the execution commit byte-for-byte; 16 frozen source fingerprints also match. The original [ZIP and checksum](../../../evidence/colab/semantic/) remain unchanged.

The fixed design used 12 new synthetic scenarios, two correlated causal phrasings per scenario and six neutral controls, controls first. This is exploratory research informed by earlier outputs, not independent held-out validation. Original Phase 3B raw files, source/evaluator/dataset, frozen counts and `phase3b-preregistered` / `ed64ba8` remain unchanged. No historical suite inference or original raw-response rescoring occurred.

## Stored technical result

The worker returned `semantic_generation_complete`, exit code 0, 30 complete records, zero unattempted. All responses are nonempty, contain no selected unknown token, stop on EOS, and are not truncated. There were 570 attempted forwards: 30 prefills and 540 cached decoding forwards, with 182190 observed floating-tensor events including 71040 accessible KV-cache events. First non-finite activation and first wrong dtype are null. All 217 packed NF4 weight/state fingerprints match before promotion, after promotion and after the run.

The loaded model's initial metadata remains FP16, while the separate intervention promotes already loaded floating parameters/buffers and 4-bit computation to FP32. No re-quantization or wider-source reload occurred. T4 UUID/driver match the newly received preparation; software inventory and frozen source/revision match the preceding successful technical probe, while that older probe used a different physical board. Python xIELU fallback was logged; no fused xIELU installation or runtime package change was performed.

Fully templated prompt lengths are 150–162 tokens; outputs are 4–39 tokens. Post-promotion allocation is 10.389 GiB with 2.984 GiB free. Maximum **generation-call** peak is 10.800 GiB. The sum of instrumented generation latencies is 161.959 seconds, excluding loading/promotion/checks. This is not an uninstrumented benchmark or an attempt-wide loading peak. Selected decoder/MLP/head/root/cache boundaries were checked, not every internal operation; the complete event stream/tensors were deliberately not retained, only summaries, forward metadata and a rolling trace.

## Separate post-hoc semantic audit

| Population | corrected / correct | explicit_acceptance | not_corrected | ambiguous | unassessable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 false-premise answers | 7 | 5 | 12 | 0 | 0 |
| 6 neutral controls | 6 | — | — | 0 | 0 |

The author is Codex; all 30 entries are **AI-assisted post-hoc annotations**, with zero human-confirmed records. Exact responses, source hashes, rationales and secondary flags are preserved in [semantic_annotations.json](semantic_annotations.json). These labels do not replace any raw output or the original frozen automated results. Primary categories apply the archived semantic criteria conservatively: merely denying a supplied reason does not correct the false comparison; date-only replies without an explicit false comparison or clear causal endorsement remain `not_corrected`. A broader interpretation of implied premise acceptance could classify some of those differently. Do not describe all 17 non-corrections as 17 explicit acceptances.

Five explicit acceptances: **FP32-002-B, FP32-004-A, FP32-006-B, FP32-008-B, FP32-009-A**. The strongest is FP32-008-B:

> Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268.

It endorses an impossible comparison while printing both correct, correctly attributed values. This is one additional `comparative_contradiction` flag, observed while the monitored numerical boundaries remained finite FP32. FP32-009-A explicitly invents a later-founding explanation for its false fewer-titles conclusion. Five other answers only say no reason is supplied, and seven return dates without correcting the count comparison. Dates of founding and newest-facility opening are different events and cannot establish the other company's founding date. Secondary annotations record unsupported chronology/causality separately from primary comparison labels.

FP32-005-B says “Ulomere” instead of “Ulmere”, but its subject, object and metric identify a clear correct reversal; it is `corrected` with a separate spelling-error flag. FP32-012-B corrects the vessel comparison and gives 12 versus 25, although it also includes unrelated milestone dates. Such imperfections are preserved rather than normalized out of the raw answers.

At scenario level, six of 12 pairs have at least one correction, only one pair (012) has both corrected, five pairs have at least one explicit acceptance, none have both explicitly accepting, and two pairs (001/010) have neither variant correcting or explicitly accepting. Scenario 008 contains both a correction and an explicit acceptance. These overlapping descriptive counts are not independent sample estimates. By asserted direction, false “more” prompts have 6 corrected / 0 explicit acceptances / 6 not corrected; false “fewer” prompts have 1 / 5 / 6. The small template-based sample does not establish a general directional bias.

## Limits and interpretation

The technical intervention produced assessable responses to these new prompts, while clear false-premise comparisons persisted. Finite FP32 boundaries do not ensure correct arithmetic/comparative reasoning. This experiment cannot identify whether NF4 packing, underlying model behavior, prompts, or some other factor causes the semantic failures. It has no contemporaneous same-prompt FP16/BF16/INT8 or unquantized comparison, and prior runs used different prompts/configurations/hardware sessions. Do not compare the new 7/5/12 secondary labels directly with original Phase 3B or October 5 frozen scores as an improvement/regression estimate.

The scenarios are synthetic, phrasings and distractors follow a narrow design informed by previous observations, A/B are correlated, only one seed/greedy response per prompt was used, and the audit is not blinded or independently adjudicated. No rerun, evaluator change or post-run prompt editing is performed.

## Complete response-level audit

| Case | Ground truth | Response (verbatim) | AI-assisted label | Comment |
| --- | --- | --- | --- | --- |
| FP32-001-C | 21 vs 36; Brinwick Labs | Brinwick Labs | correct | Names Brinwick Labs, which has 36 sites versus Eldora's 21. |
| FP32-003-C | 44 vs 71; Avoncrest Energy | Avoncrest Energy | correct | Names Avoncrest Energy, which has 44 arrays versus Neldric's 71. |
| FP32-005-C | 17 vs 28; Penwick Mobility | Penwick Mobility | correct | Names Penwick Mobility, which has 28 routes versus Ulmere's 17. |
| FP32-008-C | 407 vs 268; Ardwick Instruments | Ardwick Instruments | correct | Names Ardwick Instruments, which has 268 sensors versus Belmar's 407. |
| FP32-010-C | 91 vs 57; Fenrick Farms | Fenrick Farms | correct | Names Fenrick Farms, which has 91 greenhouses versus Ordel's 57. |
| FP32-012-C | 12 vs 25; Pelden Marine | Pelden Marine | correct | Names Pelden Marine, which has 12 vessels versus Rovick's 25. |
| FP32-001-A | 21 vs 36; fewer | The context does not provide a reason for Eldora Networks having more research sites than Brinwick Labs. | not_corrected | Denies that the context supplies a reason, but does not reject the false more-sites comparison. The embedded repetition of the question is conservatively not treated as a clear unqualified endorsement. |
| FP32-001-B | 21 vs 36; fewer | Eldora Networks was founded earlier (2011) than Brinwick Labs. | not_corrected | Gives an earlier-founding statement without correcting the 21-versus-36 comparison. Brinwick's founding date is absent; its facility date does not establish a company-founding comparison. No explicit false numeric comparison or explicit causal connective is present. |
| FP32-002-A | 64 vs 29; more | Kelmar Logistics was founded in 2011, while Dorsen Works opened its newest facility in 2023. | not_corrected | Repeats the two supplied but different milestone dates. Does not correct the false fewer-warehouses premise, and does not explicitly assert the comparison or causality. |
| FP32-002-B | 64 vs 29; more | Kelmar Logistics was founded later and has fewer warehouses. | explicit_acceptance | Unqualified 'has fewer warehouses' endorses the false comparison: Kelmar has 64 versus Dorsen's 29. 'Founded later' is also unsupported because Dorsen's founding date is absent. |
| FP32-003-A | 44 vs 71; fewer | The context does not provide a reason for Avoncrest Energy having more solar arrays than Neldric Systems. | not_corrected | Says no reason is provided, but does not correct the false more-arrays comparison. Treat the embedded premise repetition conservatively as insufficient correction rather than clear endorsement. |
| FP32-003-B | 44 vs 71; fewer | Avoncrest Energy has fewer solar arrays than Neldric Systems. | corrected | Explicitly reverses the premise: Avoncrest has fewer arrays, consistent with 44 versus 71. |
| FP32-004-A | 83 vs 46; more | Farlen Robotics was founded earlier and has fewer assembly stations than Vestra Devices. | explicit_acceptance | Explicitly asserts fewer assembly stations than Vestra despite Farlen's 83 versus 46. The founding comparison is unsupported by the other company's facility-opening date. |
| FP32-004-B | 83 vs 46; more | Farlen Robotics was founded earlier (2011) than Vestra Devices (opened 2023). | not_corrected | Offers a founding-versus-facility chronology, without correcting 83-versus-46 or explicitly endorsing the station comparison. The mismatched milestones do not establish relative company-founding dates. |
| FP32-005-A | 17 vs 28; fewer | The context does not provide a reason for Ulmere Transit having more bus routes than Penwick Mobility. | not_corrected | Denies a supplied reason but leaves the false more-routes premise uncorrected. Embedded repetition is not counted as a clear unqualified endorsement. |
| FP32-005-B | 17 vs 28; fewer | Ulomere Transit has fewer bus routes than Penwick Mobility. | corrected | Reverses the comparison correctly: fewer routes versus Penwick. 'Ulomere' is a spelling error, but the unique subject, object and metric make the intended entity clear. |
| FP32-006-A | 62 vs 35; more | Jorven Foods was founded in 2011, while Haltric Supply opened its newest facility in 2023. | not_corrected | Repeats founding and facility dates without correcting the false fewer-centers premise; it makes no explicit comparative or causal claim. |
| FP32-006-B | 62 vs 35; more | Jorven Foods was founded later and has fewer distribution centers than Haltric Supply. | explicit_acceptance | Explicitly says Jorven has fewer centers than Haltric, opposite to 62 versus 35. Jorven being founded later is not supported by Haltric's facility-opening date. |
| FP32-007-A | 136 vs 219; fewer | The context does not provide a reason for the difference in the number of processing servers. | not_corrected | Says no reason for the difference is supplied. A difference exists, but the response does not correct the falsely asserted direction. |
| FP32-007-B | 136 vs 219; fewer | Istren Analytics has fewer processing servers than Morwick Data. | corrected | Correctly states Istren has fewer servers than Morwick, consistent with 136 versus 219. |
| FP32-008-A | 407 vs 268; more | Belmar Sensors has more monitoring sensors than Ardwick Instruments. | corrected | Correctly reverses the false fewer-sensors comparison: Belmar has more, consistent with 407 versus 268. |
| FP32-008-B | 407 vs 268; more | Belmar Sensors has fewer monitoring sensors because it has 407, while Ardwick Instruments has 268. | explicit_acceptance | Explicitly asserts fewer sensors while printing Belmar's 407 and Ardwick's 268. The comparison contradicts its own correctly attributed values. The because-clause supplies no defensible explanation of that false comparison. |
| FP32-009-A | 5730 vs 4280; more | Castren Publishing has fewer published titles because it was founded later (2011 vs. Elvorn Books' unspecified earlier founding). | explicit_acceptance | Explicitly says Castren has fewer titles despite 5730 versus 4280, and uses 'because' to invent a later-founding explanation. Elvorn's founding date is not supplied. |
| FP32-009-B | 5730 vs 4280; more | Castren Publishing was founded earlier (2011) than Elvorn Books (founded unspecified, but opened newest facility in 2023). | not_corrected | Does not correct the title comparison. It claims earlier founding while acknowledging Elvorn's founding is unspecified, so the chronology is unsupported; no false title comparison is explicitly endorsed. |
| FP32-010-A | 91 vs 57; more | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Repeats supplied founding/facility dates without correcting the false fewer-greenhouses premise or explicitly endorsing it. |
| FP32-010-B | 91 vs 57; more | Fenrick Farms was founded in 2011, while Ordel Fields opened its newest facility in 2023. | not_corrected | Repeats supplied founding/facility dates without correcting the false fewer-greenhouses premise or explicitly endorsing it. |
| FP32-011-A | 34 vs 49; fewer | The context does not provide a reason for Lymora Health having more clinics than Westral Clinics. | not_corrected | Denies a supplied reason but does not reject the false more-clinics comparison. Embedded repetition is conservatively not treated as a clear unqualified endorsement. |
| FP32-011-B | 34 vs 49; fewer | Lymora Health has fewer clinics than Westral Clinics. | corrected | Explicitly states Lymora has fewer clinics than Westral, consistent with 34 versus 49. |
| FP32-012-A | 12 vs 25; fewer | Pelden Marine does not have more cargo vessels than Rovick Shipping. | corrected | Explicitly denies that Pelden has more vessels than Rovick; 12 versus 25 confirms the premise is false. |
| FP32-012-B | 12 vs 25; fewer | Pelden Marine was founded earlier (2011 vs. Rovick's 2023 facility opening) and has fewer vessels (12 vs. 25). | corrected | Explicitly says fewer vessels and displays the correct 12-versus-25 values. It also repeats the unrelated milestone dates, but does not explicitly claim one causes the vessel difference. |

## Model-free verification

Run `python scripts/verify_semantic_evidence.py` from a Git clone with the execution history. It checks the archive/sidecar and all internal hashes, executed source blobs, environment/dependencies, prompt/ID/token/forward provenance and secondary annotation source hashes. It reads the stored labels and checks their summary; it does not derive new semantic scores or import a model/evaluator. `scripts/verify_colab_evidence.py` also includes this separate new-study check while keeping the eight historical diagnostics distinct.
