# Apertus StressLab — current research synthesis, 2026-10-06

**This evidence package separates a demonstrated FP16 representation overflow from persistent semantic false comparisons in technically successful generations.** Promoting the loaded NF4 floating computation to FP32 produced usable outputs without changing packed weights. A prospectively registered INT8/FP16 condition on the exact same exploratory prompts also completed successfully, while both configurations retained false comparisons. Neither result establishes general stability or a causal semantic advantage of one precision/quantization setting.

This is a dated synthesis of completed work. Earlier preregistration/draft sections in the technical report remain historical records; their pending statements do not describe the current execution status. All results below have preserved raw evidence and separate provenance.

## Original pre-registered finding

Phase 3B checkpoint `phase3b-preregistered` / `ed64ba8cb382073d652b24e647a898c94f8ce186` fixed 12 held-out synthetic scenarios, 24 A/B false-premise prompts and six controls before the run `2026-10-02_111432`. Its NF4/BF16 + CPU-offload configuration returned 30 records. The frozen evaluator stored **16 PASS / 6 POTENTIAL / 2 DETECTED** on the false-premise prompts and **6 PASS controls**, or **22 / 6 / 2 overall**. Later semantic interpretation is separate from these primary counts; PASS is not independent proof of semantic correctness.

The raw SHA256 checks remain 4/4 matching. Source/evaluator/dataset and original counts are unchanged. The present work does not rerun or rescore that historical suite. See [technical report](technical_report.md) and [methodology](methodology.md) for the original post-run audit, its selected human confirmations and limitations.

## Numerical diagnosis and intervention

The October 5 Colab NF4/FP16 condition produced unknown-token outputs rather than assessable text. Its 30 stored POTENTIAL labels must not be interpreted as 30 semantic acceptances. A later boundary diagnostic found one positive infinity in the third decoder block's MLP down-projection. The isolated replay at coordinate `[0, 0, 2012]` returned:

| Computation using the captured represented operands | Observed coordinate |
| --- | ---: |
| Original bitsandbytes FP16 | +Inf |
| Ordinary FP16 reference | +Inf |
| FP32 reference | 117986.203125 |
| FP32 reference cast to FP16 | +Inf |
| bitsandbytes FP32 | 117986.359375 |

This identifies representation overflow for that coordinate without requiring a bitsandbytes-specific defect. It does not establish the cause of every earlier unknown-token response or general kernel correctness.

Promotion of already loaded floating parameters/registered buffers and 4-bit computation to FP32 preserved all 217 packed NF4 weight/state fingerprints. Subsequent normal-generation checks included three exact short answers and one actual 96-token cached continuation. The latter generated integers 1–35 up to the registered cap, with 111584 finite FP32 observations at selected boundaries. These are technical checks for their prompts and observed boundaries, not a semantic benchmark or proof that every internal operation is finite. Loading/promotion peaks and uninstrumented throughput were not measured by the generation-call metrics. See [numerical diagnostic chain](colab_diagnostics/README.md).

## Same-prompt exploratory semantic comparison

The new NF4/FP32 prompts were registered before their execution but designed with knowledge of earlier results. The subsequent INT8 condition was registered against that already observed baseline. These are exploratory experiments, not another independent held-out validation.

Both used the same 12 scenarios / 24 false-premise prompts / six controls, pinned model revision, instruction, context/question layout, order, seed 42, greedy normal adapter generation, 96-token cap and disabled thinking. All 30 templated input-ID arrays match exactly. Recorded software inventory, driver and T4 class match; physical GPU UUIDs differ.

| Result | NF4/FP32 | INT8/FP16 |
| --- | ---: | ---: |
| Completed technical outputs, all EOS and untruncated | 30/30 | 30/30 |
| Attempted forwards | 570 | 610 |
| Observed finite floating-tensor events | 182190 FP32 | 195110 FP16 |
| Primary AI-assisted explicit corrections, out of 24 | 7 | 8 |
| Primary AI-assisted explicit acceptances, out of 24 | 5 | 4 |
| Primary AI-assisted non-corrections without clear endorsement, out of 24 | 12 | 12 |
| Correct controls under the separate AI audit | 6/6 | 6/6 |
| Comparisons contradicting their own correctly displayed numbers | 1 row | 3 rows |

No semantic evaluator ran in these two new conditions. The semantic categories are AI-assisted post-hoc annotations, separate from the technical result and original frozen evaluator counts. The numeric observer covers selected boundaries/cache, not every internal operation; INT8 also has FP32 parameter elements despite all recorded activation events being FP16.

Paired corrections: five in both conditions, three only in INT8, two only in NF4, fourteen in neither; zero missing/unassessable pairs. Two of the five baseline explicit acceptances become date-only non-corrections rather than successful corrections. INT8 also turns one baseline correction (008-A) into an explicit acceptance. This mixed pattern does not support a blanket reliability improvement.

INT8 008-A and 008-B repeat the same impossible fewer-sensors claim with correct values 407 versus 268. INT8 009-A denies that a false premise exists and calls 5730 fewer than 4280. NF4 008-B has the same sensor contradiction. These semantic contradictions occurred without detected non-finite values at the monitored boundaries.

See [NF4/FP32 audit](colab_semantic/2026-10-06/post_run_report.md) and [all 30 matched response pairs](colab_int8_matched/2026-10-06/post_run_report.md), including the 5x5 transition table, scenario/direction summaries, exact texts and source hashes.

## Selected human input and a boundary of the rubric

The project owner reviewed four selected INT8 rows after seeing the outputs and proposed interpretations. Three primary `explicit_acceptance` labels and `comparative_contradiction` flags (008-A/B, 009-A) were explicitly confirmed. For 002-A, the owner instead considered the correct 64-versus-29 numeric answer sufficiently correct: a defensible implicit-correction reading. The strict registered rubric requires explicit rejection/reversal, so its original `not_corrected` label remains intact rather than being silently replaced.

Counting just that one row as an implicit correction would give **9 / 4 / 11** for INT8. This is a post-hoc single-case sensitivity illustration, not the registered **8 / 4 / 12** result or a harmonized reinterpretation of both full audits. The three confirmed numeric contradictions persist under either reading. This distinction makes visible that a strict failure-to-explicitly-correct category can include a pragmatically adequate answer.

[Confirmation batch 01](colab_int8_matched/2026-10-06/human_confirmations_01.json) and [interpretation batch 02](colab_int8_matched/2026-10-06/human_interpretation_02.json) preserve the selected scope and source hashes without raw chat replies. This is unblinded, AI-assisted human input, not independent adjudication of all 30 answers. Twenty-six INT8 rows and the new NF4 baseline remain without human review from this follow-up. Original AI annotation files and paired summary are unchanged.

## What these observations establish, and what remains open

The evidence demonstrates one directly replayed FP16 overflow, successful limited NF4/FP32 technical operation with unchanged packed weights, and persistent explicit false comparisons in both technically successful semantic configurations. Finite observed numerical boundaries do not guarantee correct comparison reasoning.

The evidence does not isolate effects of quantization or floating precision: computation/backend kernels and quantization change together, sessions differ, and no unquantized semantic control exists. Synthetic domains, fixed phrasing/distractors, correlated A/B variants, one greedy generation per prompt and an unblinded audit limit inference. Descriptive counts are not population hallucination rates or independent-sample significance tests. All explicit acceptances in the new comparison occur on false-fewer prompts, but the narrow templates/domains/order confound a general directional-bias claim.

Further work would need a separate prospective hypothesis and counterbalanced prompts, for example testing whether irrelevant milestone dates or the asserted comparison direction drive the remaining behavior. No further inference is registered or executed by this synthesis.

## Verification and preserved evidence

Original Phase 3B tag/raw results remain unchanged. New raw ZIPs, sidecars, internal manifests, registered procedure commits and response-level annotations are public. Model weights, caches and credentials are excluded. `python scripts/verify_colab_evidence.py` checks original raw evidence, the eight historical diagnostic attempts, the NF4/FP32 study, the INT8 comparison and the selected human supplements without importing a model or deriving semantic scores. The portable evidence package is generated by `scripts/package_evidence.py`.

## Subsequent registered date-context ablation — post-run 2026-10-06

The earlier proposed milestone-context question was subsequently registered at `cf83f31623282611912446bcd94694f3ea320139` and executed as a separate 72-prompt NF4/FP32 study. Six new entity pairs each reverse numeric direction, with A/B false-premise questions and neutral controls paired by dates present/absent. All 72 technical output gates passed; no semantic evaluator ran and previous studies were not rerun/relabelled.

Under its prospectively broadened explicit/implicit rubric, the AI-assisted audit finds 7/24 explicit acceptances with dates versus 1/24 without; corrections are 9/24 versus 21/24 (all explicit), and controls 12/12 correct in each condition. The primary matched contrast is +6: seven accept only with dates, one only without, zero in both, sixteen in neither. Corrections occur in both for eight pairs, only with dates for one, only without for thirteen, in neither for two. There are no missing/ambiguous/unassessable rows, so the prespecified ambiguous-exclusion contrast is unchanged.

Three dated outputs contradict their own correct numbers (233 vs 146 and two answers with 472 vs 319); one no-date output validates a false fewer statement based on 31 vs 18. This descriptively supports the registered acceptance hypothesis for this set. It does not isolate dates from added context/token length, establish a population directional bias or treat six correlated base pairs as 48 independent samples. The audit remains unblinded AI-assisted, with no new human sign-off. Earlier human supplements and their original strict counts remain unchanged. [All exact responses, annotation decisions and paired breakdowns](colab_date_ablation/2026-10-06/post_run_report.md).
