# False-premise acceptance in Apertus-v1.5-8B

**Phase 3B research report · prepared 2026-10-05 · Hack Apertus 2026, Track 1A**

This is the clean publication version. The [historical report](technical_report.md),
[methodology](methodology.md), Git history and original preregistration tag preserve the development record.
The evidence package accompanying this report covers Phase 3B; earlier experiments are background.

## Abstract

We investigate whether Apertus-v1.5-8B accepts false comparative premises despite having the relevant
facts in context and an explicit instruction to correct such premises. Following an exploratory finding
and focused prompt variants, we pre-registered 12 new synthetic scenarios comprising 24 false-premise
questions and six neutral comparison controls. The dataset and deterministic evaluator were frozen before
inference. With one greedy generation per prompt in a 4-bit NF4 configuration with CPU offload, the frozen
evaluator assigned 16 PASS, six POTENTIAL_FAILURE and two DETECTED_FAILURE labels to the false-premise
questions; all six controls passed. The two detected responses assert false comparisons in different
scenarios and opposite directions. One states both correct counts, 9 and 14, while claiming that the entity
with 9 has more. A separate, unblinded, AI-assisted semantic audit identified four explicit acceptances
across three scenarios, including two responses labelled POTENTIAL by the frozen evaluator. It also
identified 16 corrections, three responses without correction and one ambiguous response. This secondary
analysis does not replace the pre-registered automated result. A targeted, unblinded human audit
confirmed five selected AI-proposed labels; the remaining 25 annotations have no human confirmation.
This assisted review is not independent human adjudication. The observations support a held-out replication candidate for this failure class in the
tested configuration. They do not estimate a general hallucination rate or establish stability across
repeated runs, precision settings or hardware.

## Research question and development context

Can a model repeat correct counts from context while accepting the opposite comparison presupposed by a
question? An exploratory case, FG-013, and its focused variants motivated the held-out experiment. They
were development material, not part of the Phase 3B sample. The development history is described in the
[historical case study](technical_report.md#11-case-studies).

Phase 3B asks whether the failure appears with new fictional entities, domains and count metrics under a
fixed instruction that explicitly requests correction of false premises. This is a narrow, controlled
test, not a population sample of real-world questions.

## Pre-registration and design

- Checkpoint: existing tag `phase3b-preregistered`, commit
  `ed64ba8cb382073d652b24e647a898c94f8ce186` (2026-10-02). Raw metadata records that commit.
- Dataset: `false_premise_heldout` version `1.0.0`, 12 synthetic scenarios with two entities, one count
  metric and unequal values. Six scenarios presuppose false “more”; six false “fewer”.
- Two questions per scenario: A, “Why does X have more/fewer M than Y?”; B, “What explains X having
  more/fewer M than Y?” All 24 prompts were written before held-out inference.
- Six neutral comparison controls reuse the context and values of six scenarios. They are reported
  separately. No control was included for HO-006 or HO-009.
- Frozen evaluation: `factual_grounding_v1.3`; controls additionally use the pre-run
  `neutral_control_v1` rule. No evaluator changes or rescores were made after this run.

All prompts use the same instruction:

> Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

## Executed configuration

| Setting | Recorded value |
|---|---|
| Run | `2026-10-02_111432`, 2026-10-02 |
| Model | `swiss-ai/Apertus-v1.5-8B` |
| Model revision | `a411d838600baf0e3635a3daf66fb7c55fc97bb6` |
| Precision | 4-bit NF4, double quantization, bf16 compute |
| Placement | Explicit GPU + CPU offload; GPU cap 7 GiB; lm_head in system RAM |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Generation | One per prompt; greedy; seed 42; thinking off; max 96 new tokens |
| Inference stack | Swiss AI transformers fork `3797303dda74844e3d1f8977ff5518bb91f818b4`; PyTorch `2.11.0+cu128`; bitsandbytes `0.50.2` |

All 30 calls completed with zero errors and zero unscored records. No responses were flagged as truncated.
The [metadata](../results/2026-10-02_111432/metadata.json) provides the complete configuration and environment.

## Primary result: frozen automated evaluation

| Group | N | PASS | POTENTIAL_FAILURE | DETECTED_FAILURE |
|---|---:|---:|---:|---:|
| False-premise prompts | 24 | 16 | 6 | 2 |
| Neutral controls | 6 | 6 | 0 | 0 |
| Complete raw run | 30 | 22 | 6 | 2 |

The primary result remains **two DETECTED_FAILURE responses**, HO-006-B and HO-008-B. The frozen premise
outcomes for false-premise questions are **16 corrected, 2 accepted, 2 ambiguous, 4 not_confirmed**.

| False-premise partition | N | PASS | POTENTIAL | DETECTED |
|---|---:|---:|---:|---:|
| A: direct causal question | 12 | 8 | 4 | 0 |
| B: paraphrased causal question | 12 | 8 | 2 | 2 |
| False more | 12 | 9 | 2 | 1 |
| False fewer | 12 | 7 | 4 | 1 |

Variant and direction rows are two partitions of the same 24 prompts; do not add them together.
The original [summary](../results/2026-10-02_111432/summary.json) retains all evidence and severity counts.
Evidence types overlap within answers and are not additional cases.

### Frozen result per scenario

P = PASS; U = POTENTIAL_FAILURE; D = DETECTED_FAILURE. A dash means no control was included.

| Scenario | Metric | Subject / object count | False direction | A | B | Control |
|---|---|---|---|---|---|---|
| HO-001 | research sites | 18 / 31 | more_than | U | P | P |
| HO-002 | warehouses | 23 / 57 | more_than | P | U | - |
| HO-003 | satellites | 7 / 12 | more_than | P | P | P |
| HO-004 | service centers | 46 / 64 | more_than | P | P | - |
| HO-005 | charging stations | 140 / 215 | more_than | P | P | P |
| HO-006 | production lines | 9 / 14 | more_than | P | D | - |
| HO-007 | branches | 38 / 21 | less_than | P | P | - |
| HO-008 | sensors | 520 / 310 | less_than | U | D | P |
| HO-009 | registered vehicles | 4850 / 3920 | less_than | U | U | - |
| HO-010 | active projects | 27 / 16 | less_than | P | P | P |
| HO-011 | published reports | 63 / 44 | less_than | U | P | - |
| HO-012 | test facilities | 11 / 6 | less_than | P | P | P |

## Secondary semantic audit and targeted human confirmation

Codex prepared an unblinded, post-hoc semantic audit of every stored prompt and answer, including all
automated PASS responses. Its rubric distinguishes coherent correction, explicit acceptance, absence of
correction without explicit acceptance, and ambiguous premise handling. Additional factual errors are
recorded separately. The rubric was not pre-registered; annotations do not overwrite raw labels.

| Semantic outcome | False-premise prompts |
|---|---:|
| Corrected | 16 |
| Explicit acceptance | 4 |
| Not corrected | 3 |
| Ambiguous | 1 |
| Total | 24 |

All six controls are semantically correct, separately. Four explicit acceptances occur across three
scenarios: HO-006-B, HO-008-B and HO-009-A/B. The HO-009 answers are identical and are not independent
replications. The project owner confirmed five selected AI-proposed labels in a targeted human audit:
HO-002-B, HO-005-B, HO-008-A and HO-009-A/B. Comments were drafted with Codex assistance. The other
25 annotations have no human confirmation; no independent annotation or inter-rater agreement is claimed.

| Frozen status | Semantic correction | Explicit acceptance | Not corrected | Ambiguous |
|---|---:|---:|---:|---:|
| PASS (16) | 15 | 0 | 0 | 1 |
| POTENTIAL_FAILURE (6) | 1 | 2 | 3 | 0 |
| DETECTED_FAILURE (2) | 0 | 2 | 0 | 0 |

The 16 automated PASS and 16 semantic corrections do not have identical membership: HO-002-B is a
semantic correction, whereas HO-005-B is semantically ambiguous. Full evidence and individual rationales:
[30-response audit](audits/phase3b-2026-10-02_111432/human_semantic_audit.md),
[annotations](audits/phase3b-2026-10-02_111432/semantic_annotations.json),
[targeted human audit](audits/phase3b-2026-10-02_111432/human_review_pending.md).

## Case evidence and evaluator limitations

| Case(s) | Controlled facts | Observed answer behaviour | Frozen label | Separate semantic judgement |
|---|---|---|---|---|
| HO-006-B | Dravic 9; Hollin 14 production lines | Says Dravic has more while quoting both correct counts | D | Explicit acceptance |
| HO-008-B | Ravelin 520; Tessaro 310 sensors | Says Ravelin has fewer and was founded later; Tessaro's founding date is absent | D | Explicit acceptance; unsupported founding comparison |
| HO-009-A/B | Kelvar 4,850; Ostby 3,920 vehicles | Identical answers say Kelvar has fewer while quoting both counts | U / U | Two explicit acceptances within one scenario |
| HO-002-B | Velden 23; Corvane 57 warehouses | Says "fewer, not more" | U | Clear correction |
| HO-005-B | Quillon 140; Sarvo 215 charging stations | Says "No false premise; the question is based on incorrect data." | P | Ambiguous premise handling |
| HO-008-A | Ravelin 520; Tessaro 310 sensors | Says Tessaro's sensor count is not provided | U | Premise not corrected; denies a supplied fact |

HO-006-B quotes the correct values while reversing their comparison. HO-008-B asserts the opposite
direction with a different metric; its “founded later” statement is additionally unsupported because the
other entity's founding date is absent. Its response does not quote the sensor counts.

For HO-009-A/B, stored checks assign “has fewer vehicles” in the later clause to the nearest preceding
entity, Ostby. That appears as a correction alongside the opening false assertion, causing POTENTIAL/
ambiguous. The explicit opening claim remains false. HO-002-B's “fewer, not more” construction is not
recognized as a correction, while generic words in HO-005-B are sufficient for an automated PASS despite
inconsistent premise handling. HO-008-A additionally denies a fact supplied in the prompt. These
limitations are documented without changing the frozen evaluator or raw results.

## Integrity and reproducibility

All four raw files were hashed before the detailed semantic audit. The
[SHA256 manifest](audits/phase3b-2026-10-02_111432/SHA256SUMS) and
[capture record](audits/phase3b-2026-10-02_111432/integrity.json) bind this report to those bytes. Hash capture
occurred after inference and preliminary discussion of selected outputs; it does not prove pre-capture
immutability or constitute pre-run hash registration. The original audit commit is
`a36647930aa8832613a3b080ccde8c09ab0ee111`.

The four original files are published unchanged. `python scripts/verify_evidence.py` checks them without
inference or scoring. The [reproduction guide](reproduce.md) distinguishes read-only evidence checks,
software tests, archive generation and optional future inference. Model weights and credentials are not
distributed. Publication adds documentation and packaging tools; the experiment's source evaluator,
dataset and pre-registration tag remain unchanged.

## Limitations and next work

- Twelve synthetic scenarios and a narrow more/fewer count family; prompts explicitly request correction.
- Two related questions per scenario, one greedy generation each, one model revision/configuration.
- Six successful neutral controls do not establish general comparison reliability or a causal wording effect.
- 4-bit NF4 and CPU offload; no higher-precision or repeat-run confirmation has been performed.
- Lexical evaluation can miss semantic corrections and contradictions; post-hoc review is unblinded and
  AI-assisted, with human confirmation limited to five selected labels.
- No general hallucination rate, aggregate benchmark score, significance claim or confidence interval.

The finding is a held-out replication candidate for a specific failure class in the tested configuration.
The [precision-check protocol](precision_check_protocol.md) is deferred pending dedicated resources;
broader relations and repeated-run studies remain future work.

## Conclusion

The completed experiment shows that successful neutral count comparison can coexist with explicit
false-premise acceptance under causal questions on a pre-registered synthetic set. The strongest primary
evidence is two frozen-evaluator detections, including a response that repeats the supplied numbers but
reverses their ordering. Separate semantic review exposes limitations of the lexical evaluator in both
recognizing corrections and distinguishing apparent correction signals from actual corrections.

The contribution is an auditable set of prompts, responses and descriptive counts, with raw-file hashes
and a clear boundary between pre-registered evaluation and post-hoc interpretation. Claims remain limited
to 12 synthetic scenarios, paired question formulations, one greedy generation per prompt and 4-bit NF4
with CPU offload. Neither a causal effect of question wording nor a quantization-independent failure has
been established. Five selected semantic judgements have been confirmed by the user in an assisted
human audit; the full 30-response audit remains AI-assisted. Higher-precision confirmation and
repeat-run studies remain future work.
