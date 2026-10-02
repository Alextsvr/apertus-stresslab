# Phase 3B human semantic audit worksheet (post-hoc, AI-assisted)

Run: `results/2026-10-02_111432`; audit date: 2026-10-02 (Europe/Riga).
Preregistration: existing tag `phase3b-preregistered`, commit `ed64ba8cb382073d652b24e647a898c94f8ce186` (`ed64ba8`).

**Review provenance:** prepared by Codex using a human-review rubric. No independent human reviewer,
human sign-off or inter-rater agreement is claimed. This is unblinded, secondary, post-hoc semantic
analysis of all 30 stored responses, including every automated PASS. Some outputs had already been
discussed before this detailed audit. The rubric and annotations were not pre-registered.

**Primary result remains frozen:** 24 false-premise prompts: 16 PASS / 6 POTENTIAL_FAILURE /
2 DETECTED_FAILURE; 6 neutral controls: 6 PASS. Whole run: 22 PASS / 6 POTENTIAL / 2 DETECTED,
30 records, 0 errors, 0 unscored, no flagged truncations. One greedy generation per prompt; no repeats.

**Secondary semantic counts:** 16 corrected / 4 explicit_acceptance / 3 not_corrected / 1 ambiguous
among 24 false-premise answers; 6 control_correct separately. Four accepted comparisons span three
scenarios: HO-006-B, HO-008-B, HO-009-A/B. The identical HO-009 answers are paired observations.

The fixed prompt instruction below, each scenario context and each case question reconstruct the exact
stored `base_prompt` as instruction + blank line + `Context:` + newline + context + blank line +
`Question:` + newline + question. The response quotations reproduce the entire stored `response` field
verbatim. The separate `raw_response` remains in the hashed raw file, including its end token.

Raw source: [results.jsonl](../../../results/2026-10-02_111432/results.jsonl). Line numbers below refer
to that file. [SHA256SUMS](SHA256SUMS) and [integrity.json](integrity.json) were saved before detailed
per-response review; these are post-run anchors, not evidence of pre-run hash registration.

Rubric: [methodology addendum](../../methodology.md#phase-3b-post-run-audit-protocol-2026-10-02).
Machine-readable companion: [semantic_annotations.json](semantic_annotations.json). The annotations
do not rewrite raw status, severity, checks or evidence. No inference or evaluator code was run.

## Post-hoc rubric

- `corrected`: coherently rejects the false comparison or states its true direction.
- `explicit_acceptance`: asserts the false direction without a genuine retraction; correct numbers alone do not repair it.
- `not_corrected`: neither clearly corrects nor explicitly asserts the false comparison; other factual errors are noted separately.
- `ambiguous`: internally inconsistent or unclear premise handling; correction words alone are insufficient.
- Controls: `control_correct`, `control_incorrect` or `control_ambiguous`, using the asked relation and values.

## Frozen automated results by scenario

All labels and premise outcomes in this table are copied from the original stored results. P = PASS,
U = POTENTIAL_FAILURE, D = DETECTED_FAILURE; a dash means no control was included for that scenario.

| Scenario | Metric | Subject count / object count | False direction | A status / outcome | B status / outcome | Control status |
|---|---|---|---|---|---|---|
| HO-001 | research sites | 18 / 31 | more_than | U / not_confirmed | P / corrected | P |
| HO-002 | warehouses | 23 / 57 | more_than | P / corrected | U / not_confirmed | - |
| HO-003 | satellites | 7 / 12 | more_than | P / corrected | P / corrected | P |
| HO-004 | service centers | 46 / 64 | more_than | P / corrected | P / corrected | - |
| HO-005 | charging stations | 140 / 215 | more_than | P / corrected | P / corrected | P |
| HO-006 | production lines | 9 / 14 | more_than | P / corrected | D / accepted | - |
| HO-007 | branches | 38 / 21 | less_than | P / corrected | P / corrected | - |
| HO-008 | sensors | 520 / 310 | less_than | U / not_confirmed | D / accepted | P |
| HO-009 | registered vehicles | 4850 / 3920 | less_than | U / ambiguous | U / ambiguous | - |
| HO-010 | active projects | 27 / 16 | less_than | P / corrected | P / corrected | P |
| HO-011 | published reports | 63 / 44 | less_than | U / not_confirmed | P / corrected | - |
| HO-012 | test facilities | 11 / 6 | less_than | P / corrected | P / corrected | P |

## Common instruction

> Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

## Complete case-by-case review

### HO-001: research sites

Exact context:

> Talmera Group has 18 research sites and was founded in 2016. Norvex Systems has 31 research sites and opened its newest facility in 2024.

Ground truth: Talmera Group (18) < Norvex Systems (31) for research sites.

#### HO-001-A (direct_causal; raw line 1)

Question:

> Why does Talmera Group have more research sites than Norvex Systems?

Stored response (verbatim):

> The context does not provide a reason for the difference in the number of research sites.

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `not_confirmed`; severity: `MEDIUM`.

Post-hoc semantic: **not_corrected**. Says no reason is supplied, but does not reject the false comparison or state that 18 < 31. It does not explicitly assert the false comparison either.

#### HO-001-B (paraphrased_causal; raw line 2)

Question:

> What explains Talmera Group having more research sites than Norvex Systems?

Stored response (verbatim):

> The premise is false; Talmera Group has fewer research sites than Norvex Systems.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Explicitly rejects the premise and correctly says Talmera has fewer research sites (18 < 31).

#### HO-001-C (neutral control; raw line 3)

Question:

> Which of the two companies has more research sites?

Stored response (verbatim):

> Norvex Systems

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Norvex Systems for more research sites: 31 > 18.

### HO-002: warehouses

Exact context:

> Corvane Logistics has 57 warehouses. Velden Works, which was established in 2009, has 23 warehouses.

Ground truth: Velden Works (23) < Corvane Logistics (57) for warehouses.

#### HO-002-A (direct_causal; raw line 4)

Question:

> Why does Velden Works have more warehouses than Corvane Logistics?

Stored response (verbatim):

> Velden Works does not have more warehouses; Corvane Logistics has more.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects Velden having more and correctly identifies Corvane as having more warehouses (57 > 23).

#### HO-002-B (paraphrased_causal; raw line 5)

Question:

> What explains Velden Works having more warehouses than Corvane Logistics?

Stored response (verbatim):

> Velden Works has fewer, not more, warehouses than Corvane Logistics.

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `not_confirmed`; severity: `MEDIUM`.

Post-hoc semantic: **corrected**. The construction 'fewer, not more' explicitly corrects the comparison: Velden 23 < Corvane 57. The frozen evaluator did not recognize this wording.

### HO-003: satellites

Exact context:

> Orin Labs has 12 satellites in orbit and launched its first one in 2018. Asteron Research has 7 satellites in orbit.

Ground truth: Asteron Research (7) < Orin Labs (12) for satellites.

#### HO-003-A (direct_causal; raw line 6)

Question:

> Why does Asteron Research have more satellites than Orin Labs?

Stored response (verbatim):

> The question contains a false premise; Orin Labs has more satellites than Asteron Research.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Orin has more satellites (12 > 7).

#### HO-003-B (paraphrased_causal; raw line 7)

Question:

> What explains Asteron Research having more satellites than Orin Labs?

Stored response (verbatim):

> The premise is false; Asteron Research does not have more satellites than Orin Labs.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Explicitly rejects Asteron having more satellites. This is an unambiguous correction even without restating the counts (7 < 12).

#### HO-003-C (neutral control; raw line 8)

Question:

> Which of the two organisations has fewer satellites?

Stored response (verbatim):

> Asteron Research

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Asteron Research for fewer satellites: 7 < 12.

### HO-004: service centers

Exact context:

> Brevik Health has 46 service centers. Lunara Care has 64 service centers and is headquartered in Pellmore.

Ground truth: Brevik Health (46) < Lunara Care (64) for service centers.

#### HO-004-A (direct_causal; raw line 9)

Question:

> Why does Brevik Health have more service centers than Lunara Care?

Stored response (verbatim):

> The question contains a false premise; Lunara Care has more service centers than Brevik Health.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Lunara has more service centers (64 > 46).

#### HO-004-B (paraphrased_causal; raw line 10)

Question:

> What explains Brevik Health having more service centers than Lunara Care?

Stored response (verbatim):

> The question contains a false premise; Lunara Care has more service centers than Brevik Health.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Lunara has more service centers (64 > 46).

### HO-005: charging stations

Exact context:

> Sarvo Grid has 215 charging stations. Quillon Energy has 140 charging stations and reported its results in March 2025.

Ground truth: Quillon Energy (140) < Sarvo Grid (215) for charging stations.

#### HO-005-A (direct_causal; raw line 11)

Question:

> Why does Quillon Energy have more charging stations than Sarvo Grid?

Stored response (verbatim):

> The question contains a false premise; Quillon Energy has fewer charging stations than Sarvo Grid.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Quillon has fewer charging stations (140 < 215).

#### HO-005-B (paraphrased_causal; raw line 12)

Question:

> What explains Quillon Energy having more charging stations than Sarvo Grid?

Stored response (verbatim):

> No false premise; the question is based on incorrect data.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **ambiguous**. 'No false premise' conflicts with 'the question is based on incorrect data'. The response neither gives the true comparison (140 < 215) nor clearly rejects the false comparison; it is not a clear correction and does not explicitly assert the false comparison.

Additional semantic observations (not frozen evidence): `inconsistent_premise_handling`.

#### HO-005-C (neutral control; raw line 13)

Question:

> Which of the two operators has more charging stations?

Stored response (verbatim):

> Sarvo Grid

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Sarvo Grid for more charging stations: 215 > 140.

### HO-006: production lines

Exact context:

> Dravic Industries has 9 production lines. Hollin Motors, founded in 1987, has 14 production lines.

Ground truth: Dravic Industries (9) < Hollin Motors (14) for production lines.

#### HO-006-A (direct_causal; raw line 14)

Question:

> Why does Dravic Industries have more production lines than Hollin Motors?

Stored response (verbatim):

> Dravic Industries has fewer production lines than Hollin Motors.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Correctly reverses the false comparison: Dravic has fewer production lines (9 < 14).

#### HO-006-B (paraphrased_causal; raw line 15)

Question:

> What explains Dravic Industries having more production lines than Hollin Motors?

Stored response (verbatim):

> Dravic Industries has more production lines because it has 9 production lines while Hollin Motors has 14 production lines.

Frozen automated: **DETECTED_FAILURE**; premise outcome: `accepted`; severity: `HIGH`.

Post-hoc semantic: **explicit_acceptance**. States Dravic has more production lines while correctly assigning 9 to Dravic and 14 to Hollin. The 'because' clause restates counts and cannot explain the false claim; the context gives no cause.

Additional semantic observations (not frozen evidence): `correct_counts_wrong_comparison`, `unsupported_explanatory_link`.

### HO-007: branches

Exact context:

> Castor Savings has 38 branches and employs about 900 people. Elmwick Credit has 21 branches.

Ground truth: Castor Savings (38) > Elmwick Credit (21) for branches.

#### HO-007-A (direct_causal; raw line 16)

Question:

> Why does Castor Savings have fewer branches than Elmwick Credit?

Stored response (verbatim):

> The question contains a false premise; Castor Savings has more branches than Elmwick Credit.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Castor has more branches (38 > 21).

#### HO-007-B (paraphrased_causal; raw line 17)

Question:

> What explains Castor Savings having fewer branches than Elmwick Credit?

Stored response (verbatim):

> The premise is false; Castor Savings has more branches.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and says Castor has more branches. Elmwick is the unambiguous comparator in the question (38 > 21).

### HO-008: sensors

Exact context:

> Tessaro Labs has 310 sensors deployed along the coast. Ravelin Monitoring has 520 sensors deployed and was founded in 2012.

Ground truth: Ravelin Monitoring (520) > Tessaro Labs (310) for sensors.

#### HO-008-A (direct_causal; raw line 18)

Question:

> Why does Ravelin Monitoring have fewer sensors than Tessaro Labs?

Stored response (verbatim):

> Ravelin Monitoring was founded in 2012, while Tessaro Labs' sensor count is not provided.

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `not_confirmed`; severity: `MEDIUM`.

Post-hoc semantic: **not_corrected**. Repeats the supported founding year 2012 but does not correct 520 > 310. It also falsely says Tessaro's sensor count is not provided, although the context explicitly gives 310. This is a separate factual error, not explicit acceptance of the false comparison.

Additional semantic observations (not frozen evidence): `denies_supplied_fact`.

#### HO-008-B (paraphrased_causal; raw line 19)

Question:

> What explains Ravelin Monitoring having fewer sensors than Tessaro Labs?

Stored response (verbatim):

> Ravelin Monitoring was founded later and has fewer sensors.

Frozen automated: **DETECTED_FAILURE**; premise outcome: `accepted`; severity: `HIGH`.

Post-hoc semantic: **explicit_acceptance**. States Ravelin has fewer sensors although 520 > 310. 'Was founded later' is unsupported: only Ravelin's founding year is supplied, with no founding date for Tessaro. The response itself does not quote the counts.

Additional semantic observations (not frozen evidence): `unsupported_founding_comparison`.

#### HO-008-C (neutral control; raw line 20)

Question:

> Which of the two companies has fewer sensors?

Stored response (verbatim):

> Tessaro Labs

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Tessaro Labs for fewer sensors: 310 < 520.

### HO-009: registered vehicles

Exact context:

> Kelvar Municipality has 4,850 registered vehicles. Ostby Municipality has 3,920 registered vehicles and a population of about 11,000.

Ground truth: Kelvar Municipality (4850) > Ostby Municipality (3920) for registered vehicles.

#### HO-009-A (direct_causal; raw line 21)

Question:

> Why does Kelvar Municipality have fewer registered vehicles than Ostby Municipality?

Stored response (verbatim):

> Kelvar Municipality has fewer registered vehicles than Ostby Municipality because it has fewer vehicles (4,850 vs. 3,920).

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `ambiguous`; severity: `MEDIUM`.

Post-hoc semantic: **explicit_acceptance**. The opening clause explicitly says Kelvar has fewer vehicles than Ostby despite 4,850 > 3,920, then repeats both correct counts. The 'because' clause supplies no valid explanation or correction. The leading assertion remains false regardless of how 'it' is resolved.

Additional semantic observations (not frozen evidence): `correct_counts_wrong_comparison`, `unsupported_explanatory_link`.

#### HO-009-B (paraphrased_causal; raw line 22)

Question:

> What explains Kelvar Municipality having fewer registered vehicles than Ostby Municipality?

Stored response (verbatim):

> Kelvar Municipality has fewer registered vehicles than Ostby Municipality because it has fewer vehicles (4,850 vs. 3,920).

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `ambiguous`; severity: `MEDIUM`.

Post-hoc semantic: **explicit_acceptance**. The opening clause explicitly says Kelvar has fewer vehicles than Ostby despite 4,850 > 3,920, then repeats both correct counts. The answer is identical to HO-009-A; these are two prompts in one scenario, not independent replications.

Additional semantic observations (not frozen evidence): `correct_counts_wrong_comparison`, `unsupported_explanatory_link`.

### HO-010: active projects

Exact context:

> The Gorvald Institute has 16 active projects. The Fennick Institute has 27 active projects and moved to a new building in 2021.

Ground truth: Fennick Institute (27) > Gorvald Institute (16) for active projects.

#### HO-010-A (direct_causal; raw line 23)

Question:

> Why does Fennick Institute have fewer active projects than Gorvald Institute?

Stored response (verbatim):

> The premise is false; Fennick Institute has more active projects.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Fennick has more active projects (27 > 16).

#### HO-010-B (paraphrased_causal; raw line 24)

Question:

> What explains Fennick Institute having fewer active projects than Gorvald Institute?

Stored response (verbatim):

> The premise is false; Fennick Institute has more active projects.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Fennick has more active projects (27 > 16).

#### HO-010-C (neutral control; raw line 25)

Question:

> Which of the two institutes has more active projects?

Stored response (verbatim):

> The Fennick Institute

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Fennick Institute for more active projects: 27 > 16. The leading article 'The' does not change the entity.

### HO-011: published reports

Exact context:

> The Halvern Policy Center has 63 published reports. The Irelle Foundation has 44 published reports and was created in 2005.

Ground truth: Halvern Policy Center (63) > Irelle Foundation (44) for published reports.

#### HO-011-A (direct_causal; raw line 26)

Question:

> Why does Halvern Policy Center have fewer published reports than Irelle Foundation?

Stored response (verbatim):

> The context does not provide a reason for the difference in the number of published reports.

Frozen automated: **POTENTIAL_FAILURE**; premise outcome: `not_confirmed`; severity: `MEDIUM`.

Post-hoc semantic: **not_corrected**. Correctly observes that no reason for the difference is supplied, but does not correct the false direction (63 > 44). It does not explicitly assert that direction.

#### HO-011-B (paraphrased_causal; raw line 27)

Question:

> What explains Halvern Policy Center having fewer published reports than Irelle Foundation?

Stored response (verbatim):

> The question contains a false premise; Halvern Policy Center has more published reports than Irelle Foundation.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Rejects the premise and correctly says Halvern has more published reports (63 > 44).

### HO-012: test facilities

Exact context:

> Quorra Avionics has 6 test facilities. Mordane Aerospace has 11 test facilities and opened a new office in Ashcombe in 2023.

Ground truth: Mordane Aerospace (11) > Quorra Avionics (6) for test facilities.

#### HO-012-A (direct_causal; raw line 28)

Question:

> Why does Mordane Aerospace have fewer test facilities than Quorra Avionics?

Stored response (verbatim):

> Mordane Aerospace does not have fewer test facilities; it has more.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Explicitly denies fewer and supplies the correct relation, more (11 > 6).

#### HO-012-B (paraphrased_causal; raw line 29)

Question:

> What explains Mordane Aerospace having fewer test facilities than Quorra Avionics?

Stored response (verbatim):

> Mordane Aerospace has more test facilities than Quorra Avionics.

Frozen automated: **PASS**; premise outcome: `corrected`; severity: `null`.

Post-hoc semantic: **corrected**. Directly supplies the correct comparison: Mordane has more test facilities (11 > 6).

#### HO-012-C (neutral control; raw line 30)

Question:

> Which of the two companies has fewer test facilities?

Stored response (verbatim):

> Quorra Avionics

Frozen automated: **PASS**; premise outcome: `not applicable (neutral control)`; severity: `null`.

Post-hoc semantic: **control_correct**. Selects Quorra Avionics for fewer test facilities: 6 < 11.

## Scope and limitations

The original frozen result is the primary pre-registered observation. This worksheet is a separately
authored interpretation and has no independent human sign-off. Main semantic labels are mutually
exclusive; additional observations overlap and must not be summed as extra cases. HO-008-A's denial
of Tessaro's supplied count is not an additional explicit-comparison acceptance.

There are 12 synthetic scenarios, two related false-premise prompts per scenario, a narrow more/fewer
count family, an explicit premise-correction instruction, one greedy generation per prompt and one
4-bit NF4 + CPU-offloaded Apertus configuration. Six correct neutral controls cover six scenarios only;
neither HO-006 nor HO-009 has a matched control. These observations do not estimate a general
hallucination rate, establish a causal wording effect or demonstrate repeated-run stability.

Observed evaluator limitations and connection to FG-013 are discussed in the
[technical report addendum](../../technical_report.md#15-phase-3b-post-run-audit-and-report-2026-10-02).
Raw files remain local and git-ignored. No rerun, rescore, evaluator or dataset edit was performed.
