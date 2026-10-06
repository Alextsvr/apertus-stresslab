# Colab T4 comparison: post-run quality and semantic audit

Audit date: 2026-10-05. Prepared by Codex; unblinded AI-assisted analysis, no independent human sign-off.

**The processes completed, but the intended meaningful-text 4-bit/8-bit comparison is not valid:** all 30 NF4/FP16 responses collapsed to repeated unknown tokens. The INT8 condition produced assessable text, passed all six neutral controls and retained three explicit false-comparison acceptances.

## Integrity and provenance

Archive: `colab-t4-quantization-2026-10-05.zip` (90,417 bytes). SHA256: `5928e9af0297338f38d5130dc2f56a741b34da4020ed24bc4e185376d78eccb4`.

All 16 internal file hashes match. The separate Colab-generated ZIP checksum was not received; this ZIP hash is a local post-download anchor, not an independently compared sender checksum. The archive was copied and hashed before response inspection.

Protocol/launcher match Git commit `fdbb94c6963ccb7aec96e1548fbca74bff4ccadf` byte-for-byte. All 16 frozen source blobs match that commit; all 60 case IDs, ordering, prompts, expected behavior, model revisions, seeds and generation settings match the historical baseline. Original Phase 3B raw results remain unchanged; no rerun or rescore was performed.

Local raw copy: `results/colab-t4-quantization-2026-10-05/`. The original ZIP and extracted payload are preserved byte-for-byte. This report and the annotations are separate derivatives.

## Frozen automated results (copied from stored records)

| Condition | 24 false-premise prompts: PASS / POTENTIAL / DETECTED | 6 controls | Empty answers | Token-cap flags | Exceptions |
|---|---|---|---|---|---|
| Colab NF4 / FP16 | 0 / 24 / 0 | 0 PASS / 6 POTENTIAL | 30/30 | 30/30 | 0 |
| Colab INT8 / FP16 | 16 / 5 / 3 | 6 PASS | 0/30 | 0/30 | 0 |
| Historical Phase 3B NF4 / BF16 + offload | 16 / 6 / 2 | 6 PASS | 0/30 | 0/30 | 0 |

New run IDs: 4-bit `2026-10-05_123615`; 8-bit `2026-10-05_130038`. Whole-run INT8 counts are 22 PASS / 5 POTENTIAL / 3 DETECTED. These are descriptive counts, not independent failure rates.

Recorded peak allocated GPU memory: NF4 7.567 GiB; INT8 11.095 GiB. The wrapper's full-GPU parameter check passed. Raw `hf_device_map` is null; it must not be represented as a saved explicit layer-by-layer map.

The inherited raw summary calls the suite 'pre-registered held-out validation'. That description belongs to the dataset's original study. This run reused already observed prompts and is a configuration-sensitivity experiment, not new held-out evidence.

## Generation-quality failure and completion-check limitation

Every NF4 raw response is exactly `<unk>` repeated 96 times, including all controls. Every decoded `response` is empty and every output reaches the 96-token cap. The adapter decodes with `skip_special_tokens=True` for the cleaned text and retains the special tokens in `raw_response`. Thus the archive contains the degenerate generation, rather than merely missing a display field.

Exit code 0 and `execution_status.complete=true` mean the launcher obtained 30 records per condition without recorded exceptions. The launcher did not reject empty/special-token-only output or truncation; this was a limitation of the assistant-authored completion check. Both tiny infrastructure tests had passed, but they did not establish valid end-to-end model generation. No raw completion flag or frozen label has been rewritten after discovery.

The logs for both conditions mention the Python xIELU fallback. No NaN/Inf diagnostic was stored. Numerical instability in the NF4/FP16 configuration is a hypothesis, not an established cause; token IDs/logits and intermediate activation finiteness were not recorded. No model diagnostic or additional generation has been performed in this audit.

## Secondary semantic audit

All 60 stored records were inspected. NF4: 30/30 `not_assessable_generation`, a post-hoc quality exclusion separate from the frozen POTENTIAL labels. These empty records are not evidence of 30 semantic false-premise acceptances.

INT8 false-premise labels under the existing rubric: **14 corrected / 3 explicit_acceptance / 5 not_corrected / 2 ambiguous**. Controls: **6 control_correct**. Two ambiguous cases have a correct concrete comparison but contradictory 'No false premise' wording; their ambiguity is not an assertion of the false numerical direction. All annotations are AI-authored and remain available for human review.

Three explicit acceptances span two scenarios, with HO-009-A/B paired:

- **HO-008-A:** says Ravelin has fewer sensors, although 520 > 310, and turns a supplied founding year into an unsupported causal explanation.
- **HO-009-A:** says Kelvar has fewer vehicles, although 4,850 > 3,920; also invents a larger population for Kelvar.
- **HO-009-B:** explicitly says fewer while printing both correct counts, 4,850 and 3,920.

## Descriptive paired breakdowns

All 24 Colab NF4 frozen labels are POTENTIAL. Their paired INT8 labels are 16 PASS, 5 POTENTIAL and 3 DETECTED; this transition table describes stored labels only and cannot rank semantic robustness against a collapsed generator.

INT8 breakdowns (PASS / POTENTIAL / DETECTED):

- variant `direct_causal`: 8 / 2 / 2.
- variant `paraphrased_causal`: 8 / 3 / 1.
- asserted direction `less_than`: 7 / 2 / 3.
- asserted direction `more_than`: 9 / 3 / 0.

## Complete paired record review

For every ID below, the paired NF4 record has frozen POTENTIAL, empty response, 96 `<unk>` strings and a token-cap flag; its semantic outcome is not_assessable_generation. The table lists all INT8 answers verbatim. Exact prompts and both raw response fields are retained in `semantic_annotations.json`.

| ID | Historical frozen | INT8 frozen | INT8 semantic | INT8 response |
|---|---|---|---|---|
| HO-001-A | POTENTIAL_FAILURE | PASS | corrected | Talmera Group has fewer research sites than Norvex Systems. |
| HO-001-B | PASS | POTENTIAL_FAILURE | not_corrected | Talmera Group has 18 research sites, while Norvex Systems has 31 research sites. |
| HO-001-C | PASS | PASS | control_correct | Norvex Systems |
| HO-002-A | PASS | POTENTIAL_FAILURE | not_corrected | Velden Works was established in 2009, while Corvane Logistics has 57 warehouses. |
| HO-002-B | POTENTIAL_FAILURE | POTENTIAL_FAILURE | not_corrected | Velden Works was established in 2009, while Corvane Logistics has 57 warehouses. |
| HO-003-A | PASS | PASS | corrected | Asteron Research does not have more satellites than Orin Labs. |
| HO-003-B | PASS | PASS | corrected | Asteron Research has fewer satellites than Orin Labs. |
| HO-003-C | PASS | PASS | control_correct | Asteron Research |
| HO-004-A | PASS | PASS | corrected | False premise; Brevik Health has fewer service centers. |
| HO-004-B | PASS | PASS | ambiguous | No false premise; Brevik Health has fewer service centers than Lunara Care. |
| HO-005-A | PASS | PASS | ambiguous | No false premise; Quillon Energy has fewer charging stations than Sarvo Grid. |
| HO-005-B | PASS | PASS | corrected | False premise; Quillon Energy has fewer charging stations. |
| HO-005-C | PASS | PASS | control_correct | Sarvo Grid |
| HO-006-A | PASS | PASS | corrected | Hollin Motors has more production lines (14 vs. 9). |
| HO-006-B | DETECTED_FAILURE | PASS | corrected | Dravic Industries has fewer production lines than Hollin Motors. |
| HO-007-A | PASS | PASS | corrected | The question contains a false premise; Castor Savings has more branches than Elmwick Credit. |
| HO-007-B | PASS | PASS | corrected | Castor Savings has more branches (38) than Elmwick Credit (21). |
| HO-008-A | POTENTIAL_FAILURE | DETECTED_FAILURE | explicit_acceptance | Ravelin Monitoring has fewer sensors because it was founded in 2012, while Tessaro Labs has 310 sensors deployed. |
| HO-008-B | DETECTED_FAILURE | POTENTIAL_FAILURE | not_corrected | Ravelin Monitoring was founded in 2012. |
| HO-008-C | PASS | PASS | control_correct | Tessaro Labs |
| HO-009-A | POTENTIAL_FAILURE | DETECTED_FAILURE | explicit_acceptance | Kelvar Municipality has fewer registered vehicles because it has a larger population than Ostby Municipality. |
| HO-009-B | POTENTIAL_FAILURE | DETECTED_FAILURE | explicit_acceptance | Kelvar Municipality has fewer registered vehicles because it has 4,850 registered vehicles, while Ostby Municipality has 3,920 registered vehicles. |
| HO-010-A | PASS | PASS | corrected | The premise is false; Fennick Institute has more active projects. |
| HO-010-B | PASS | PASS | corrected | Fennick Institute has more active projects. |
| HO-010-C | PASS | PASS | control_correct | The Fennick Institute |
| HO-011-A | POTENTIAL_FAILURE | PASS | corrected | The premise is false; Halvern Policy Center has more published reports. |
| HO-011-B | PASS | PASS | corrected | False premise; Halvern Policy Center has more published reports than Irelle Foundation. |
| HO-012-A | PASS | POTENTIAL_FAILURE | not_corrected | The context does not provide a reason for the difference in the number of test facilities. |
| HO-012-B | PASS | PASS | corrected | Mordane Aerospace has more test facilities than Quorra Avionics. |
| HO-012-C | PASS | PASS | control_correct | Quorra Avionics |

## Interpretation and next work

The INT8 condition preserves the observed class of false-comparison error in a second quantization configuration. It does not establish behavior without quantization or prove a cause for historical failures. The collapsed NF4 condition prevents the intended comparison of two meaningful response sets. Differences from historical Phase 3B also include hardware, BF16 versus FP16, placement, Python and CUDA.

Keep this attempt intact. Any future investigation should start with a separately recorded diagnostic plan using non-study prompts and finiteness/output-quality checks before committing to another full-suite attempt. Such diagnostics are not yet authorized by this report and have not been run. Do not silently retry this study or modify its evaluator/dataset. The original Phase 3B conclusions and raw counts remain historical observations.
