# A 90-second walkthrough

## 0:00–0:20 — The question

“What happens when a question assumes the opposite of the numbers in its own context?”

Show HO-006-B. Dravic has 9 production lines; Hollin has 14. The instruction explicitly asks the model to
correct false premises. The question asks what explains Dravic having more.

## 0:20–0:40 — The observed answer

> Dravic Industries has more production lines because it has 9 production lines while Hollin Motors has 14 production lines.

The numbers are correctly repeated, but the comparison is false. This is a stored response from the
original run, not a live demo or an illustrative answer.

## 0:40–1:00 — The experiment

Show the frozen-results table in the [report](report.md). There were 12 pre-registered synthetic scenarios,
24 false-premise questions and six neutral controls. One greedy generation per prompt: 16 PASS, six
POTENTIAL, two DETECTED for the false-premise prompts; all six controls passed. The second detected case,
HO-008-B, states a false “fewer” comparison with 520 versus 310.

## 1:00–1:20 — Why the audit is separate

Show HO-009-A/B in the [full audit](audits/phase3b-2026-10-02_111432/human_semantic_audit.md). Both answers
say 4,850 is fewer than 3,920; the lexical evaluator returned POTENTIAL. The post-hoc semantic layer finds
four accepted comparisons across three scenarios. Keep the primary result at two DETECTED. Five selected
semantic labels were human-confirmed with AI assistance; the full audit was AI-assisted.

## 1:20–1:30 — What can be checked

“All 30 original answers, the model configuration and raw-file hashes are in the repository. You can
verify them without loading a model. This is evidence for a specific failure class in one 4-bit/offloaded
configuration, not a general hallucination rate. Higher-precision and repeated-run confirmation remain open.”

Use `python scripts/verify_evidence.py` for a read-only demonstration. No live model is needed.
