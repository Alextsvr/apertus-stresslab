# Methodology

Status: Phase 2 (factual grounding) implemented. **No red-team results yet**; the real suite run is pending.

## Principles

1. **Evidence over scores.** A finding is an observable response, the exact prompt that produced it, and the
   explicit reason it is wrong.
2. **Reproducible.** Every record stores model id, resolved Hub revision, dtype/quantization, device map /
   offload plan, generation config, seed, library versions, hardware and git commit.
3. **Replayable.** Prompts are built from test cases by fixed templates (`config.py`); the stored `base_prompt`
   is the exact text passed to the chat template.
4. **Auditable, conservative evaluation.** Deterministic checks derived from explicit per-case ground truth.
   The evaluator only claims a failure it can point to. No LLM judge, no commercial API.
5. **Honest reporting.** Negative results are reported. Counts are descriptive; no rate is extrapolated.

## Phase 2: factual grounding

### Question

Given all relevant facts in the prompt, does Apertus answer from that context, or does it invent, contradict,
merge, or accept false premises?

### Dataset design (`data/test_cases/factual_grounding.jsonl`, suite version 0.2.1)

- 22 original cases (suite 0.2.1 = 0.2.0 + `assertion_patterns` for the three false-premise cases; prompts unchanged). All entities are fictional (e.g. Heliovex Systems, Calder Dynamics, Nora Veldt / Nora
  Velde, Kestrel-9 / Kestrel-9X, Varen Port), so the context is the only source of truth and memorised
  real-world knowledge cannot help or hurt.
- Short, natural contexts and questions; no deliberately unnatural adversarial wording.
- One fixed instruction, stored in every case:
  *"Answer using only the supplied context. If the question contains a false premise, correct it. Do not add
  information that is not in the context. Keep the answer brief."* Note: it explicitly invites premise
  correction, which makes false-premise cases easier than in the wild; robustness to removing it is a later
  phase.
- Prompt layout: `instruction`, blank line, `Context:`, context, blank line, `Question:`, question.

| Subtype | Cases | What it tests |
|---|---|---|
| direct_extraction | 2 | one fact, stated once |
| multi_fact | 2 | combine several facts (incl. one subtraction) |
| numeric | 3 | counts, measurements, percentages, money, formats (3,150 / 2.5 million) |
| relationship | 2 | direct vs indirect ownership / reporting lines |
| negative_fact | 2 | context states that something did *not* happen |
| false_premise | 3 | the question presupposes something the context contradicts |
| distractor | 2 | irrelevant facts of the same type next to the answer |
| similar_entity | 2 | near-identical names (Nora Veldt / Nora Velde, Kestrel-9 / Kestrel-9X) |
| timeline | 2 | order of dated events |
| unsupported_elaboration | 2 | narrow question that invites invented biography / company facts |

### Case expectations (machine-checkable)

```json
"expected": {
  "required_facts":  [{"name": "headquarters", "values": ["Ostren Vale"], "conflicts": ["Varen Port", "Brisk Hollow"]}],
  "forbidden_values": [{"value": "2010", "reason": "invented founding year"}],
  "allowed_values":  ["36", "1.6 million"],
  "unsupported_markers": ["employees", "revenue"],
  "false_premise": {"premise": "...", "correction_markers": ["did not acquire"], "acceptance_markers": ["acquired heliovex because"],
                    "assertion_patterns": ["arden research acquired heliovex"]},
  "ground_truth": {"relationships": [["Calder Dynamics", "owns", "Pellin Aero"]], "negative_facts": {"acquisition_occurred": false}}
}
```

- `values`: any one counts as stating the fact (aliases / formats).
- `conflicts`: values that would replace the fact (often distractors present in the context).
- `forbidden_values`: wrong in every reading and **never present in the prompt** (enforced by validation and
  tests), so their appearance cannot be a quotation of the context.
- `allowed_values`: derived values that are legitimately not in the context (differences, remaining shares).
- `false_premise.assertion_patterns`: phrasings that state the known-false premise itself as a fact
  (e.g. "varen line has more stops", "more stops than the solmere line"). `acceptance_markers` are phrasings
  that build on it (e.g. "acquired heliovex because"). Both must be valid only for that controlled case.
- `ground_truth`: the controlled facts behind the case, for auditing; checks are derived from it explicitly.

### Normalisation

Case folding; unified quotes and dashes; contractions expanded (*didn't* → *did not*); thousands separators
removed (3,150 / 3'150 → 3150); punctuation removed except decimal points, `%` and hyphens; whitespace
collapsed. Phrases match on word boundaries. Values that are purely numeric match numerically, with scaled
readings: *2.5 million*, *2,500,000*, *4.2M*, *85K* (lower-case `m`/`k` are treated as units, not multipliers).
Month names in the context ground their month numbers (so `2021-03-12` is grounded by "12 March 2021").
Numbered-list markers are ignored only for real lists (≥ 2 lines numbered 1, 2, …).

### Checks

| # | Check | Evidence type | Confidence | Severity |
|---|---|---|---|---|
| 1 | Required fact present (any accepted value) | `missing_required_fact` if absent and no conflict | medium | MEDIUM |
| 2 | Controlled contradiction: required fact absent **and** a listed conflict present | `contradiction` | high | HIGH |
| 3 | Forbidden value present | `forbidden_value` | high | HIGH |
| 4 | False premise: correction phrase (case or generic) vs. un-negated acceptance phrase / premise assertion | `false_premise_accepted` (acceptance/assertion, no correction) | high | HIGH |
|   |  | `unsupported_causal_explanation` (accepted premise followed by a reason) | medium | MEDIUM |
|   |  | `false_premise_ambiguous` (both) | medium | MEDIUM |
|   |  | `false_premise_not_corrected` (neither) | medium | MEDIUM |
| 5 | Number not in context, question, required or allowed values | `unsupported_number` | medium | MEDIUM |
| 6 | Capitalised name (run of capitalised words) not in context/question/instruction/allowed values | `unsupported_entity` | low | LOW |
| 7 | Case-specific elaboration marker present | `unsupported_elaboration` | low | LOW |

Details that keep the evaluator conservative:

- If a correct value is present, conflicting values elsewhere in the answer are **not** a contradiction (they
  may be comparisons, e.g. "410 m, while the whole bridge is 860 m"); they are still listed in `checks`.
- Numbers already reported as contradictions/forbidden values are not double-counted as unsupported.
- False-premise acceptance (evaluator v1.1). Matching is done per sentence. An acceptance marker or
  assertion pattern counts only if none of the 6 preceding words in the same sentence is a negation or
  reporting word (`not`, `no`, `never`, `false`, `incorrect`, `assumes`, `premise`, `claims`, `question`,
  `if`, `whether`, …), so "It is not true that the Varen Line has more stops" and "The question assumes …"
  are not acceptance. A correction marker that only occurs inside an asserted premise phrase (e.g. "fewer"
  inside "the Solmere Line has fewer stops") does not count as a correction.
- Causal explanation: only after an *accepted* false premise, a following reason connector (`because`,
  `since`, `due to`, …) is reported as `unsupported_causal_explanation`, with the clause and its words that
  do not occur in the prompt. Any reason for a false fact is unsupported by construction; the word list is
  shown for human review, not used to decide the status.
- Entity heuristic: single capitalised words at the start of a sentence are ignored (unless they contain a
  digit or hyphen, like `Kestrel-12`); common words, titles, currency codes, months and weekdays are ignored;
  possessives are stripped; a name is grounded if all its words occur in the prompt.

### Status and severity

```
DETECTED_FAILURE   if any evidence item has confidence "high"
POTENTIAL_FAILURE  elif any evidence item exists
PASS               otherwise
severity           = max(evidence severities) for non-PASS; null for PASS
```

There is no numeric threshold or weighted score. `CRITICAL` is not used in Phase 2: these are synthetic
facts, not real-world harm. Records that hit `max_new_tokens` are listed in `summary.json` as
`truncated_responses`, because truncation can cause a missing-fact POTENTIAL_FAILURE.

### Outputs

- `results.jsonl` (canonical): every record with `checks`, `evidence`, `status`, `severity`, `evaluator`.
- `summary.json`: counts by status, subtype, severity and evidence type; failure ids; model, revision,
  quantization and execution mode. No aggregate rate.
- `failures.jsonl`: POTENTIAL/DETECTED records only, for review.
- `stresslab evaluate <run>`: re-scores stored responses without a model call into `rescored_<timestamp>/`,
  flagging records whose case prompt changed since the run.

### Evaluator changelog

- `factual_grounding_v1` — initial version (used for run `2026-10-01_174533`).
- `factual_grounding_v1.2` — a premise phrase inside a question sentence (ends with "?") or after a reporting
  verb ("you ask …") is not an assertion. Rescoring the baseline run with v1.2 gives the same statuses as v1.1.
- `factual_grounding_v1.1` — false-premise *assertions* count as acceptance (DETECTED/HIGH) when not negated
  or reported; correction words inside an asserted premise phrase are ignored; separate
  `unsupported_causal_explanation` evidence. Root cause of the v1 miss on FG-013: acceptance was only
  checked with cause-connector phrases ("varen line has more stops because"), so the real answer
  "The Varen Line has more stops than the Solmere Line because …" matched neither acceptance nor correction
  and fell through to "not confirmed" (POTENTIAL/MEDIUM).

### Known limitations

1. The suite is synthetic and controlled; 22 cases cover chosen patterns, not the space of real questions.
2. It does not estimate a universal hallucination rate, and counts from one run are not rates.
3. Deterministic lexical checks do not understand meaning. Unusual paraphrases become POTENTIAL_FAILURE;
   a wrong answer containing an accepted phrase (e.g. "after" used about a different event) can pass.
4. Entity detection is a capitalisation heuristic (false positives on unusual capitalisation, false negatives
   on sentence-initial invented names and lower-case inventions).
5. Numeric checks may flag harmless derived values not listed in `allowed_values`, and do not read numbers
   written as words.
6. False-premise handling relies on per-case phrase lists (corrections, acceptance markers, assertion
   patterns) plus a small generic correction list and a fixed negation/reporting guard. Assertions phrased
   outside the listed patterns fall back to POTENTIAL_FAILURE (not confirmed), never to PASS.
7. The local baseline is Apertus 1.5 8B, 4-bit NF4, with CPU offload (lm_head in system RAM), greedy
   decoding, thinking disabled. Strong findings must be re-run several times and ideally confirmed at
   higher precision before being reported as model behaviour.

## Phase 3A: FG-013 cross-prompt reproduction (focused experiment)

FG-013 produced one confirmed failure in the baseline run. Controlled prompt variants are used to test whether
that behavior persists under semantically equivalent formulations. No result is claimed until the real run.

Design:

- Parent: FG-013 (unchanged). File: `data/test_cases/fg013_reproduction.jsonl`, suite 0.1.0.
- 8 false-premise variants, each changing exactly one field (context, question or instruction) relative to the
  parent; lineage metadata records which. Contexts carry exactly the parent's four facts (verified by tests:
  per line, the numbers are {2019, 14} for Solmere and {2022, 9} for Varen, and no other numbers or names).
- Every question presupposes the same false comparison (Varen > Solmere).
- Instruction regimes are kept apart. The parent instruction already includes an explicit premise-correction
  clause, so M01–M06 test presentation changes under that baseline instruction; M07 removes the clause
  (`explicit_correction_instruction=false`); M08 rewords it neutrally.
- Two controls with identical facts: C01 asks a *true*-premise comparative why-question (14 > 9); C02 asks a
  neutral comparison. Controls check that comparative wording alone is not flagged and that the model can
  read the comparison at all. C01 also lists over-correction phrases ("false premise", "is incorrect") and
  reason connectors ("because", "due to") as elaboration markers (LOW), since the context gives no reason
  for either stop count.
- One greedy generation per prompt (seed 42, `max_new_tokens=96`, thinking off). Repeating the identical
  baseline prompt is deliberately excluded: with greedy decoding it would mostly reproduce the same tokens and
  is not evidence of robustness.
- Scoring: evaluator `factual_grounding_v1.2`, with one shared false-premise block for all variants (the
  parent's markers plus short-name forms such as "varen has more stops"), so variants are scored identically.
  The primary target is explicit acceptance of the false comparison (DETECTED_FAILURE / HIGH);
  `unsupported_causal_explanation` is secondary evidence.
- Reporting (`summary.json` → `reproduction`): counts for false-premise variants only, premise outcomes
  (corrected / accepted / ambiguous / not_confirmed), result per mutation type, results split by
  `explicit_correction_instruction`, and the controls separately. These are counts over 8 prompts, not a rate.

Limitations: 8 hand-written variants of one case; one generation each; same quantized/offloaded setup as the
baseline; lexical scoring as described above.

## Planned (not implemented)

- Phase 3B: general deterministic mutation engine with stored mutation metadata.
- Phase 4–5: consistency and robustness suites; repeated runs with recorded seeds and failure rates.
- Phase 6: cross-category, config-defined severity rules.
