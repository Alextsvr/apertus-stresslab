"""Neutral-control answer check for the Phase 3B held-out suite (`neutral_control_v1`).

The frozen evaluator `factual_grounding_v1.3` (grounding.py, unchanged) scores every held-out case. For
neutral controls ("Which of the two ... has more/fewer M?") its required-fact phrases are relational
("<winner> has more M", "more M than <loser>"), so a bare entity answer such as "Norvex Systems" — a perfectly
valid answer to a "which" question — would be reported as a missing fact. This module adds one narrowly
scoped, explicit rule on top of the frozen result, for controls only:

* The WHOLE answer (after removing markdown, quotes, a leading "the"/"answer:"/"it is", trailing punctuation
  and parentheticals that name neither entity) must equal an alias of `heldout.expected_answer` -> the
  missing-fact evidence is withdrawn (PASS unless other evidence remains), or an alias of
  `heldout.wrong_answer` -> a high-confidence `contradiction` replaces it (DETECTED_FAILURE).
  An entity merely mentioned inside a longer answer never counts: such answers keep the frozen result.
* If an answer states both the correct and the reversed comparison, the frozen evaluator would PASS it
  (a correct phrase is present); here it becomes POTENTIAL_FAILURE (`control_answer_ambiguous`).

False-premise variants are returned exactly as scored by the frozen evaluator.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from stresslab import grounding
from stresslab.grounding import Evaluation
from stresslab.schemas import HeldOut, Severity, Status, TestCase

CONTROL_SCORER = "neutral_control_v1"
_LEADING_FILLERS = ("the answer is ", "answer ", "it is ", "it s ", "that is ", "the ")
_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]


def _aliases(names: list[str]) -> set[str]:
    out = set()
    for n in names:
        a = grounding.normalize(n)
        out.add(a)
        out.add(a.removeprefix("the "))
    return out


def entity_only_answer(response: str, h: HeldOut) -> Optional[str]:
    """'correct' / 'wrong' if the whole answer is just one of the two entities, else None."""
    correct, wrong = _aliases(h.expected_answer_aliases), _aliases(h.wrong_answer_aliases)
    text = re.sub(r"[*_`\"“”]", " ", response)

    def drop_paren(m: re.Match) -> str:
        inner = grounding.normalize(m.group(1))
        mentions = any(grounding.contains_phrase(inner, a) for a in correct | wrong)
        return m.group(0) if mentions else " "

    text = re.sub(r"\(([^()]*)\)", drop_paren, text)
    norm = grounding.normalize(text)
    changed = True
    while changed:
        changed = False
        for filler in _LEADING_FILLERS:
            if norm.startswith(filler):
                norm, changed = norm[len(filler):].strip(), True
    if norm in correct:
        return "correct"
    if norm in wrong:
        return "wrong"
    return None


def _item(kind: str, detail: str, confidence: str, severity: Severity, expected: Any, observed: Any) -> dict:
    return {"type": kind, "detail": detail, "expected": expected, "observed": observed,
            "confidence": confidence, "severity": severity.value}


def score_neutral_control(case: TestCase, response: str, base: Evaluation) -> Evaluation:
    h = case.heldout
    fact = next((c for c in base.checks.get("required_fact_checks", []) if c["name"] == "correct_comparison"), None)
    kind = entity_only_answer(response, h)
    evidence = [e for e in base.evidence]
    missing = [e for e in evidence if e["type"] == "missing_required_fact"]

    if kind == "correct":
        evidence = [e for e in evidence if e not in missing]
        outcome = "entity_only_correct"
    elif kind == "wrong":
        evidence = [e for e in evidence if e not in missing]
        evidence.insert(0, _item("contradiction", "neutral control answered with the wrong entity", "high",
                                 Severity.HIGH, h.expected_answer, h.wrong_answer))
        outcome = "entity_only_wrong"
    elif fact and fact["found"] and fact["conflicts_found"]:
        evidence.append(_item("control_answer_ambiguous",
                              "answer states both the correct and the reversed comparison", "medium",
                              Severity.MEDIUM, fact["matched"], fact["conflicts_found"]))
        outcome = "ambiguous"
    elif fact and fact["found"]:
        outcome = "relational_correct"
    elif base.checks.get("contradictions"):
        outcome = "relational_reversed"
    else:
        outcome = "not_answered"

    if any(e["confidence"] == "high" for e in evidence):
        status = Status.DETECTED_FAILURE
    elif evidence:
        status = Status.POTENTIAL_FAILURE
    else:
        status = Status.PASS
    severity = max((Severity(e["severity"]) for e in evidence), key=_SEVERITY_ORDER.index) if evidence else None

    checks = dict(base.checks)
    checks["neutral_control_check"] = {
        "scorer": CONTROL_SCORER, "asked_relation": h.asked_relation, "expected_answer": h.expected_answer,
        "wrong_answer": h.wrong_answer, "entity_only_answer": kind, "outcome": outcome,
        "frozen_evaluator_status": base.status.value,
    }
    return Evaluation(f"{base.evaluator}+{CONTROL_SCORER}", status, severity, checks, evidence)


def evaluate_heldout(case: TestCase, response: str) -> Evaluation:
    """Frozen v1.3 for every case; neutral controls additionally get the explicit answer check."""
    base = grounding.evaluate(case, response)
    if case.heldout is not None and case.heldout.control:
        return score_neutral_control(case, response, base)
    return base
