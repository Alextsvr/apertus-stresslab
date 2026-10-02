"""Evaluator registry: which deterministic evaluator applies to which suite.

The evaluator is chosen from the suite name, so `run --suite factual_grounding` is scored
automatically and `run --suite smoke` stays UNSCORED exactly as in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from stresslab import controls, grounding
from stresslab.grounding import Evaluation
from stresslab.schemas import TestCase


@dataclass(frozen=True)
class Evaluator:
    name: str
    evaluate: Callable[[TestCase, str], Evaluation]
    validate: Callable[[TestCase], None]


_GROUNDING = Evaluator(grounding.EVALUATOR_NAME, grounding.evaluate, grounding.validate_case)

EVALUATORS: dict[str, Evaluator] = {
    "factual_grounding": _GROUNDING,
    # Phase 3A: controlled variants of FG-013 are scored by the same evaluator (no second evaluator).
    "fg013_reproduction": _GROUNDING,
    # Phase 3B: pre-registered held-out validation. Same frozen evaluator for every case; neutral controls
    # additionally get the explicit entity-answer check (stresslab.controls, neutral_control_v1).
    "false_premise_heldout": Evaluator(f"{grounding.EVALUATOR_NAME} (+{controls.CONTROL_SCORER} for controls)",
                                       controls.evaluate_heldout, grounding.validate_case),
}


def evaluator_for(name: Optional[str]) -> Optional[Evaluator]:
    """`name` is a suite name / evaluator key; unknown or None -> no evaluator (UNSCORED)."""
    if not name:
        return None
    return EVALUATORS.get(name)
