"""Loading test cases and turning them into prompts."""

from __future__ import annotations

import json
from pathlib import Path

from stresslab.config import DEFAULT_CASES_DIR, GROUNDED_PROMPT_TEMPLATE
from stresslab.schemas import TestCase


def load_cases(path: Path) -> list[TestCase]:
    """Load a JSONL file of test cases. Fails loudly on malformed lines or duplicate ids."""
    path = Path(path)
    cases: list[TestCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                case = TestCase.model_validate(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - re-raise with location
                raise ValueError(f"{path}:{lineno}: invalid test case: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate test id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


def suite_path(suite: str, cases_dir: Path = DEFAULT_CASES_DIR) -> Path:
    path = Path(cases_dir) / f"{suite}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in Path(cases_dir).glob("*.jsonl"))
        raise FileNotFoundError(f"Unknown suite {suite!r}. Available: {available}")
    return path


def build_prompt(case: TestCase) -> str:
    """Deterministic prompt construction. No context -> the bare question."""
    if case.context:
        return GROUNDED_PROMPT_TEMPLATE.format(context=case.context.strip(), question=case.question.strip())
    return case.question.strip()
