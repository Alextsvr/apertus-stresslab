"""Run summaries: descriptive counts only.

Deliberately no aggregate "hallucination rate": a ~20-case synthetic suite with one run per
case does not support such a claim. Counts are reported per status, subtype and severity.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from stresslab.schemas import RunMetadata, Status, TestResult

FAILURE_STATUSES = {Status.POTENTIAL_FAILURE, Status.DETECTED_FAILURE}


def is_failure(result: TestResult) -> bool:
    return result.status in FAILURE_STATUSES


def build_summary(results: Iterable[TestResult], metadata: Optional[RunMetadata] = None) -> dict[str, Any]:
    results = list(results)
    status_counts = Counter(r.status.value for r in results)

    by_subtype: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_subtype.setdefault(r.subtype or "unspecified", {})
        bucket[r.status.value] = bucket.get(r.status.value, 0) + 1

    severity_counts = Counter(r.severity.value for r in results if r.severity is not None)
    evidence_types = Counter(e.get("type", "?") for r in results for e in r.evidence)

    model = metadata.model if metadata else None
    extra = model.extra if model else {}
    return {
        "run_id": metadata.run_id if metadata else (results[0].run_id if results else None),
        "suite": metadata.suite if metadata else None,
        "test_suite_version": metadata.test_suite_version if metadata else None,
        "evaluator": next((r.evaluator for r in results if r.evaluator), None),
        "model": model.model_id if model else (results[0].model if results else None),
        "revision": (model.resolved_revision or model.requested_revision) if model else None,
        "is_real_model": model.is_real_model if model else None,
        "dtype": model.dtype if model else None,
        "quantization": model.quantization if model else None,
        "execution": extra.get("execution") if extra else None,
        "seed": metadata.seed if metadata else None,
        "generation_config": metadata.generation_config.model_dump() if metadata else None,
        "total": len(results),
        "pass": status_counts.get(Status.PASS.value, 0),
        "potential_failure": status_counts.get(Status.POTENTIAL_FAILURE.value, 0),
        "detected_failure": status_counts.get(Status.DETECTED_FAILURE.value, 0),
        "unscored": status_counts.get(Status.UNSCORED.value, 0),
        "error": status_counts.get(Status.ERROR.value, 0),
        "by_status": dict(sorted(status_counts.items())),
        "by_subtype": dict(sorted(by_subtype.items())),
        "by_severity": dict(sorted(severity_counts.items())),
        "evidence_types": dict(sorted(evidence_types.items())),
        "failure_ids": [r.test_id for r in results if is_failure(r)],
        "truncated_responses": [r.test_id for r in results if r.usage.get("hit_max_new_tokens")],
        "note": "Descriptive counts from a single run per case on a small synthetic suite; "
                "not a hallucination rate. Statuses come from a conservative deterministic evaluator.",
    }
