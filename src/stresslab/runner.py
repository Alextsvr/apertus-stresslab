"""Runner: executes test cases against an adapter and persists every record.

One model call per case. Without an evaluator (e.g. smoke) records are stored UNSCORED,
exactly as in Phase 1. With an evaluator (Phase 2 factual grounding) each response is scored
right after generation. Adapter exceptions are stored with status ERROR and the run continues.
After the run, summary.json (and, for evaluated runs, failures.jsonl) are written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from stresslab import __version__
from stresslab.cases import build_prompt
from stresslab.config import SUITE_VERSIONS
from stresslab.environment import collect_environment
from stresslab.evaluators import Evaluator
from stresslab.models import ModelAdapter
from stresslab.schemas import (
    GenerationConfig,
    Reproducibility,
    RunMetadata,
    Status,
    TestCase,
    TestResult,
    utc_now_iso,
)
from stresslab.storage import (
    append_result,
    create_run_dir,
    new_run_id,
    read_metadata,
    read_results,
    write_failures,
    write_metadata,
    write_summary,
)
from stresslab.summary import build_summary, is_failure


@dataclass
class RunOutcome:
    run_dir: Path
    results: list[TestResult]

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == Status.ERROR)


def run_cases(
    cases: Iterable[TestCase],
    adapter: ModelAdapter,
    gen_config: GenerationConfig,
    seed: int,
    results_root: Path,
    command: str,
    suite: Optional[str] = None,
    test_suite_version: Optional[str] = None,
    environment: Optional[dict] = None,
    notes: Optional[str] = None,
    evaluator: Optional[Evaluator] = None,
) -> RunOutcome:
    cases = list(cases)
    run_dir = create_run_dir(results_root)
    run_id = run_dir.name

    metadata = RunMetadata(
        run_id=run_id,
        command=command,
        suite=suite,
        test_suite_version=test_suite_version,
        num_cases=len(cases),
        seed=seed,
        generation_config=gen_config,
        environment=environment if environment is not None else collect_environment(),
        stresslab_version=__version__,
        notes=notes,
    )
    # Write early so that even a crash during model loading leaves a trace.
    write_metadata(run_dir, metadata)

    try:
        adapter.load()
    except Exception as exc:
        metadata.notes = f"Model load failed: {type(exc).__name__}: {exc}"
        try:  # keep whatever the adapter knows (e.g. the planned device map) as evidence
            metadata.model = adapter.info()
        except Exception:  # noqa: BLE001
            pass
        metadata.finished_at = utc_now_iso()
        write_metadata(run_dir, metadata)
        raise
    model_info = adapter.info()
    metadata.model = model_info
    write_metadata(run_dir, metadata)

    results: list[TestResult] = []
    for case in cases:
        prompt = build_prompt(case)
        record = TestResult(
            run_id=run_id,
            test_id=case.id,
            category=case.category,
            subtype=case.subtype,
            lineage=case.lineage,
            model=model_info.model_id,
            model_revision=model_info.resolved_revision or model_info.requested_revision,
            seed=seed,
            generation_config=gen_config,
            base_prompt=prompt,
            expected_behavior=case.expected,
            response="",
            reproducibility=Reproducibility(runs=1, failures=0, rate=0.0, seeds=[seed]),
            notes=None if model_info.is_real_model else "NOT A REAL MODEL: echo adapter output.",
        )
        try:
            gen = adapter.generate(prompt, gen_config, seed)
            record.response = gen.text
            record.raw_response = gen.raw_text
            record.usage = {
                "input_tokens": gen.input_tokens,
                "output_tokens": gen.output_tokens,
                "latency_s": gen.latency_s,
                "peak_gpu_memory_gib": gen.peak_gpu_memory_gib,
                "hit_max_new_tokens": gen.output_tokens is not None
                and gen.output_tokens >= gen_config.max_new_tokens,
            }
        except Exception as exc:  # noqa: BLE001 - recorded as evidence, run continues
            record.status = Status.ERROR
            record.error = f"{type(exc).__name__}: {exc}"
        if evaluator is not None and record.status != Status.ERROR:
            apply_evaluation(record, case, evaluator)
        record.timestamp = utc_now_iso()
        append_result(run_dir, record)
        results.append(record)

    metadata.finished_at = utc_now_iso()
    write_metadata(run_dir, metadata)
    write_summary(run_dir, build_summary(results, metadata))
    if evaluator is not None:
        write_failures(run_dir, [r for r in results if is_failure(r)])
    return RunOutcome(run_dir=run_dir, results=results)


def apply_evaluation(record: TestResult, case: TestCase, evaluator: Evaluator) -> TestResult:
    """Score one response in place. Single run per case: reproducibility stays 1 run."""
    ev = evaluator.evaluate(case, record.response)
    record.evaluator = ev.evaluator
    record.status = ev.status
    record.severity = ev.severity
    record.checks = ev.checks
    record.evidence = ev.evidence
    failed = int(is_failure(record))
    record.reproducibility.failures = failed
    record.reproducibility.rate = float(failed)
    return record


def single_prompt_case(prompt: str, test_id: str = "ADHOC-001") -> TestCase:
    """Wrap a free-form prompt (Phase 0 `infer`) as a test case so it uses the same pipeline."""
    return TestCase(id=test_id, category="smoke", question=prompt, tags=["adhoc"])


def rescore_run(run_dir: Path, cases: Iterable[TestCase], evaluator: Evaluator) -> tuple[Path, list[TestResult]]:
    """Re-evaluate the stored responses of an existing run with the current evaluator.

    No model call. The canonical results.jsonl of the run is never modified; the rescored
    records go to <run_dir>/rescored_<timestamp>/. Records whose stored prompt no longer
    matches the current case file are flagged in `notes` instead of being silently mixed.
    """
    by_id = {c.id: c for c in cases}
    metadata = read_metadata(run_dir)
    out_dir = create_run_dir(Path(run_dir), f"rescored_{new_run_id()}")
    rescored: list[TestResult] = []
    original_status: dict[str, str] = {}
    for record in read_results(run_dir):
        original_status[record.test_id] = record.status.value
        record = record.model_copy(deep=True)
        case = by_id.get(record.test_id)
        if case is None:
            record.notes = (record.notes or "") + " [rescore: test case no longer exists]"
        elif record.status != Status.ERROR:
            if build_prompt(case) != record.base_prompt:
                record.notes = (record.notes or "") + " [rescore: case prompt changed since the run]"
            apply_evaluation(record, case, evaluator)
        append_result(out_dir, record)
        rescored.append(record)
    summary = build_summary(rescored, metadata)
    summary["rescored_from"] = str(run_dir)
    summary["rescored_at"] = utc_now_iso()
    summary["rescore_evaluator"] = evaluator.name
    summary["rescore_test_suite_version"] = SUITE_VERSIONS.get(metadata.suite or "")
    summary["original_status_by_test"] = original_status
    summary["status_changes"] = {
        r.test_id: {"from": original_status.get(r.test_id), "to": r.status.value}
        for r in rescored if original_status.get(r.test_id) != r.status.value
    }
    write_summary(out_dir, summary)
    write_failures(out_dir, [r for r in rescored if is_failure(r)])
    return out_dir, rescored
