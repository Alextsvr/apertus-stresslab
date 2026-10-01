"""Basic runner: executes prompts/test cases against an adapter and persists every record.

Phase 1 scope: no evaluation. Every successful record is stored with status UNSCORED;
adapter exceptions are stored with status ERROR (and the run continues).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from stresslab import __version__
from stresslab.cases import build_prompt
from stresslab.environment import collect_environment
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
from stresslab.storage import append_result, create_run_dir, write_metadata


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
        record.timestamp = utc_now_iso()
        append_result(run_dir, record)
        results.append(record)

    metadata.finished_at = utc_now_iso()
    write_metadata(run_dir, metadata)
    return RunOutcome(run_dir=run_dir, results=results)


def single_prompt_case(prompt: str, test_id: str = "ADHOC-001") -> TestCase:
    """Wrap a free-form prompt (Phase 0 `infer`) as a test case so it uses the same pipeline."""
    return TestCase(id=test_id, category="smoke", question=prompt, tags=["adhoc"])
