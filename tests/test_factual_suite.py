"""Phase 2 dataset, runner integration, summary, CLI and serialization (no model, no network)."""

import json
from collections import Counter

import pytest

from stresslab.cases import build_prompt, load_cases, suite_path
from stresslab.cli import main
from stresslab.config import DEFAULT_CASES_DIR, GROUNDING_INSTRUCTION
from stresslab.evaluators import evaluator_for
from stresslab.grounding import contains_value, evaluate, extract_numbers, normalize, parse_expected, validate_case
from stresslab.models import ModelAdapter
from stresslab.runner import rescore_run, run_cases
from stresslab.schemas import GenerationConfig, Generation, ModelInfo, Severity, Status, TestResult
from stresslab.storage import FAILURES_FILE, SUMMARY_FILE, read_metadata, read_results
from stresslab.summary import build_summary

FAKE_ENV = {"python_version": "test"}
EXPECTED_SUBTYPES = {
    "direct_extraction", "multi_fact", "numeric", "relationship", "negative_fact", "false_premise",
    "distractor", "similar_entity", "timeline", "unsupported_elaboration",
}


@pytest.fixture(scope="module")
def fg_cases():
    return load_cases(suite_path("factual_grounding", DEFAULT_CASES_DIR))


# ---------------------------------------------------------------- dataset


def test_suite_size_and_ids(fg_cases):
    assert 18 <= len(fg_cases) <= 25
    ids = [c.id for c in fg_cases]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("FG-") for i in ids)


def test_every_case_validates_and_is_machine_checkable(fg_cases):
    for case in fg_cases:
        validate_case(case)
        exp = parse_expected(case)
        assert exp.required_facts or exp.false_premise, case.id
        assert exp.ground_truth, f"{case.id}: ground_truth must document the controlled facts"
        assert case.instruction == GROUNDING_INSTRUCTION


def test_subtype_coverage(fg_cases):
    counts = Counter(c.subtype for c in fg_cases)
    assert set(counts) == EXPECTED_SUBTYPES
    assert all(n >= 2 for n in counts.values())


def test_false_premise_cases_are_marked(fg_cases):
    for case in fg_cases:
        has_premise = parse_expected(case).false_premise is not None
        assert has_premise == (case.subtype == "false_premise"), case.id


@pytest.mark.parametrize("subtype", ["direct_extraction", "relationship", "distractor", "similar_entity"])
def test_extractive_answers_are_literally_in_context(fg_cases, subtype):
    """Ground truth sanity: for extractive subtypes, each required fact is stated in the context."""
    for case in (c for c in fg_cases if c.subtype == subtype):
        ctx = normalize(case.context)
        for fact in parse_expected(case).required_facts:
            assert any(contains_value(ctx, extract_numbers(case.context), v) for v in fact.values), (case.id, fact.name)


def test_reference_answers_pass_and_conflicts_are_detected(fg_cases):
    """A response made of the first accepted value per fact passes; swapping in a conflict is detected."""
    for case in fg_cases:
        exp = parse_expected(case)
        if exp.false_premise:
            continue
        good = ". ".join(f.values[0] for f in exp.required_facts) + "."
        assert evaluate(case, good).status is Status.PASS, (case.id, good)
        for fact in exp.required_facts:
            if fact.conflicts:
                others = [f.values[0] for f in exp.required_facts if f is not fact]
                bad = ". ".join(others + [fact.conflicts[0]]) + "."
                assert evaluate(case, bad).status is Status.DETECTED_FAILURE, (case.id, bad)


def test_instructed_prompt_format_and_smoke_prompt_unchanged(fg_cases):
    prompt = build_prompt(fg_cases[0])
    assert prompt.startswith(GROUNDING_INSTRUCTION + "\n\nContext:\n")
    assert "\n\nQuestion:\n" in prompt
    smoke = load_cases(suite_path("smoke", DEFAULT_CASES_DIR))
    assert build_prompt(smoke[1]).startswith("Answer the question using only the information in the context.")


def test_smoke_suite_untouched():
    smoke = load_cases(suite_path("smoke", DEFAULT_CASES_DIR))
    assert [c.id for c in smoke] == ["SMOKE-001", "SMOKE-002", "SMOKE-003", "SMOKE-004"]
    assert evaluator_for("smoke") is None


# ---------------------------------------------------------------- runner integration


class ScriptedAdapter(ModelAdapter):
    """Mock adapter returning a fixed response per test id (by matching the question text)."""

    name = "scripted"

    def __init__(self, by_question: dict[str, str]):
        self.by_question = by_question
        self.calls = 0

    def info(self):
        return ModelInfo(adapter=self.name, model_id="mock/model", resolved_revision="rev1", is_real_model=False,
                         quantization="4bit", extra={"execution": "test"})

    def generate(self, prompt, config, seed):
        self.calls += 1
        for question, answer in self.by_question.items():
            if prompt.endswith(question):
                return Generation(text=answer, output_tokens=5)
        return Generation(text="I do not know.", output_tokens=5)


def _scripted(cases):
    by_id = {c.id: c for c in cases}
    return by_id, ScriptedAdapter({
        by_id["FG-001"].question: "Nora Veldt was born in Varen Port.",                  # PASS
        by_id["FG-016"].question: "The main span is 860 metres.",                         # DETECTED (conflict)
        by_id["FG-022"].question: "Brisk Hollow, and it has 120 employees.",              # POTENTIAL
    })


def test_evaluated_run_writes_scored_results_summary_and_failures(tmp_path, fg_cases):
    by_id, adapter = _scripted(fg_cases)
    cases = [by_id["FG-001"], by_id["FG-016"], by_id["FG-022"]]
    outcome = run_cases(cases, adapter, GenerationConfig(max_new_tokens=96), 42, tmp_path, "test",
                        suite="factual_grounding", environment=FAKE_ENV, evaluator=evaluator_for("factual_grounding"))
    assert adapter.calls == 3  # exactly one model call per case

    stored = {r.test_id: r for r in read_results(outcome.run_dir)}
    assert stored["FG-001"].status is Status.PASS and stored["FG-001"].severity is None
    assert stored["FG-016"].status is Status.DETECTED_FAILURE and stored["FG-016"].severity is Severity.HIGH
    assert stored["FG-022"].status is Status.POTENTIAL_FAILURE
    assert stored["FG-016"].subtype == "distractor"
    assert stored["FG-016"].evaluator == "factual_grounding_v1"
    assert stored["FG-016"].evidence[0]["type"] == "contradiction"
    assert stored["FG-016"].reproducibility.model_dump() == {"runs": 1, "failures": 1, "rate": 1.0, "seeds": [42]}

    summary = json.loads((outcome.run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["total"] == 3
    assert (summary["pass"], summary["potential_failure"], summary["detected_failure"]) == (1, 1, 1)
    assert summary["by_subtype"]["distractor"] == {"DETECTED_FAILURE": 1}
    assert summary["by_severity"]["HIGH"] == 1
    assert summary["model"] == "mock/model" and summary["revision"] == "rev1"
    assert summary["quantization"] == "4bit"
    assert "hallucination_rate" not in summary

    failures = [json.loads(line) for line in (outcome.run_dir / FAILURES_FILE).read_text().splitlines()]
    assert sorted(f["test_id"] for f in failures) == ["FG-016", "FG-022"]


def test_error_records_are_not_evaluated(tmp_path, fg_cases):
    class Broken(ScriptedAdapter):
        def generate(self, prompt, config, seed):
            raise RuntimeError("boom")

    outcome = run_cases(fg_cases[:1], Broken({}), GenerationConfig(), 1, tmp_path, "t",
                        environment=FAKE_ENV, evaluator=evaluator_for("factual_grounding"))
    record = read_results(outcome.run_dir)[0]
    assert record.status is Status.ERROR and record.evidence == [] and record.evaluator is None


def test_unevaluated_run_stays_unscored_and_has_no_failures_file(tmp_path, fg_cases):
    _, adapter = _scripted(fg_cases)
    outcome = run_cases(fg_cases[:2], adapter, GenerationConfig(), 1, tmp_path, "t", environment=FAKE_ENV)
    assert all(r.status is Status.UNSCORED for r in read_results(outcome.run_dir))
    assert (outcome.run_dir / SUMMARY_FILE).exists()
    assert not (outcome.run_dir / FAILURES_FILE).exists()


def test_rescore_writes_new_dir_and_keeps_canonical_results(tmp_path, fg_cases):
    by_id, adapter = _scripted(fg_cases)
    cases = [by_id["FG-001"], by_id["FG-016"]]
    outcome = run_cases(cases, adapter, GenerationConfig(), 42, tmp_path, "t", suite="factual_grounding",
                        environment=FAKE_ENV)  # unscored run
    original = (outcome.run_dir / "results.jsonl").read_bytes()
    out_dir, rescored = rescore_run(outcome.run_dir, fg_cases, evaluator_for("factual_grounding"))
    assert (outcome.run_dir / "results.jsonl").read_bytes() == original
    assert out_dir.parent == outcome.run_dir and out_dir.name.startswith("rescored_")
    assert [r.status for r in rescored] == [Status.PASS, Status.DETECTED_FAILURE]
    summary = json.loads((out_dir / SUMMARY_FILE).read_text())
    assert summary["rescored_from"] == str(outcome.run_dir)


# ---------------------------------------------------------------- summary & serialization


def _result(test_id, status, severity=None, subtype="numeric"):
    return TestResult(run_id="r", test_id=test_id, category="factual_grounding", subtype=subtype, model="m",
                      seed=1, generation_config=GenerationConfig(), base_prompt="p", response="x",
                      status=status, severity=severity)


def test_build_summary_counts():
    results = [
        _result("A", Status.PASS),
        _result("B", Status.POTENTIAL_FAILURE, Severity.MEDIUM),
        _result("C", Status.DETECTED_FAILURE, Severity.HIGH, subtype="timeline"),
        _result("D", Status.ERROR),
    ]
    s = build_summary(results)
    assert (s["total"], s["pass"], s["potential_failure"], s["detected_failure"], s["error"]) == (4, 1, 1, 1, 1)
    assert s["by_subtype"] == {"numeric": {"ERROR": 1, "PASS": 1, "POTENTIAL_FAILURE": 1},
                               "timeline": {"DETECTED_FAILURE": 1}}
    assert s["by_severity"] == {"HIGH": 1, "MEDIUM": 1}
    assert s["failure_ids"] == ["B", "C"]
    json.dumps(s)  # serialisable


def test_evaluated_result_roundtrip(fg_cases):
    ev = evaluate(fg_cases[0], "Nora Veldt was born in Zurich.")
    r = _result("FG-001", ev.status, ev.severity)
    r.checks, r.evidence, r.evaluator = ev.checks, ev.evidence, ev.evaluator
    again = TestResult.model_validate_json(r.model_dump_json())
    assert again == r and again.status is Status.POTENTIAL_FAILURE


# ---------------------------------------------------------------- CLI


def test_cli_run_factual_grounding_selects_evaluator(tmp_path, capsys):
    code = main(["run", "--suite", "factual_grounding", "--adapter", "echo", "--only", "FG-001", "FG-012",
                 "--max-new-tokens", "96", "--results-dir", str(tmp_path)])
    assert code == 0
    run_dir = next(tmp_path.iterdir())
    records = read_results(run_dir)
    assert [r.test_id for r in records] == ["FG-001", "FG-012"]
    assert all(r.evaluator == "factual_grounding_v1" for r in records)
    assert all(r.status is not Status.UNSCORED for r in records)
    meta = read_metadata(run_dir)
    assert meta.suite == "factual_grounding" and meta.test_suite_version == "0.2.0"
    assert meta.generation_config.max_new_tokens == 96
    assert (run_dir / SUMMARY_FILE).exists() and (run_dir / FAILURES_FILE).exists()
    assert "Status counts:" in capsys.readouterr().out


def test_cli_evaluator_none_keeps_unscored(tmp_path):
    main(["run", "--suite", "factual_grounding", "--adapter", "echo", "--limit", "1", "--evaluator", "none",
          "--results-dir", str(tmp_path)])
    assert read_results(next(tmp_path.iterdir()))[0].status is Status.UNSCORED


def test_cli_smoke_still_unscored(tmp_path):
    assert main(["run", "--suite", "smoke", "--adapter", "echo", "--results-dir", str(tmp_path)]) == 0
    assert {r.status for r in read_results(next(tmp_path.iterdir()))} == {Status.UNSCORED}


def test_cli_unknown_only_id_is_an_error(tmp_path, capsys):
    code = main(["run", "--suite", "factual_grounding", "--adapter", "echo", "--only", "FG-999",
                 "--results-dir", str(tmp_path)])
    assert code == 2 and "FG-999" in capsys.readouterr().err


def test_cli_show_failures_only_and_evaluate(tmp_path, capsys):
    main(["run", "--suite", "factual_grounding", "--adapter", "echo", "--limit", "2", "--results-dir", str(tmp_path)])
    run_dir = next(tmp_path.iterdir())
    capsys.readouterr()
    assert main(["show", str(run_dir), "--failures-only"]) == 0
    assert "Status counts:" in capsys.readouterr().out
    assert main(["evaluate", str(run_dir)]) == 0
    assert "original results.jsonl unchanged" in capsys.readouterr().out
    assert any(p.name.startswith("rescored_") for p in run_dir.iterdir())
