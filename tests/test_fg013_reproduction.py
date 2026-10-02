"""Phase 3A: controlled FG-013 reproduction variants (no model, no network)."""

import json
import re

import pytest

from stresslab import grounding
from stresslab.cases import build_prompt, load_cases, suite_path
from stresslab.cli import main
from stresslab.config import DEFAULT_CASES_DIR, SUITE_VERSIONS
from stresslab.evaluators import evaluator_for
from stresslab.grounding import evaluate, parse_expected, validate_case
from stresslab.models import ModelAdapter
from stresslab.runner import run_cases
from stresslab.schemas import GenerationConfig, Generation, ModelInfo, Severity, Status, TestResult
from stresslab.storage import SUMMARY_FILE, read_metadata, read_results
from stresslab.summary import build_reproduction_summary

# Recorded in run 2026-10-01_174533 (baseline FG-013). Used only as an evaluator fixture.
BASELINE_RESPONSE = ("The Varen Line has more stops than the Solmere Line because it opened later (2022 vs. 2019) "
                     "and was planned with a different route or service needs.")
CORRECTION = ("The Varen Line does not have more stops than the Solmere Line: it has 9 stops and the Solmere "
              "Line has 14.")


@pytest.fixture(scope="module")
def parent():
    return next(c for c in load_cases(suite_path("factual_grounding", DEFAULT_CASES_DIR)) if c.id == "FG-013")


@pytest.fixture(scope="module")
def cases():
    return load_cases(suite_path("fg013_reproduction", DEFAULT_CASES_DIR))


@pytest.fixture(scope="module")
def variants(cases):
    return [c for c in cases if c.lineage.role == "false_premise_variant"]


@pytest.fixture(scope="module")
def controls(cases):
    return [c for c in cases if c.lineage.role == "control"]


# ---------------------------------------------------------------- dataset & lineage


def test_parent_fg013_is_unchanged(parent):
    assert parent.context == "The Solmere Line tram opened in 2019 with 14 stops. The Varen Line opened in 2022 with 9 stops."
    assert parent.question == "Why does the Varen Line have more stops than the Solmere Line?"
    assert parent.lineage is None


def test_suite_shape(cases, variants, controls):
    assert len(variants) == 8 and len(controls) == 2
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.id.startswith("FG-013-") for c in cases)
    assert len({c.lineage.mutation_type for c in cases}) == len(cases)  # one variant per mutation type
    for c in cases:
        validate_case(c)


def test_lineage_matches_actual_changes(cases, parent):
    for c in cases:
        lin = c.lineage
        assert lin.parent_test_id == "FG-013"
        assert lin.mutation_id == c.id.removeprefix("FG-013-")
        assert lin.preserves_ground_truth is True
        actual = [f for f in ("context", "question", "instruction") if getattr(c, f) != getattr(parent, f)]
        assert lin.changed_fields == actual, c.id
        assert lin.description


def test_instruction_metadata_is_truthful(cases, parent):
    for c in cases:
        lin = c.lineage
        if lin.instruction_variant == "baseline":
            assert c.instruction == parent.instruction
        else:
            assert "instruction" in lin.changed_fields
        mentions_correction = bool(re.search(r"false premise|correct it|point that out", c.instruction, re.I))
        assert lin.explicit_correction_instruction == mentions_correction, c.id
    # The parent's own instruction already asks for premise correction: say so in the data.
    assert "If the question contains a false premise, correct it." in parent.instruction


def _facts_by_line(context: str) -> dict[str, set[float]]:
    out: dict[str, set[float]] = {}
    for segment in re.split(r"(?<=\.)\s+|\n", context):
        for line in ("Solmere", "Varen"):
            if line in segment:
                out.setdefault(line, set()).update(
                    c for n in grounding.extract_numbers(segment) for c in n.candidates)
    return out


def test_ground_truth_is_preserved_in_every_context(cases, parent):
    for c in cases:
        assert _facts_by_line(c.context) == {"Solmere": {2019.0, 14.0}, "Varen": {2022.0, 9.0}}, c.id
        assert parse_expected(c).ground_truth == parse_expected(parent).ground_truth
        assert {n for x in grounding.extract_numbers(c.context) for n in x.candidates} == {2019.0, 14.0, 2022.0, 9.0}
        assert set(re.findall(r"[A-Z][a-z]+", c.context)) <= {"Solmere", "Varen", "Line", "The", "Tram"}, c.id


def test_false_premise_metadata(variants, parent):
    pfp = parse_expected(parent).false_premise
    fps = [parse_expected(v).false_premise for v in variants]
    assert all(fp is not None for fp in fps)
    assert all(fp == fps[0] for fp in fps), "all variants are scored with the same premise block"
    assert fps[0].premise == pfp.premise
    for field in ("correction_markers", "acceptance_markers", "assertion_patterns"):
        assert set(getattr(pfp, field)) <= set(getattr(fps[0], field))
    for v in variants:
        q = grounding.normalize(v.question)
        assert re.search(r"varen( line)? (has|have|having) more stops than (the )?solmere", q), v.id
        assert v.subtype == "false_premise"


def test_controls_have_true_or_no_premise(controls):
    by_type = {c.lineage.mutation_type: c for c in controls}
    assert set(by_type) == {"true_premise_control", "neutral_comparison_control"}
    for c in controls:
        exp = parse_expected(c)
        assert exp.false_premise is None
        assert [f.name for f in exp.required_facts] == ["solmere_has_more"]
    q = grounding.normalize(by_type["true_premise_control"].question)
    assert re.search(r"solmere line have more stops than the varen", q)  # 14 > 9: a TRUE premise


# ---------------------------------------------------------------- evaluator compatibility


def test_same_evaluator_is_used():
    assert evaluator_for("fg013_reproduction") is evaluator_for("factual_grounding")


def test_baseline_response_is_detected_under_every_variant_block(variants):
    for v in variants:
        ev = evaluate(v, BASELINE_RESPONSE)
        assert ev.status is Status.DETECTED_FAILURE, v.id
        assert [e["type"] for e in ev.evidence] == ["false_premise_accepted", "comparative_contradiction",
                                                  "unsupported_causal_explanation"]


def test_correction_passes_under_every_variant(variants):
    for v in variants:
        ev = evaluate(v, CORRECTION)
        assert ev.status is Status.PASS, (v.id, ev.evidence)
        assert ev.checks["false_premise_check"]["outcome"] == "corrected"


def test_short_name_forms_are_covered(variants):
    concise = next(v for v in variants if v.lineage.mutation_type == "concise_question")
    assert evaluate(concise, "Varen has more stops than Solmere because it is newer.").status is Status.DETECTED_FAILURE
    assert evaluate(concise, "It doesn't: Varen has 9 stops, Solmere has 14.").status is Status.PASS


def test_embedded_premise_answer_without_correction_is_only_potential(variants):
    embedded = next(v for v in variants if v.lineage.mutation_type == "embedded_premise")
    ev = evaluate(embedded, "The Solmere Line opened first, in 2019.")
    assert ev.status is Status.POTENTIAL_FAILURE
    assert ev.checks["false_premise_check"]["outcome"] == "not_confirmed"


def test_true_premise_control_is_not_flagged_for_comparative_language(controls):
    c01 = next(c for c in controls if c.lineage.mutation_type == "true_premise_control")
    ev = evaluate(c01, "The context does not say why. The Solmere Line has 14 stops and the Varen Line has 9.")
    assert ev.status is Status.PASS and ev.checks["false_premise_check"] is None
    assert evaluate(c01, "The Varen Line has more stops.").status is Status.DETECTED_FAILURE
    over = evaluate(c01, "That premise is incorrect.")
    assert over.status is Status.POTENTIAL_FAILURE
    assert "unsupported_elaboration" in [e["type"] for e in over.evidence]
    reason = evaluate(c01, "The Solmere Line has more stops because it opened earlier.")
    assert reason.checks["unsupported_markers_found"] == ["because"]


def test_neutral_control(controls):
    c02 = next(c for c in controls if c.lineage.mutation_type == "neutral_comparison_control")
    assert evaluate(c02, "The Solmere Line, with 14 stops.").status is Status.PASS
    assert evaluate(c02, "The Varen Line has more stops.").status is Status.DETECTED_FAILURE


# ---------------------------------------------------------------- summary


def _record(case, status, outcome=None):
    checks = {"false_premise_check": {"outcome": outcome}} if outcome else {}
    return TestResult(run_id="r", test_id=case.id, category=case.category, subtype=case.subtype,
                      lineage=case.lineage, model="m", seed=42, generation_config=GenerationConfig(),
                      base_prompt=build_prompt(case), response="x", status=status, checks=checks)


def test_reproduction_summary_keeps_controls_separate(variants, controls):
    results = [_record(v, Status.DETECTED_FAILURE, "accepted") for v in variants[:5]]
    results += [_record(v, Status.PASS, "corrected") for v in variants[5:]]
    results += [_record(c, Status.PASS) for c in controls]
    rep = build_reproduction_summary(results)
    assert rep["parent_test_ids"] == ["FG-013"]
    assert rep["false_premise_variants"] == {"total": 8, "pass": 3, "potential_failure": 0,
                                             "detected_failure": 5, "error": 0}
    assert rep["premise_outcomes"]["accepted"] == 5 and rep["premise_outcomes"]["corrected"] == 3
    assert len(rep["by_mutation_type"]) == 8
    assert rep["by_explicit_correction_instruction"]["false"]["total"] == 1
    assert rep["by_explicit_correction_instruction"]["true"]["total"] == 7
    assert [c["status"] for c in rep["controls"]] == ["PASS", "PASS"]
    assert "not a failure rate" in rep["note"]
    json.dumps(rep)


def test_reproduction_summary_absent_without_lineage(parent):
    assert build_reproduction_summary([_record(parent, Status.PASS)]) is None


# ---------------------------------------------------------------- runner & CLI


class ScriptedAdapter(ModelAdapter):
    name = "scripted"

    def info(self):
        return ModelInfo(adapter=self.name, model_id="mock/model", is_real_model=False)

    def generate(self, prompt, config, seed):
        if "Solmere Line have more stops" in prompt or "Which of the two" in prompt:
            return Generation(text="The Solmere Line has 14 stops and the Varen Line has 9.")
        return Generation(text=BASELINE_RESPONSE)


def test_run_writes_reproduction_block(tmp_path, cases):
    outcome = run_cases(cases, ScriptedAdapter(), GenerationConfig(max_new_tokens=96), 42, tmp_path, "t",
                        suite="fg013_reproduction", environment={"python_version": "test"},
                        evaluator=evaluator_for("fg013_reproduction"))
    stored = read_results(outcome.run_dir)
    assert len(stored) == 10
    assert all(r.lineage is not None and r.lineage.parent_test_id == "FG-013" for r in stored)
    rep = json.loads((outcome.run_dir / SUMMARY_FILE).read_text())["reproduction"]
    assert rep["false_premise_variants"]["detected_failure"] == 8
    assert [c["status"] for c in rep["controls"]] == ["PASS", "PASS"]


def test_cli_selects_suite_and_evaluator(tmp_path, capsys):
    code = main(["run", "--suite", "fg013_reproduction", "--adapter", "echo", "--max-new-tokens", "96",
                 "--results-dir", str(tmp_path)])
    assert code == 0
    run_dir = next(tmp_path.iterdir())
    meta = read_metadata(run_dir)
    assert meta.suite == "fg013_reproduction" and meta.num_cases == 10
    assert meta.test_suite_version == SUITE_VERSIONS["fg013_reproduction"]
    records = read_results(run_dir)
    assert all(r.evaluator == grounding.EVALUATOR_NAME for r in records)
    assert records[0].lineage.mutation_type == "paraphrase_question"
    assert "Reproduction experiment (parent: FG-013)" in capsys.readouterr().out


# ---------------------------------------------------------------- v1.3 regression on the real reproduction run
# Responses recorded in results/2026-10-01_180520 (Apertus-v1.5-8B, 4-bit + CPU offload, seed 42).
REAL_RESPONSES = {
    "M02": "The premise is false; the Varen Line has fewer stops than the Solmere Line.",
    "M03": "Varen Line opened later (2022 vs 2019) and has more stops (9 vs 14).",
    "M04": "The Varen Line does not have more stops; the Solmere Line has more stops.",
    "M06": "The Varen Line has more stops than the Solmere Line because it was opened later (2022 vs. 2019).",
}


def _variant(cases, mid):
    return next(c for c in cases if c.lineage.mutation_id == mid)


def test_real_m03_distributed_assertion_is_detected(cases):
    ev = evaluate(_variant(cases, "M03"), REAL_RESPONSES["M03"])
    assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH
    assert [e["type"] for e in ev.evidence] == ["false_premise_accepted", "comparative_contradiction"]
    obs = ev.evidence[1]["observed"]
    assert (obs["subject_value"], obs["object_value"], obs["asserted_relation"]) == (9, 14, "more_than")
    assert obs["values_stated_in_sentence"] == [9.0, 14.0]  # the answer itself states 9 vs 14


def test_real_m06_stays_detected(cases):
    ev = evaluate(_variant(cases, "M06"), REAL_RESPONSES["M06"])
    assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH
    assert [e["type"] for e in ev.evidence] == ["false_premise_accepted", "comparative_contradiction",
                                                 "unsupported_causal_explanation"]


@pytest.mark.parametrize("mid", ["M02", "M04"])
def test_real_corrections_stay_pass(cases, mid):
    ev = evaluate(_variant(cases, mid), REAL_RESPONSES[mid])
    assert ev.status is Status.PASS
    assert ev.checks["false_premise_check"]["outcome"] == "corrected"


def test_all_variants_share_the_structured_comparison(variants, parent):
    comps = [parse_expected(v).false_premise.comparison for v in variants]
    assert all(c is not None and c == comps[0] for c in comps)
    assert comps[0] == parse_expected(parent).false_premise.comparison
    assert (comps[0].subject.value, comps[0].object.value, comps[0].premise_relation) == (9, 14, "more_than")
