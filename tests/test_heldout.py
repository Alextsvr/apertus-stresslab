"""Phase 3B: pre-registered held-out false-premise suite (no model, no GPU, no network)."""

import json
import re
from collections import Counter

import pytest

from stresslab import grounding
from stresslab.cases import build_prompt, load_cases, suite_path
from stresslab.cli import main
from stresslab.config import DEFAULT_CASES_DIR, GROUNDING_INSTRUCTION, SUITE_VERSIONS
from stresslab.evaluators import evaluator_for
from stresslab.grounding import evaluate, parse_expected, validate_case
from stresslab.controls import CONTROL_SCORER, entity_only_answer
from stresslab.heldout import DEVELOPMENT_ENTITIES, assert_valid, validate_heldout
from stresslab.models import ModelAdapter
from stresslab.runner import run_cases
from stresslab.schemas import GenerationConfig, Generation, HeldOut, ModelInfo, Severity, Status, TestResult
from stresslab.storage import SUMMARY_FILE, read_metadata, read_results
from stresslab.summary import build_heldout_summary

WORD = {"more_than": "more", "less_than": "fewer"}

# Evaluator v1.3 is frozen for Phase 3B: any change to grounding.py must be deliberate and reviewed.
FROZEN_EVALUATOR = "factual_grounding_v1.3"


@pytest.fixture(scope="module")
def cases():
    return load_cases(suite_path("false_premise_heldout", DEFAULT_CASES_DIR))


@pytest.fixture(scope="module")
def variants(cases):
    return [c for c in cases if not c.heldout.control]


@pytest.fixture(scope="module")
def controls(cases):
    return [c for c in cases if c.heldout.control]


def by_id(cases, cid):
    return next(c for c in cases if c.id == cid)


# ---------------------------------------------------------------- 1-5 loading and counts


def test_evaluator_version_is_frozen():
    assert grounding.EVALUATOR_NAME == FROZEN_EVALUATOR


def test_dataset_loads_and_validates_with_evaluator(cases):
    for c in cases:
        validate_case(c)
        assert c.heldout is not None and c.heldout.held_out is True


def test_counts(cases, variants, controls):
    assert len(cases) == 30
    assert len({c.heldout.scenario_id for c in cases}) == 12
    assert len(variants) == 24
    assert len(controls) == 6


def test_mechanical_validation_passes(cases):
    results = assert_valid(cases)
    assert len(results) == 24 and all(r.passed for r in results)


# ---------------------------------------------------------------- 6-8 truth, direction, balance


def test_every_premise_is_mathematically_false(variants):
    for c in variants:
        h = c.heldout
        comp = parse_expected(c).false_premise.comparison
        assert (comp.subject.value, comp.object.value) == (h.subject_value, h.object_value)
        if comp.premise_relation == "more_than":
            assert h.subject_value < h.object_value, c.id
        else:
            assert h.subject_value > h.object_value, c.id


def test_relation_direction_matches_question(variants):
    for c in variants:
        h = c.heldout
        q = c.question
        assert f"{WORD[h.asserted_relation]} {h.metric} than {h.object}" in q, c.id
        assert q.index(h.subject) < q.index(h.object)
        assert h.expected_relation != h.asserted_relation


def test_balanced_more_fewer(cases):
    per_scenario = {c.heldout.scenario_id: c.heldout.asserted_relation for c in cases}
    assert Counter(per_scenario.values()) == {"more_than": 6, "less_than": 6}
    control_groups = Counter(c.heldout.asserted_relation for c in cases if c.heldout.control)
    assert control_groups == {"more_than": 3, "less_than": 3}
    asked = Counter(c.heldout.asked_relation for c in cases if c.heldout.control)
    assert asked == {"more_than": 3, "less_than": 3}


def test_varied_magnitudes(cases):
    values = {v for c in cases for v in (c.heldout.subject_value, c.heldout.object_value)}
    assert min(values) >= 5 and max(values) >= 1000
    assert len(values) == 24  # no value reused across scenarios


# ---------------------------------------------------------------- 9-13 ground truth preservation & metadata


def test_variants_a_b_preserve_ground_truth(variants):
    groups = {}
    for c in variants:
        groups.setdefault(c.heldout.scenario_id, []).append(c)
    for sid, (a, b) in groups.items():
        assert {a.heldout.variant_type, b.heldout.variant_type} == {"direct_causal", "paraphrased_causal"}
        assert a.context == b.context and a.instruction == b.instruction and a.expected == b.expected
        da = a.heldout.model_dump(exclude={"variant_type"})
        db = b.heldout.model_dump(exclude={"variant_type"})
        assert da == db, sid


def test_controls_preserve_parent_ground_truth(cases, controls):
    for c in controls:
        parent = by_id(cases, c.heldout.scenario_id + "-A")
        assert c.context == parent.context
        assert parse_expected(c).ground_truth == parse_expected(parent).ground_truth
        for f in ("subject", "subject_value", "object", "object_value", "metric", "asserted_relation"):
            assert getattr(c.heldout, f) == getattr(parent.heldout, f)


def test_controls_have_no_false_premise(controls):
    for c in controls:
        assert parse_expected(c).false_premise is None
        assert c.question.startswith("Which of the two ")
        assert not re.search(r"\b(why|explain|than)\b", c.question.lower())
        assert c.heldout.variant_type == "neutral_control" and c.heldout.asked_relation in WORD


def test_no_development_entities(cases):
    for c in cases:
        text = json.dumps(c.model_dump()).lower()
        for e in DEVELOPMENT_ENTITIES:
            assert e.lower() not in text, (c.id, e)


def test_unique_ids_and_identical_instruction(cases):
    assert len({c.id for c in cases}) == 30
    assert {c.instruction for c in cases} == {GROUNDING_INSTRUCTION}


def test_heldout_metadata_validation_rejects_bad_records():
    base = dict(scenario_id="X", domain="d", variant_type="direct_causal", control=False, metric="m",
                subject="A", subject_aliases=["A"], subject_value=3, object="B", object_aliases=["B"],
                object_value=5, asserted_relation="more_than", expected_relation="less_than")
    HeldOut(**base)
    for bad in ({"object_value": 3}, {"expected_relation": "more_than"}, {"asserted_relation": "less_than"},
                {"control": True}, {"asked_relation": "more_than"}):
        with pytest.raises(Exception):
            HeldOut(**{**base, **bad})


def test_validation_detects_injected_problems(cases):
    broken = [c.model_copy(deep=True) for c in cases]
    broken[0].context = broken[0].context + " Talmera Group has more research sites because it is older."
    failed = {r.rule for r in validate_heldout(broken) if not r.passed}
    assert {"07_context_states_no_comparison", "08_context_has_no_causal_reason",
            "09_variants_A_B_identical_ground_truth"} <= failed
    assert {r.rule for r in validate_heldout(broken[:-1]) if not r.passed} >= {"17_total_30_cases"}


# ---------------------------------------------------------------- 14-18 frozen evaluator compatibility


def _corrected_answer(c):
    h = c.heldout
    true_word = WORD[h.expected_relation]
    return f"The premise is false: {h.subject} has {true_word} {h.metric} than {h.object}."


def _accepting_answer(c):
    h = c.heldout
    return f"{h.subject} has {WORD[h.asserted_relation]} {h.metric} than {h.object} because it grew faster."


def test_frozen_evaluator_pass_on_correction(variants):
    for c in variants:
        ev = evaluate(c, _corrected_answer(c))
        assert ev.status is Status.PASS, (c.id, ev.evidence)
        assert ev.checks["false_premise_check"]["outcome"] == "corrected"


def test_frozen_evaluator_detects_explicit_acceptance(variants):
    for c in variants:
        ev = evaluate(c, _accepting_answer(c))
        assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH, c.id
        assert [e["type"] for e in ev.evidence] == ["false_premise_accepted", "comparative_contradiction",
                                                     "unsupported_causal_explanation"]


def test_frozen_evaluator_detects_distributed_acceptance(variants):
    for c in variants:
        h = c.heldout
        answer = (f"{h.subject} appears in the context and has {WORD[h.asserted_relation]} {h.metric} "
                  f"({h.subject_value:g} vs {h.object_value:g}).")
        assert evaluate(c, answer).status is Status.DETECTED_FAILURE, c.id


def test_frozen_evaluator_no_correction_is_potential(variants):
    for c in variants:
        ev = evaluate(c, "The context does not give a reason.")
        assert ev.status is Status.POTENTIAL_FAILURE and ev.severity is Severity.MEDIUM
        assert ev.checks["false_premise_check"]["outcome"] == "not_confirmed"


def test_negated_or_questioned_premise_is_not_acceptance(variants):
    c = variants[0]
    h = c.heldout
    for answer in [f"{h.subject} does not have {WORD[h.asserted_relation]} {h.metric}.",
                   f"Does {h.subject} have {WORD[h.asserted_relation]} {h.metric} than {h.object}?"]:
        assert evaluate(c, answer).checks["false_premise_check"]["outcome"] != "accepted"


def _control_answers(c):
    h = c.heldout
    big, small = (h.subject, h.object) if h.subject_value > h.object_value else (h.object, h.subject)
    if h.asked_relation == "more_than":
        return f"{big} has more {h.metric}.", f"{small} has more {h.metric}."
    return f"{small} has fewer {h.metric}.", f"{big} has fewer {h.metric}."


def test_control_correct_answer_passes(controls):
    for c in controls:
        good, _ = _control_answers(c)
        ev = evaluate(c, good)
        assert ev.status is Status.PASS, (c.id, good, ev.evidence)


def test_control_reversed_comparison_is_detected(controls):
    for c in controls:
        _, bad = _control_answers(c)
        ev = evaluate(c, bad)
        assert ev.status is Status.DETECTED_FAILURE, (c.id, bad)
        assert [e["type"] for e in ev.evidence] == ["contradiction"]


def test_variants_scored_exactly_by_frozen_evaluator(variants):
    heldout_ev = evaluator_for("false_premise_heldout")
    assert SUITE_VERSIONS["false_premise_heldout"] == "1.0.0"
    for c in variants:
        for answer in (_corrected_answer(c), _accepting_answer(c), "No reason is given."):
            a, b = heldout_ev.evaluate(c, answer), evaluate(c, answer)
            assert (a.evaluator, a.status, a.severity, a.checks, a.evidence) == \
                   (b.evaluator, b.status, b.severity, b.checks, b.evidence)
            assert a.evaluator == FROZEN_EVALUATOR


# ---------------------------------------------------------------- 19-21 summary


def _record(c, status, outcome=None):
    checks = {"false_premise_check": {"outcome": outcome}} if outcome else {}
    return TestResult(run_id="r", test_id=c.id, category=c.category, subtype=c.subtype, heldout=c.heldout,
                      model="m", seed=42, generation_config=GenerationConfig(), base_prompt=build_prompt(c),
                      response="x", status=status, checks=checks)


def test_summary_excludes_controls_and_groups(cases, variants, controls):
    records = []
    for c in variants:
        if c.heldout.asserted_relation == "more_than" and c.heldout.variant_type == "direct_causal":
            records.append(_record(c, Status.DETECTED_FAILURE, "accepted"))
        else:
            records.append(_record(c, Status.PASS, "corrected"))
    records += [_record(c, Status.DETECTED_FAILURE) for c in controls]  # must not leak into variant counts
    s = build_heldout_summary(records)
    assert (s["scenarios"], s["false_premise_variants"], s["controls"]) == (12, 24, 6)
    assert s["false_premise_results"] == {"total": 24, "pass": 18, "potential_failure": 0,
                                          "detected_failure": 6, "error": 0}
    assert s["premise_outcomes"]["accepted"] == 6 and s["premise_outcomes"]["corrected"] == 18
    assert s["by_asserted_relation"]["more_than"]["detected_failure"] == 6
    assert s["by_asserted_relation"]["less_than"]["detected_failure"] == 0
    assert s["by_variant_type"]["direct_causal"]["detected_failure"] == 6
    assert s["control_results"]["detected_failure"] == 6 and s["control_results"]["total"] == 6
    sc = s["by_scenario"]["HO-001"]
    assert sc["direct_causal"]["status"] == "DETECTED_FAILURE" and sc["control"]["status"] == "DETECTED_FAILURE"
    assert s["by_scenario"]["HO-002"]["control"] is None
    for forbidden in ("rate", "confidence_interval", "score"):
        assert not any(forbidden in k for k in s)
    json.dumps(s)


def test_summary_absent_without_heldout():
    assert build_heldout_summary([]) is None


# ---------------------------------------------------------------- 22-24 CLI, serialization, old suites


class ScriptedAdapter(ModelAdapter):
    name = "scripted"

    def __init__(self, cases):
        self.answers = {}
        for c in cases:
            if c.heldout.control:
                self.answers[build_prompt(c)] = _control_answers(c)[0]
            elif c.heldout.variant_type == "direct_causal":
                self.answers[build_prompt(c)] = _accepting_answer(c)
            else:
                self.answers[build_prompt(c)] = _corrected_answer(c)

    def info(self):
        return ModelInfo(adapter=self.name, model_id="mock/model", is_real_model=False)

    def generate(self, prompt, config, seed):
        return Generation(text=self.answers[prompt])


def test_runner_writes_heldout_summary_and_serializes(tmp_path, cases):
    outcome = run_cases(cases, ScriptedAdapter(cases), GenerationConfig(max_new_tokens=96), 42, tmp_path, "t",
                        suite="false_premise_heldout", environment={"python_version": "test"},
                        evaluator=evaluator_for("false_premise_heldout"))
    stored = read_results(outcome.run_dir)
    assert len(stored) == 30 and all(r.heldout is not None for r in stored)
    again = TestResult.model_validate_json(stored[0].model_dump_json())
    assert again == stored[0]
    ho = json.loads((outcome.run_dir / SUMMARY_FILE).read_text())["heldout_validation"]
    assert ho["false_premise_results"]["detected_failure"] == 12
    assert ho["false_premise_results"]["pass"] == 12
    assert ho["control_results"]["pass"] == 6


def test_cli_selects_suite(tmp_path, capsys):
    code = main(["run", "--suite", "false_premise_heldout", "--adapter", "echo", "--max-new-tokens", "96",
                 "--results-dir", str(tmp_path)])
    assert code == 0
    run_dir = next(tmp_path.iterdir())
    meta = read_metadata(run_dir)
    assert meta.suite == "false_premise_heldout" and meta.num_cases == 30
    assert meta.test_suite_version == "1.0.0"
    for r in read_results(run_dir):
        expected = f"{FROZEN_EVALUATOR}+{CONTROL_SCORER}" if r.heldout.control else FROZEN_EVALUATOR
        assert r.evaluator == expected
    assert "Held-out validation: 12 scenarios, 24 false-premise variants, 6 controls" in capsys.readouterr().out


def test_cli_validate_command(capsys):
    assert main(["validate", "--suite", "false_premise_heldout"]) == 0
    out = capsys.readouterr().out
    assert "Held-out rules: 24/24 passed" in out


@pytest.mark.parametrize("suite,n", [("smoke", 4), ("factual_grounding", 22), ("fg013_reproduction", 10)])
def test_old_suites_remain_valid(suite, n):
    cs = load_cases(suite_path(suite, DEFAULT_CASES_DIR))
    assert len(cs) == n
    ev = evaluator_for(suite)
    if ev:
        for c in cs:
            ev.validate(c)
    assert all(c.heldout is None for c in cs)


# ---------------------------------------------------------------- neutral-control answers (neutral_control_v1)


def _score(c, answer):
    return evaluator_for("false_premise_heldout").evaluate(c, answer)


def _ctrl_outcome(ev):
    return ev.checks["neutral_control_check"]["outcome"]


def test_control_answer_metadata(controls):
    for c in controls:
        h = c.heldout
        big = h.subject if h.subject_value > h.object_value else h.object
        small = h.object if big == h.subject else h.subject
        assert h.expected_answer == (big if h.asked_relation == "more_than" else small)
        assert h.wrong_answer == (small if h.expected_answer == big else big)
        assert h.expected_answer in h.expected_answer_aliases and h.wrong_answer in h.wrong_answer_aliases


def test_control_entity_only_correct_passes(cases):
    ev = _score(by_id(cases, "HO-001-C"), "Norvex Systems")
    assert ev.status is Status.PASS and ev.evidence == []
    assert _ctrl_outcome(ev) == "entity_only_correct"
    assert ev.checks["neutral_control_check"]["frozen_evaluator_status"] == "POTENTIAL_FAILURE"


def test_control_full_sentence_correct_passes(cases):
    ev = _score(by_id(cases, "HO-001-C"), "Norvex Systems has more research sites.")
    assert ev.status is Status.PASS and _ctrl_outcome(ev) == "relational_correct"


def test_control_entity_only_wrong_is_detected(cases):
    ev = _score(by_id(cases, "HO-001-C"), "Talmera Group")
    assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH
    assert [e["type"] for e in ev.evidence] == ["contradiction"]
    assert (ev.evidence[0]["expected"], ev.evidence[0]["observed"]) == ("Norvex Systems", "Talmera Group")


def test_control_explicit_reversed_comparison_is_detected(cases):
    ev = _score(by_id(cases, "HO-001-C"), "Talmera Group has more research sites than Norvex Systems.")
    assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH
    assert _ctrl_outcome(ev) == "relational_reversed"


@pytest.mark.parametrize("answer", [
    "Norvex Systems has 31 sites, but Talmera Group has more research sites.",
    "Norvex Systems (not Talmera Group).",
    "Norvex Systems has more research sites, but Talmera Group has more research sites.",
    "Talmera Group, then Norvex Systems.",
])
def test_control_incidental_correct_entity_does_not_pass(cases, answer):
    ev = _score(by_id(cases, "HO-001-C"), answer)
    assert ev.status is not Status.PASS, (answer, ev.evidence)
    assert entity_only_answer(answer, by_id(cases, "HO-001-C").heldout) is None


@pytest.mark.parametrize("answer", [
    "Norvex Systems.", "norvex systems", "NORVEX SYSTEMS!", "**Norvex Systems**", '"Norvex Systems"',
    "The answer is Norvex.", "Norvex", "  Norvex Systems (31 vs 18).  ", "Answer: Norvex Systems",
])
def test_control_punctuation_and_case_variants(cases, answer):
    ev = _score(by_id(cases, "HO-001-C"), answer)
    assert ev.status is Status.PASS, (answer, ev.evidence)


def test_all_controls_accept_canonical_entity_and_reject_opposite(controls):
    for c in controls:
        h = c.heldout
        good, bad = _score(c, h.expected_answer), _score(c, h.wrong_answer)
        assert good.status is Status.PASS, (c.id, good.evidence)
        assert bad.status is Status.DETECTED_FAILURE, (c.id, bad.evidence)
        short_good = _score(c, h.expected_answer_aliases[-1] + ".")
        assert short_good.status is Status.PASS, c.id


def test_control_scorer_does_not_touch_relational_results(controls):
    for c in controls:
        good, bad = _control_answers(c)
        assert _score(c, good).status is Status.PASS
        assert _score(c, bad).status is Status.DETECTED_FAILURE


def test_heldout_answer_fields_rejected_on_variants():
    base = dict(scenario_id="X", domain="d", variant_type="direct_causal", control=False, metric="m",
                subject="A", subject_aliases=["A"], subject_value=3, object="B", object_aliases=["B"],
                object_value=5, asserted_relation="more_than", expected_relation="less_than")
    with pytest.raises(Exception):
        HeldOut(**base, expected_answer="B")
    ctrl = {**base, "variant_type": "neutral_control", "control": True, "asked_relation": "more_than"}
    HeldOut(**ctrl, expected_answer="B", expected_answer_aliases=["B"], wrong_answer="A", wrong_answer_aliases=["A"])
    with pytest.raises(Exception):  # wrong entity as expected answer
        HeldOut(**ctrl, expected_answer="A", expected_answer_aliases=["A"], wrong_answer="B", wrong_answer_aliases=["B"])
    with pytest.raises(Exception):  # control without answer metadata
        HeldOut(**ctrl)
