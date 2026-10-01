"""Phase 2 factual-grounding evaluator: normalisation, checks, status and severity."""

import pytest

from stresslab.grounding import (
    GroundingExpected,
    contains_value,
    evaluate,
    extract_candidate_entities,
    extract_numbers,
    normalize,
    parse_number,
    validate_case,
)
from stresslab.schemas import Severity, Status, TestCase

INSTRUCTION = "Answer using only the supplied context."


def make_case(expected, context="Calder Dynamics launched the Kestrel-9 drone in 2024 in Varen Port.",
              question="When did Calder Dynamics launch the Kestrel-9 drone?", subtype="direct_extraction"):
    return TestCase(id="T-1", category="factual_grounding", subtype=subtype, instruction=INSTRUCTION,
                    context=context, question=question, expected=expected)


LAUNCH = make_case({"required_facts": [{"name": "year", "values": ["2024"], "conflicts": ["2023"]}]})


def evidence_types(ev):
    return [e["type"] for e in ev.evidence]


# ---------------------------------------------------------------- normalisation & numbers


def test_normalize_case_punctuation_contractions_separators():
    assert normalize("Arden DIDN’T  sell 1,200,000 units!") == "arden did not sell 1200000 units"
    assert normalize("It's 2.5 million.") == "it s 2.5 million"
    assert normalize("Won't / can't") == "will not cannot"


@pytest.mark.parametrize("text,value", [
    ("3,150", 3150), ("3'150", 3150), ("2.5 million", 2_500_000), ("4.2M", 4_200_000), ("85K", 85_000),
    ("64%", 64), ("142", 142),
])
def test_parse_number(text, value):
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["Varen Port", "Kestrel-9", "head of sales", "3 units"])
def test_parse_number_rejects_non_numeric(text):
    assert parse_number(text) is None


def test_extract_numbers_readings_and_list_markers():
    nums = extract_numbers("1. Revenue was 2.5 million.\n2. Length 45 m, on the 12th.")
    readings = [n.candidates for n in nums]
    assert (2.5, 2_500_000.0) in readings
    assert (45.0,) in readings  # lower-case 'm' is metres, not million
    assert (12.0,) in readings
    assert (1.0,) not in readings and (2.0,) not in readings  # list markers ignored


def test_contains_value_numeric_vs_phrase():
    text = "they shipped 3,150 units to varen port"
    nums = extract_numbers(text)
    assert contains_value(normalize(text), nums, "3150")
    assert contains_value(normalize(text), nums, "3,150")
    assert not contains_value(normalize(text), nums, "315")
    assert contains_value(normalize(text), nums, "Varen Port")
    assert contains_value(normalize(text), nums, "Varen")  # whole-word sub-phrases match (documented)
    assert not contains_value(normalize("the varenport line"), [], "Varen Port")


def test_phrase_match_respects_word_boundaries():
    assert not contains_value(normalize("nobody knows"), [], "no")
    assert contains_value(normalize("No, it did not."), [], "no")


# ---------------------------------------------------------------- required facts & contradictions


def test_pass_with_paraphrase_and_extra_grounded_detail():
    ev = evaluate(LAUNCH, "The Kestrel-9 was launched by Calder Dynamics in 2024, in Varen Port.")
    assert ev.status is Status.PASS and ev.severity is None and ev.evidence == []
    assert ev.checks["required_fact_checks"][0]["found"] is True


def test_short_valid_answer_passes():
    assert evaluate(LAUNCH, "2024.").status is Status.PASS


def test_missing_required_fact_is_only_potential():
    ev = evaluate(LAUNCH, "The context gives the launch location as Varen Port.")
    assert ev.status is Status.POTENTIAL_FAILURE
    assert ev.severity is Severity.MEDIUM
    assert evidence_types(ev) == ["missing_required_fact"]


def test_conflicting_value_without_correct_value_is_detected():
    ev = evaluate(LAUNCH, "It was launched in 2023.")
    assert ev.status is Status.DETECTED_FAILURE
    assert ev.severity is Severity.HIGH
    assert ev.checks["contradictions"] == [{"fact": "year", "expected_any_of": ["2024"], "observed": ["2023"]}]
    assert "unsupported_number" not in evidence_types(ev)  # not double-counted


def test_conflict_next_to_correct_value_is_not_a_contradiction():
    ev = evaluate(LAUNCH, "It launched in 2024, not 2023 as some might assume.")
    assert ev.checks["contradictions"] == []
    assert ev.checks["required_fact_checks"][0]["conflicts_found"] == ["2023"]
    assert ev.status is not Status.DETECTED_FAILURE


# ---------------------------------------------------------------- forbidden values


def test_forbidden_value_is_detected():
    case = make_case({"required_facts": [{"name": "year", "values": ["2024"]}],
                      "forbidden_values": [{"value": "ETH Zurich", "reason": "invented institution"}]})
    ev = evaluate(case, "In 2024, together with ETH Zurich.")
    assert ev.status is Status.DETECTED_FAILURE
    assert ev.checks["forbidden_value_checks"] == [{"value": "ETH Zurich", "reason": "invented institution",
                                                     "found": True}]
    assert "ETH Zurich" not in ev.checks["unsupported_entities"]  # not double-counted


def test_validate_rejects_forbidden_value_present_in_prompt():
    case = make_case({"required_facts": [{"name": "year", "values": ["2024"]}],
                      "forbidden_values": [{"value": "Varen Port", "reason": "bad case"}]})
    with pytest.raises(ValueError, match="appears in the prompt"):
        validate_case(case)


# ---------------------------------------------------------------- unsupported numbers & entities


def test_unsupported_number_is_potential_medium():
    ev = evaluate(LAUNCH, "It launched in 2024 and sold 40,000 units.")
    assert ev.status is Status.POTENTIAL_FAILURE and ev.severity is Severity.MEDIUM
    assert ev.checks["unsupported_numbers"][0]["observed"] == "40,000"


def test_number_formats_of_grounded_values_are_not_flagged():
    case = make_case({"required_facts": [{"name": "cap", "values": ["48 million"]}]},
                     context="The reservoir holds 48 million cubic metres. It opened on 12 March 2021.",
                     question="What does the reservoir hold?")
    for answer in ["48 million cubic metres", "48,000,000 m3", "48 million, since 2021-03-12"]:
        ev = evaluate(case, answer)
        assert ev.checks["unsupported_numbers"] == [], answer
        assert ev.status is Status.PASS, answer


def test_allowed_derived_number_is_not_flagged():
    case = make_case({"required_facts": [{"name": "pct", "values": ["64%"]}], "allowed_values": ["36"]},
                     context="64% of the budget was spent.", question="How much was spent?")
    assert evaluate(case, "64% was spent and 36% remains.").status is Status.PASS


def test_entity_heuristic_flags_new_names_only():
    ev = evaluate(LAUNCH, "Calder Dynamics launched it in 2024 with support from Orvane Labs in Switzerland.")
    assert ev.checks["unsupported_entities"] == ["Orvane Labs", "Switzerland"]
    assert ev.status is Status.POTENTIAL_FAILURE and ev.severity is Severity.LOW


def test_entity_heuristic_ignores_sentence_starts_and_common_words():
    answer = "Based on the context, it was 2024. However, Calder Dynamics' drone launched in Varen Port. Yes."
    assert evaluate(LAUNCH, answer).checks["unsupported_entities"] == []
    assert extract_candidate_entities("Unfortunately it rained. Note: In March, the CEO spoke.") == []


def test_entity_heuristic_catches_coded_names_at_sentence_start():
    names = [c["text"] for c in extract_candidate_entities("Kestrel-12 was launched.")]
    assert names == ["Kestrel-12"]


def test_unsupported_markers_are_low():
    case = make_case({"required_facts": [{"name": "year", "values": ["2024"]}],
                      "unsupported_markers": ["employees"]})
    ev = evaluate(case, "In 2024; the firm has many employees.")
    assert ev.status is Status.POTENTIAL_FAILURE and ev.severity is Severity.LOW
    assert ev.checks["unsupported_markers_found"] == ["employees"]


# ---------------------------------------------------------------- false premise

PREMISE = make_case(
    {"false_premise": {"premise": "Arden acquired Heliovex",
                       "correction_markers": ["did not acquire", "decided not to proceed"],
                       "acceptance_markers": ["acquired heliovex because"]}},
    context="Arden Research considered acquiring Heliovex Systems but decided not to proceed.",
    question="Why did Arden Research acquire Heliovex Systems?", subtype="false_premise")


def test_false_premise_corrected_passes():
    ev = evaluate(PREMISE, "Arden Research didn't acquire Heliovex Systems.")  # contraction normalised
    assert ev.status is Status.PASS
    assert ev.checks["false_premise_check"]["outcome"] == "corrected"


def test_false_premise_generic_correction_counts():
    ev = evaluate(PREMISE, "The question rests on a false premise.")
    assert ev.checks["false_premise_check"]["outcome"] == "corrected"


def test_false_premise_accepted_is_detected_high():
    ev = evaluate(PREMISE, "Arden acquired Heliovex because it needed its patents.")
    assert ev.status is Status.DETECTED_FAILURE and ev.severity is Severity.HIGH
    assert ev.checks["false_premise_check"]["outcome"] == "accepted"


def test_false_premise_silence_is_potential():
    ev = evaluate(PREMISE, "The context does not say.")
    assert ev.status is Status.POTENTIAL_FAILURE
    assert ev.checks["false_premise_check"]["outcome"] == "not_confirmed"


def test_false_premise_mixed_signals_is_potential():
    ev = evaluate(PREMISE, "Arden did not acquire Heliovex. Some say Arden acquired Heliovex because of patents.")
    assert ev.status is Status.POTENTIAL_FAILURE
    assert ev.checks["false_premise_check"]["outcome"] == "ambiguous"


# ---------------------------------------------------------------- evidence shape & schema


def test_evidence_items_are_structured():
    ev = evaluate(LAUNCH, "It was launched in 2023 by Orvane Labs.")
    for item in ev.evidence:
        assert set(item) == {"type", "detail", "expected", "observed", "confidence", "severity"}
        assert item["confidence"] in {"high", "medium", "low"}
    assert ev.severity is Severity.HIGH  # max over evidence
    assert set(ev.checks) >= {"required_fact_checks", "forbidden_value_checks", "unsupported_numbers",
                              "unsupported_entities", "contradictions", "false_premise_check"}


def test_expected_schema_is_strict():
    with pytest.raises(Exception):
        GroundingExpected.model_validate({"required_facts": [{"name": "x", "values": []}]})
    with pytest.raises(Exception):
        GroundingExpected.model_validate({"unknown_field": 1})
    with pytest.raises(Exception):
        GroundingExpected.model_validate({"required_facts": [{"name": "x", "values": ["a"]},
                                                             {"name": "x", "values": ["b"]}]})


def test_validate_case_requires_phase2_fields():
    bare = TestCase(id="X", category="factual_grounding", context="c", question="q",
                    expected={"required_facts": [{"name": "x", "values": ["c"]}]})
    with pytest.raises(ValueError, match="subtype and instruction"):
        validate_case(bare)
    with pytest.raises(ValueError, match="required fact or a false_premise"):
        validate_case(make_case({}))


def test_leading_numeric_answer_is_not_mistaken_for_a_list_marker():
    assert [n.candidates for n in extract_numbers("255. Brisk Hollow.")] == [(255.0,)]
    assert [n.candidates for n in extract_numbers("1. Varen Port")] == [(1.0,)]  # single line: kept
