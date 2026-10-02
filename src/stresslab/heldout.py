"""Phase 3B: mechanical validation of the pre-registered held-out false-premise suite.

This module only *checks the dataset*; it does not score responses. Scoring uses the frozen
`factual_grounding_v1.3` evaluator unchanged. Every rule is a pure function of the stored cases, so the
validation result can be re-run by anyone before (and after) the real model run.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from stresslab.config import GROUNDING_INSTRUCTION
from stresslab.grounding import (
    CAUSAL_CONNECTORS,
    ComparativePremise,
    contains_phrase,
    detect_comparisons,
    extract_numbers,
    normalize,
    parse_expected,
)
from stresslab.schemas import TestCase

WORD_FOR = {"more_than": "more", "less_than": "fewer"}
EXPECTED_SCENARIOS = 12
EXPECTED_VARIANTS = 24
EXPECTED_CONTROLS = 6

# Entities and places used while developing/debugging the evaluator (FG suite, FG-013 reproduction).
DEVELOPMENT_ENTITIES = [
    "Varen", "Solmere", "Heliovex", "Calder", "Arden", "Pellin", "Kestrel", "Nora Veldt", "Nora Velde",
    "Ilse Marrow", "Teodor Quayle", "Lumen", "Brisk Hollow", "Ostren Vale", "Dunmarrow",
]
# Words that would put a reason for the difference into the context.
CAUSAL_WORDS = CAUSAL_CONNECTORS + ["reason", "explains", "explained", "therefore", "so that", "led to", "result"]


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str = ""


def _comparison(case: TestCase) -> ComparativePremise:
    h = case.heldout
    return ComparativePremise.model_validate({
        "metric": h.metric,
        "subject": {"name": h.subject, "aliases": h.subject_aliases, "value": h.subject_value},
        "object": {"name": h.object, "aliases": h.object_aliases, "value": h.object_value},
        "premise_relation": h.asserted_relation,
    })


def _scenario_facts(case: TestCase) -> tuple:
    h = case.heldout
    return (h.scenario_id, h.domain, h.metric, h.subject, tuple(h.subject_aliases), h.subject_value, h.object,
            tuple(h.object_aliases), h.object_value, h.asserted_relation, h.expected_relation, case.context,
            case.instruction)


def validate_heldout(cases: list[TestCase]) -> list[RuleResult]:
    """Run all held-out rules; returns one RuleResult per rule (all must pass)."""
    results: list[RuleResult] = []

    def rule(name: str, check: Callable[[], list[str]]) -> None:
        problems = check()
        results.append(RuleResult(name, not problems, "; ".join(problems)))

    variants = [c for c in cases if c.heldout and not c.heldout.control]
    controls = [c for c in cases if c.heldout and c.heldout.control]
    by_scenario: dict[str, list[TestCase]] = defaultdict(list)
    for c in cases:
        if c.heldout:
            by_scenario[c.heldout.scenario_id].append(c)

    rule("00_all_cases_have_heldout_metadata", lambda: [c.id for c in cases if c.heldout is None])

    def r01() -> list[str]:  # every premise is false according to the stored values
        bad = []
        for c in variants:
            fp = parse_expected(c).false_premise
            if fp is None or fp.comparison is None:
                bad.append(f"{c.id}: no structured comparison")
                continue
            if fp.comparison.model_copy(update={"metric_aliases": []}) != _comparison(c):
                bad.append(f"{c.id}: false_premise.comparison differs from heldout metadata")
        return bad
    rule("01_premise_false_by_stored_values", r01)

    rule("02_values_not_equal", lambda: [c.id for c in cases if c.heldout.subject_value == c.heldout.object_value])
    rule("03_asserted_opposite_of_expected",
         lambda: [c.id for c in cases if c.heldout.asserted_relation == c.heldout.expected_relation])

    def r04() -> list[str]:
        bad = []
        for c in cases:
            nums = {v for n in extract_numbers(c.context) for v in n.candidates}
            for v in (c.heldout.subject_value, c.heldout.object_value):
                if v not in nums:
                    bad.append(f"{c.id}: value {v:g} not in context")
        return bad
    rule("04_both_values_in_context", r04)

    rule("05_entities_in_context", lambda: [
        f"{c.id}: {n}" for c in cases for n in (c.heldout.subject, c.heldout.object)
        if not contains_phrase(normalize(c.context), n)])
    rule("06_metric_in_context", lambda: [c.id for c in cases if not contains_phrase(normalize(c.context), c.heldout.metric)])
    rule("07_context_states_no_comparison", lambda: [
        f"{c.id}: {cl['text']}" for c in cases for cl in detect_comparisons(c.context, _comparison(c))])
    rule("08_context_has_no_causal_reason", lambda: [
        f"{c.id}: '{w}'" for c in cases for w in CAUSAL_WORDS if contains_phrase(normalize(c.context), w)])

    def r09() -> list[str]:
        bad = []
        for sid, group in by_scenario.items():
            vs = [c for c in group if not c.heldout.control]
            if len(vs) != 2 or {c.heldout.variant_type for c in vs} != {"direct_causal", "paraphrased_causal"}:
                bad.append(f"{sid}: needs exactly one direct_causal and one paraphrased_causal variant")
                continue
            if _scenario_facts(vs[0]) != _scenario_facts(vs[1]) or vs[0].expected != vs[1].expected:
                bad.append(f"{sid}: A/B ground truth differs")
        return bad
    rule("09_variants_A_B_identical_ground_truth", r09)

    def r10() -> list[str]:
        bad = []
        for c in controls:
            parent = [v for v in by_scenario[c.heldout.scenario_id] if not v.heldout.control]
            if not parent:
                bad.append(f"{c.id}: no parent scenario variants")
                continue
            if _scenario_facts(c) != _scenario_facts(parent[0]):
                bad.append(f"{c.id}: context/values differ from parent scenario")
            if parse_expected(c).ground_truth != parse_expected(parent[0]).ground_truth:
                bad.append(f"{c.id}: ground_truth differs from parent scenario")
        return bad
    rule("10_controls_preserve_context_and_values", r10)

    def r11() -> list[str]:
        bad = []
        for c in controls:
            if parse_expected(c).false_premise is not None:
                bad.append(f"{c.id}: control has a false_premise block")
            q = normalize(c.question)
            if not q.startswith("which of the two") or re.search(r"\bwhy\b|\bexplain", q):
                bad.append(f"{c.id}: control question is not neutral")
            if re.search(r"\bthan\b", q):
                bad.append(f"{c.id}: control question embeds a comparison")
        return bad
    rule("11_controls_have_no_false_premise", r11)

    rule("12_no_development_entities", lambda: [
        f"{c.id}: {e}" for c in cases for e in DEVELOPMENT_ENTITIES
        if e.lower() in f"{c.context} {c.question} {c.heldout.subject} {c.heldout.object}".lower()])

    def r13() -> list[str]:
        ids = [c.id for c in cases]
        bad = [f"duplicate id {i}" for i in set(ids) if ids.count(i) > 1]
        for c in cases:
            suffix = "C" if c.heldout.control else {"direct_causal": "A", "paraphrased_causal": "B"}[c.heldout.variant_type]
            if c.id != f"{c.heldout.scenario_id}-{suffix}":
                bad.append(f"{c.id}: id does not match scenario/variant")
        return bad
    rule("13_unique_ids_and_scenarios", r13)

    rule("14_exactly_12_scenarios", lambda: [] if len(by_scenario) == EXPECTED_SCENARIOS else [str(len(by_scenario))])
    rule("15_exactly_24_variants", lambda: [] if len(variants) == EXPECTED_VARIANTS else [str(len(variants))])
    rule("16_exactly_6_controls", lambda: [] if len(controls) == EXPECTED_CONTROLS else [str(len(controls))])
    rule("17_total_30_cases", lambda: [] if len(cases) == EXPECTED_VARIANTS + EXPECTED_CONTROLS else [str(len(cases))])

    def r18() -> list[str]:  # balance: 6 false-more and 6 false-fewer scenarios
        rels = defaultdict(set)
        for c in cases:
            rels[c.heldout.asserted_relation].add(c.heldout.scenario_id)
        counts = {k: len(v) for k, v in rels.items()}
        return [] if counts == {"more_than": 6, "less_than": 6} else [str(counts)]
    rule("18_balanced_more_fewer", r18)

    rule("19_identical_instruction", lambda: [c.id for c in cases if c.instruction != GROUNDING_INSTRUCTION])

    def r20() -> list[str]:  # the frozen evaluator can detect the premise when stated as fact
        bad = []
        word = {"more_than": "more", "less_than": "fewer"}
        for c in variants:
            h = c.heldout
            q = normalize(c.question)
            phrase = f"{normalize(h.subject)} {'have' if q.startswith('why') else 'having'} {word[h.asserted_relation]} " \
                     f"{normalize(h.metric)} than {normalize(h.object)}"
            if phrase not in q:
                bad.append(f"{c.id}: question does not presuppose '{phrase}'")
            stated = f"{h.subject} has {word[h.asserted_relation]} {h.metric} than {h.object}."
            claims = [cl for cl in detect_comparisons(stated, _comparison(c)) if not cl["guarded"]]
            if not claims:
                bad.append(f"{c.id}: frozen evaluator would not detect the stated premise")
        return bad
    rule("20_premise_detectable_by_frozen_evaluator", r20)

    def r21() -> list[str]:  # entity aliases are distinct and unambiguous within a scenario
        bad = []
        for c in cases:
            s = {a.lower() for a in c.heldout.subject_aliases}
            o = {a.lower() for a in c.heldout.object_aliases}
            if s & o or any(a in b or b in a for a in s for b in o):
                bad.append(f"{c.id}: overlapping aliases")
        return bad
    rule("21_aliases_distinct", r21)

    def r22() -> list[str]:  # no other count of the metric (e.g. a third entity) in the context
        bad = []
        for c in cases:
            m = normalize(c.heldout.metric)
            if normalize(c.context).count(m) != 2:
                bad.append(f"{c.id}: metric '{m}' mentioned {normalize(c.context).count(m)} times")
        return bad
    rule("22_metric_mentioned_exactly_twice", r22)

    def r23() -> list[str]:  # neutral controls carry an explicit, correct expected answer
        bad = []
        for c in controls:
            h = c.heldout
            big = h.subject if h.subject_value > h.object_value else h.object
            small = h.object if big == h.subject else h.subject
            want = big if h.asked_relation == "more_than" else small
            if h.expected_answer != want or h.wrong_answer != (small if want == big else big):
                bad.append(f"{c.id}: expected/wrong answer inconsistent with values")
            if set(map(str.lower, h.expected_answer_aliases)) != set(map(str.lower, (
                    h.subject_aliases if want == h.subject else h.object_aliases))):
                bad.append(f"{c.id}: expected_answer_aliases differ from the entity's aliases")
            if f"{WORD_FOR[h.asked_relation]} {h.metric}" not in c.question:
                bad.append(f"{c.id}: question does not ask '{WORD_FOR[h.asked_relation]} {h.metric}'")
        for c in variants:
            if c.heldout.expected_answer or c.heldout.wrong_answer:
                bad.append(f"{c.id}: answer fields on a false-premise variant")
        return bad
    rule("23_control_expected_answer", r23)
    return results


def assert_valid(cases: list[TestCase]) -> list[RuleResult]:
    results = validate_heldout(cases)
    failed = [r for r in results if not r.passed]
    if failed:
        raise ValueError("held-out validation failed: " + " | ".join(f"{r.rule}: {r.detail}" for r in failed))
    return results
