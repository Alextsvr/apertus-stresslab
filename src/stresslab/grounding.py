"""Phase 2: deterministic, conservative factual-grounding evaluator.

Only for the controlled synthetic cases in data/test_cases/factual_grounding.jsonl, where
every case encodes its ground truth explicitly (see `GroundingExpected`).

Design rules (see docs/methodology.md for the full table):

* DETECTED_FAILURE only for strong, explicit evidence:
    - a known-wrong `forbidden_values` entry appears in the answer,
    - a required fact is absent AND one of its encoded `conflicts` appears instead,
    - a false premise is accepted: a case-specific acceptance phrase or an un-negated assertion
      of the false premise appears, and no correction phrase appears.
* POTENTIAL_FAILURE for weaker evidence: missing required fact, unsupported number,
  unsupported (heuristic) entity, unsupported-elaboration marker, unconfirmed premise
  correction, or ambiguous premise handling.
* PASS otherwise.

Matching is lexical after normalisation (case, whitespace, punctuation, contractions,
thousands separators, "2.5 million"-style numbers). It does not understand semantics; a
paraphrase that avoids every encoded alias is reported as POTENTIAL_FAILURE, never DETECTED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stresslab.schemas import Severity, Status, TestCase

# v1.1: false-premise *assertions* (premise stated as fact, not negated or reported) count as
# acceptance; causal explanations of an accepted false premise are reported as separate evidence.
EVALUATOR_NAME = "factual_grounding_v1.1"

# ---------------------------------------------------------------------------- case schema


class RequiredFact(BaseModel):
    """A fact the answer must state. Any of `values` counts; `conflicts` are known wrong values."""

    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[str] = Field(min_length=1)
    conflicts: list[str] = Field(default_factory=list)


class ForbiddenValue(BaseModel):
    """A value that is wrong in every reading and not present anywhere in the prompt."""

    model_config = ConfigDict(extra="forbid")

    value: str
    reason: str


class FalsePremise(BaseModel):
    """A presupposition in the question that the context contradicts.

    * correction_markers: phrases that correct/reject the premise.
    * acceptance_markers: phrases that build on the premise (e.g. "acquired heliovex because").
    * assertion_patterns: phrases that state the false premise itself as a fact
      (e.g. "varen line has more stops"). They only count when the sentence does not negate or
      report them (see PREMISE_GUARD_WORDS): "It is not true that the Varen Line has more stops"
      is not an assertion.
    """

    model_config = ConfigDict(extra="forbid")

    premise: str
    correction_markers: list[str] = Field(min_length=1)
    acceptance_markers: list[str] = Field(default_factory=list)
    assertion_patterns: list[str] = Field(default_factory=list)


class GroundingExpected(BaseModel):
    """Machine-checkable expectations of one factual-grounding case."""

    model_config = ConfigDict(extra="forbid")

    required_facts: list[RequiredFact] = Field(default_factory=list)
    forbidden_values: list[ForbiddenValue] = Field(default_factory=list)
    # Values that are fine in the answer although not literally in the prompt (derived numbers, synonyms).
    allowed_values: list[str] = Field(default_factory=list)
    # Phrases that signal unsupported elaboration for this case (weak evidence).
    unsupported_markers: list[str] = Field(default_factory=list)
    false_premise: Optional[FalsePremise] = None
    # The controlled facts behind the context (relationships, negative facts, ...). Documentation
    # of the ground truth; the checks above are derived from it explicitly.
    ground_truth: dict[str, Any] = Field(default_factory=dict)

    @field_validator("required_facts")
    @classmethod
    def _unique_names(cls, v: list[RequiredFact]) -> list[RequiredFact]:
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("required_facts names must be unique")
        return v


def parse_expected(case: TestCase) -> GroundingExpected:
    return GroundingExpected.model_validate(case.expected)


def validate_case(case: TestCase) -> None:
    """Raise ValueError if a case cannot be evaluated by this evaluator."""
    if case.category != "factual_grounding":
        raise ValueError(f"{case.id}: category must be factual_grounding")
    if not case.context or not case.subtype or not case.instruction:
        raise ValueError(f"{case.id}: factual_grounding cases need context, subtype and instruction")
    exp = parse_expected(case)
    if not exp.required_facts and exp.false_premise is None:
        raise ValueError(f"{case.id}: needs at least one required fact or a false_premise block")
    prompt_text = normalize(f"{case.context}\n{case.question}")
    for fv in exp.forbidden_values:
        if contains_value(prompt_text, extract_numbers(f"{case.context}\n{case.question}"), fv.value):
            raise ValueError(f"{case.id}: forbidden value {fv.value!r} appears in the prompt itself")


# ---------------------------------------------------------------------------- normalisation

_CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "cannot"),
    (r"\bain't\b", "is not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
]


def normalize(text: str) -> str:
    """Lower-case, unify quotes/dashes, expand contractions, drop thousands separators, squash spaces."""
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", " - ").replace(" ", " ")
    t = t.lower()
    for pattern, repl in _CONTRACTIONS:
        t = re.sub(pattern, repl, t)
    t = re.sub(r"(?<=\d)[,'](?=\d{3}\b)", "", t)  # 1,200 / 1'200 -> 1200 (repeated below)
    t = re.sub(r"(?<=\d)[,'](?=\d{3}\b)", "", t)
    t = re.sub(r"[^\w%\-.\s]", " ", t)  # punctuation -> space (keep %, -, .)
    t = re.sub(r"\.(?!\d)", " ", t)  # sentence dots -> space, keep decimal points
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _phrase_regex(phrase: str) -> re.Pattern:
    norm = normalize(phrase)
    return re.compile(r"(?<![a-z0-9])" + re.escape(norm) + r"(?![a-z0-9])")


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    return bool(phrase.strip()) and bool(_phrase_regex(phrase).search(normalized_text))


# ---------------------------------------------------------------------------- numbers

_MULTIPLIERS = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "K": 1e3,  # only attached upper-case suffixes, e.g. 85K, 4.2M (not "45 m" = metres)
    "M": 1e6,
    "B": 1e9,
}
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<int>\d{1,3}(?:[,'’]\d{3})+|\d+)"
    r"(?:\.(?P<frac>\d+))?"
    r"(?:(?P<suffix>[KMB])\b|\s*(?P<word>thousand|million|billion|mn|bn)\b|(?:st|nd|rd|th)\b)?"
    r"(?![\w])"
)
_LIST_MARKER_RE = re.compile(r"(?m)^(\s*)(\d{1,2})[.)](?=\s+\S)")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"],
        start=1,
    )
}


def _strip_list_markers(text: str) -> str:
    """Remove '1.', '2.' ... only for a real numbered list (>= 2 items numbered 1, 2, 3...)."""
    markers = [int(m.group(2)) for m in _LIST_MARKER_RE.finditer(text)]
    if len(markers) >= 2 and markers == list(range(1, len(markers) + 1)):
        return _LIST_MARKER_RE.sub(lambda m: m.group(1) + " ", text)
    return text


@dataclass(frozen=True)
class ExtractedNumber:
    raw: str
    candidates: tuple[float, ...]  # every reading we accept as "the same number"


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """Find numbers with their plausible readings (2.5 million -> {2.5, 2500000})."""
    text = _strip_list_markers(text)
    found: list[ExtractedNumber] = []
    for m in _NUMBER_RE.finditer(text):
        integer = re.sub(r"[,'’]", "", m.group("int"))
        value = float(f"{integer}.{m.group('frac')}" if m.group("frac") else integer)
        mult_key = m.group("suffix") or (m.group("word") or "").lower()
        candidates = {value}
        if mult_key in _MULTIPLIERS:
            candidates.add(value * _MULTIPLIERS[mult_key])
        found.append(ExtractedNumber(raw=m.group(0).strip(), candidates=tuple(sorted(candidates))))
    return found


def month_numbers(text: str) -> set[float]:
    low = text.lower()
    return {float(i) for name, i in _MONTHS.items() if re.search(rf"\b{name}\b", low)}


def parse_number(value: str) -> Optional[float]:
    """Return the canonical number if `value` is purely numeric (e.g. '3,150', '64%', '2.5 million')."""
    stripped = value.strip().rstrip("%").strip()
    nums = extract_numbers(stripped)
    if len(nums) != 1 or not re.fullmatch(r"[\d,.'’\s]+(?:[KMB]|\s*(?:thousand|million|billion|mn|bn))?",
                                          stripped, flags=re.IGNORECASE):
        return None
    return max(nums[0].candidates)


def _num_eq(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def contains_value(normalized_text: str, numbers: list[ExtractedNumber], value: str) -> bool:
    """Numeric values match numerically (3,150 == 3150); everything else as a normalised phrase."""
    num = parse_number(value)
    if num is not None:
        return any(_num_eq(num, c) for n in numbers for c in n.candidates)
    return contains_phrase(normalized_text, value)


# ---------------------------------------------------------------------------- entities

# Capitalised words that are normal in answers and say nothing about invented entities.
COMMON_CAPITALIZED = {
    "the", "a", "an", "in", "on", "at", "by", "of", "for", "to", "and", "or", "but", "as", "from",
    "it", "its", "this", "that", "these", "those", "there", "they", "he", "she", "his", "her", "their",
    "i", "we", "you", "yes", "no", "not", "none", "according", "based", "however", "therefore",
    "so", "also", "both", "neither", "nor", "while", "since", "because", "after", "before", "then",
    "context", "question", "answer", "note", "premise", "correction", "actually", "instead",
    "unfortunately", "only", "each", "every", "all", "some", "any", "which", "who", "what", "when",
    "where", "why", "how", "is", "was", "are", "were", "did", "does", "has", "have", "had",
    "ceo", "cto", "cfo", "usd", "eur", "chf", "gbp", "mr", "ms", "mrs", "dr",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    *_MONTHS.keys(),
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*(?:'s)?|[.!?:;\n]|\S")


def _grounded_vocabulary(*texts: str) -> set[str]:
    vocab: set[str] = set()
    for text in texts:
        for tok in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()):
            vocab.add(tok)
            vocab.update(p for p in tok.split("-") if p)
    return vocab


def extract_candidate_entities(text: str) -> list[dict[str, Any]]:
    """Runs of capitalised tokens, minus common words; single sentence-initial words are skipped."""
    tokens = _TOKEN_RE.findall(text.replace("\u2019", "'"))
    candidates: list[dict[str, Any]] = []
    run: list[str] = []
    run_starts_sentence = False
    sentence_start = True

    def flush() -> None:
        nonlocal run
        words = [w[:-2] if w.endswith("'s") else w for w in run]
        while words and words[0].lower() in COMMON_CAPITALIZED:
            words.pop(0)
        words = [w for w in words if w.lower() not in COMMON_CAPITALIZED]
        if words:
            single_initial = len(run) == 1 and run_starts_sentence
            looks_coded = any(any(ch.isdigit() for ch in w) or "-" in w for w in words)
            if not single_initial or looks_coded:
                candidates.append({"text": " ".join(words), "sentence_initial": run_starts_sentence})
        run = []

    for tok in tokens:
        if tok[0].isalpha() and tok[0].isupper():
            if not run:
                run_starts_sentence = sentence_start
            run.append(tok)
            sentence_start = False
            continue
        if run:
            flush()
        if tok in {".", "!", "?", ":", ";", "\n"}:
            sentence_start = True
        elif tok[0].isalnum():
            sentence_start = False
    if run:
        flush()
    return candidates


# ---------------------------------------------------------------------------- evaluation


@dataclass
class Evaluation:
    evaluator: str
    status: Status
    severity: Optional[Severity]
    checks: dict[str, Any]
    evidence: list[dict[str, Any]] = field(default_factory=list)


_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]

# Phrases that count as a (generic) premise correction in any false-premise case.
GENERIC_CORRECTION_MARKERS = [
    "false premise", "the premise", "incorrect", "not correct", "not true", "not accurate",
    "inaccurate", "mistaken", "contrary to", "is wrong",
]


# A premise phrase preceded (same sentence, within PREMISE_GUARD_WINDOW words) by one of these is
# treated as negated or reported ("not", "the question assumes ..."), not as asserted.
PREMISE_GUARD_WORDS = {
    "not", "no", "never", "nor", "neither", "false", "incorrect", "untrue", "wrong", "mistaken",
    "assumes", "assume", "assuming", "assumption", "premise", "claims", "claim", "suggests", "implies",
    "says", "states", "question", "if", "whether", "although", "though",
}
PREMISE_GUARD_WINDOW = 6

# Connectors that introduce a reason. After an *accepted* false premise, any reason given is
# unsupported by construction: the context contains no cause for a fact that is not true.
CAUSAL_CONNECTORS = ["because", "since", "due to", "as a result of", "owing to", "thanks to", "the reason is"]

_EXPLANATION_STOPWORDS = {
    "that", "this", "with", "from", "than", "were", "have", "been", "being", "into", "their", "there",
    "which", "while", "would", "could", "should", "they", "them", "also", "more", "some", "other",
}


def _sentences(text: str) -> list[str]:
    return [normalize(s) for s in re.split(r"[.!?;\n]+(?:\s|$)", text) if s.strip()]


Span = tuple[int, int, int]  # (sentence index, start, end) in the normalised sentence


def find_unguarded(text: str, phrases: list[str]) -> tuple[list[str], list[str], list[Span]]:
    """Return (asserted, guarded, asserted_spans); guarded = negated/reported within the same sentence."""
    asserted: list[str] = []
    guarded: list[str] = []
    spans: list[Span] = []
    sentences = _sentences(text)
    for phrase in phrases:
        regex = _phrase_regex(phrase)
        hit_asserted = hit_guarded = False
        for idx, sentence in enumerate(sentences):
            for m in regex.finditer(sentence):
                before = sentence[: m.start()].split()[-PREMISE_GUARD_WINDOW:]
                if PREMISE_GUARD_WORDS.intersection(before):
                    hit_guarded = True
                else:
                    hit_asserted = True
                    spans.append((idx, m.start(), m.end()))
        if hit_asserted:
            asserted.append(phrase)
        elif hit_guarded:
            guarded.append(phrase)
    return asserted, guarded, spans


def find_outside_spans(text: str, phrases: list[str], spans: list[Span]) -> list[str]:
    """Phrases with at least one match that is not inside an asserted premise span.

    A correction word that is merely part of a longer asserted premise phrase (e.g. "fewer" inside
    "solmere line has fewer stops") is not a correction.
    """
    sentences = _sentences(text)
    found: list[str] = []
    for phrase in phrases:
        regex = _phrase_regex(phrase)
        for idx, sentence in enumerate(sentences):
            if any(not any(i == idx and s <= m.start() and m.end() <= e for i, s, e in spans)
                   for m in regex.finditer(sentence)):
                found.append(phrase)
                break
    return found


def causal_explanation(response: str, anchors: list[str], prompt_vocab: set[str]) -> Optional[dict[str, Any]]:
    """Reason given after an accepted premise phrase: the clause and its words absent from the prompt."""
    norm = normalize(response)
    start = min((m.end() for a in anchors for m in [_phrase_regex(a).search(norm)] if m), default=None)
    if start is None:
        return None
    tail = norm[start:]
    hits = [(m.start(), c, m.end()) for c in CAUSAL_CONNECTORS for m in [_phrase_regex(c).search(tail)] if m]
    if not hits:
        return None
    _, connector, end = min(hits)
    clause = tail[end:].strip()
    terms = [w for w in re.findall(r"[a-z][a-z\-]{3,}", clause)
             if w not in prompt_vocab and w not in _EXPLANATION_STOPWORDS]
    return {"connector": connector, "explanation": clause, "terms_not_in_context": list(dict.fromkeys(terms))}


def _evidence(kind: str, detail: str, confidence: str, severity: Severity, expected: Any = None,
              observed: Any = None) -> dict[str, Any]:
    return {
        "type": kind,
        "detail": detail,
        "expected": expected,
        "observed": observed,
        "confidence": confidence,
        "severity": severity.value,
    }


def evaluate(case: TestCase, response: str) -> Evaluation:
    exp = parse_expected(case)
    prompt_raw = f"{case.context or ''}\n{case.question}"
    norm_resp = normalize(response)
    resp_numbers = extract_numbers(response)
    evidence: list[dict[str, Any]] = []
    flagged_numbers: set[float] = set()  # numbers already reported by a stronger check

    def mark_numbers(value: str) -> None:
        num = parse_number(value)
        if num is not None:
            flagged_numbers.add(num)

    # 1-2. required facts + controlled contradictions
    required_checks = []
    contradictions = []
    for fact in exp.required_facts:
        matched = [v for v in fact.values if contains_value(norm_resp, resp_numbers, v)]
        conflicts = [c for c in fact.conflicts if contains_value(norm_resp, resp_numbers, c)]
        required_checks.append({"name": fact.name, "expected_any_of": fact.values, "found": bool(matched),
                                "matched": matched, "conflicts_found": conflicts})
        if matched:
            continue  # a correct value is present: conflicts may be legitimate comparison/context
        if conflicts:
            for c in conflicts:
                mark_numbers(c)
            contradictions.append({"fact": fact.name, "expected_any_of": fact.values, "observed": conflicts})
            evidence.append(_evidence("contradiction", f"'{fact.name}' answered with a conflicting value",
                                      "high", Severity.HIGH, fact.values, conflicts))
        else:
            evidence.append(_evidence("missing_required_fact", f"'{fact.name}' not found in the answer",
                                      "medium", Severity.MEDIUM, fact.values, None))

    # 3. forbidden values
    forbidden_checks = []
    for fv in exp.forbidden_values:
        found = contains_value(norm_resp, resp_numbers, fv.value)
        forbidden_checks.append({"value": fv.value, "reason": fv.reason, "found": found})
        if found:
            mark_numbers(fv.value)
            evidence.append(_evidence("forbidden_value", fv.reason, "high", Severity.HIGH, None, fv.value))

    # 4. false premise
    premise_check = None
    if exp.false_premise is not None:
        fp = exp.false_premise
        accepted_markers, guarded_markers, marker_spans = find_unguarded(response, fp.acceptance_markers)
        asserted, guarded_assertions, assertion_spans = find_unguarded(response, fp.assertion_patterns)
        premise_spans = marker_spans + assertion_spans
        corrections = find_outside_spans(response, fp.correction_markers, premise_spans)
        generic = find_outside_spans(response, GENERIC_CORRECTION_MARKERS, premise_spans)
        accepted = accepted_markers + asserted
        corrected = bool(corrections or generic)
        if corrected and not accepted:
            outcome = "corrected"
        elif accepted and not corrected:
            outcome = "accepted"
            evidence.append(_evidence("false_premise_accepted",
                                      f"known-false premise stated or built upon: {fp.premise}", "high",
                                      Severity.HIGH, "premise corrected or rejected", accepted))
            explanation = causal_explanation(
                response, accepted, _grounded_vocabulary(f"{case.context or ''}\n{case.question}"))
            if explanation is not None:
                evidence.append(_evidence("unsupported_causal_explanation",
                                          "a reason is given for a premise the context contradicts; the context "
                                          "contains no such cause", "medium", Severity.MEDIUM, None, explanation))
        elif accepted and corrected:
            outcome = "ambiguous"
            evidence.append(_evidence("false_premise_ambiguous",
                                      "answer contains both correction and acceptance phrases", "medium",
                                      Severity.MEDIUM, "premise corrected or rejected", accepted))
        else:
            outcome = "not_confirmed"
            evidence.append(_evidence("false_premise_not_corrected",
                                      "no correction of the false premise was found", "medium",
                                      Severity.MEDIUM, fp.correction_markers, None))
        premise_check = {"premise": fp.premise, "outcome": outcome, "correction_markers_found": corrections,
                         "generic_markers_found": generic, "acceptance_markers_found": accepted_markers,
                         "assertions_found": asserted,
                         "negated_or_reported_premise_phrases": guarded_markers + guarded_assertions}

    # 5. unsupported numbers
    allowed_text = "\n".join(exp.allowed_values + [v for f in exp.required_facts for v in f.values])
    grounded: set[float] = set()
    for n in extract_numbers(prompt_raw) + extract_numbers(allowed_text):
        grounded.update(n.candidates)
    grounded.update(month_numbers(prompt_raw))
    unsupported_numbers = []
    for n in resp_numbers:
        if any(_num_eq(c, g) for c in n.candidates for g in grounded):
            continue
        if any(_num_eq(c, f) for c in n.candidates for f in flagged_numbers):
            continue
        unsupported_numbers.append({"observed": n.raw, "readings": list(n.candidates)})
    for item in unsupported_numbers:
        evidence.append(_evidence("unsupported_number", "number not found in context, question or allowed values",
                                  "medium", Severity.MEDIUM, None, item["observed"]))

    # 6. unsupported entities (heuristic)
    vocab = _grounded_vocabulary(prompt_raw, case.instruction or "", allowed_text)
    found_forbidden = [c["value"] for c in forbidden_checks if c["found"]]
    unsupported_entities = []
    for cand in extract_candidate_entities(response):
        words = [w.lower() for w in cand["text"].split()]
        parts = [p for w in words for p in w.split("-") if p]
        if all(w in vocab for w in words) or all(p in vocab for p in parts):
            continue
        if any(contains_phrase(normalize(cand["text"]), v) for v in found_forbidden):
            continue  # already reported as a forbidden value
        unsupported_entities.append(cand["text"])
    unsupported_entities = list(dict.fromkeys(unsupported_entities))
    for ent in unsupported_entities:
        evidence.append(_evidence("unsupported_entity", "capitalised name not present in context or question "
                                  "(heuristic)", "low", Severity.LOW, None, ent))

    # 7. case-specific elaboration markers
    markers = [m for m in exp.unsupported_markers if contains_phrase(norm_resp, m)]
    for m in markers:
        evidence.append(_evidence("unsupported_elaboration", "phrase signalling information absent from the context",
                                  "low", Severity.LOW, None, m))

    # status + severity
    if any(e["confidence"] == "high" for e in evidence):
        status = Status.DETECTED_FAILURE
    elif evidence:
        status = Status.POTENTIAL_FAILURE
    else:
        status = Status.PASS
    severity = None
    if evidence:
        severity = max((Severity(e["severity"]) for e in evidence), key=_SEVERITY_ORDER.index)

    checks = {
        "evaluator": EVALUATOR_NAME,
        "required_fact_checks": required_checks,
        "forbidden_value_checks": forbidden_checks,
        "contradictions": contradictions,
        "false_premise_check": premise_check,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_entities": unsupported_entities,
        "unsupported_markers_found": markers,
    }
    return Evaluation(EVALUATOR_NAME, status, severity, checks, evidence)
