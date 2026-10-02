"""Pydantic schemas for test cases, run metadata and per-test result records."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1"

Category = Literal["smoke", "factual_grounding", "consistency", "robustness"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status(str, Enum):
    PASS = "PASS"
    # Phase 2+ evaluator outcomes (conservative, see docs/methodology.md).
    POTENTIAL_FAILURE = "POTENTIAL_FAILURE"
    DETECTED_FAILURE = "DETECTED_FAILURE"
    # Reserved names from the original draft schema; not emitted by any current evaluator.
    WARNING = "WARNING"
    FAIL = "FAIL"
    # No evaluator for this suite (e.g. smoke): stored but not judged.
    UNSCORED = "UNSCORED"
    # The adapter raised an error; the record keeps the error message as evidence.
    ERROR = "ERROR"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Lineage(BaseModel):
    """Where a derived (mutated) case comes from and what was changed (Phase 3A)."""

    model_config = ConfigDict(extra="forbid")

    parent_test_id: str
    mutation_id: str
    mutation_type: str
    role: Literal["false_premise_variant", "control"]
    changed_fields: list[Literal["context", "question", "instruction"]] = Field(min_length=1)
    preserves_ground_truth: bool
    # True if the instruction explicitly asks the model to correct/challenge a wrong question.
    explicit_correction_instruction: bool
    # "baseline" = the parent's instruction text, unchanged.
    instruction_variant: Literal["baseline", "no_correction_clause", "neutral_correction"]
    description: str


Relation = Literal["more_than", "less_than"]


class HeldOut(BaseModel):
    """Phase 3B pre-registered held-out scenario metadata (one scenario = A/B variants + optional control).

    `asserted_relation` is the (false) relation the scenario's false-premise variants presuppose for
    subject vs object; `expected_relation` is the true relation from the controlled values. Controls carry the
    same scenario facts plus `asked_relation`, the relation their neutral question asks about.
    """

    model_config = ConfigDict(extra="forbid")

    held_out: Literal[True] = True
    scenario_id: str
    domain: str
    variant_type: Literal["direct_causal", "paraphrased_causal", "neutral_control"]
    control: bool
    metric: str
    subject: str
    subject_aliases: list[str] = Field(min_length=1)
    subject_value: float
    object: str
    object_aliases: list[str] = Field(min_length=1)
    object_value: float
    asserted_relation: Relation
    expected_relation: Relation
    asked_relation: Optional[Relation] = None
    # Controls only: the entity that correctly answers the neutral "Which of the two ..." question, and the
    # opposite entity. Used by the neutral-control answer check (stresslab.controls), not by grounding.py.
    expected_answer: Optional[str] = None
    expected_answer_aliases: list[str] = Field(default_factory=list)
    wrong_answer: Optional[str] = None
    wrong_answer_aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "HeldOut":
        if self.subject_value == self.object_value:
            raise ValueError("compared values must differ")
        true_rel = "more_than" if self.subject_value > self.object_value else "less_than"
        if self.expected_relation != true_rel:
            raise ValueError("expected_relation does not match the stored values")
        if self.asserted_relation == self.expected_relation:
            raise ValueError("asserted_relation must be the opposite of expected_relation (a false premise)")
        if self.control != (self.variant_type == "neutral_control"):
            raise ValueError("control flag and variant_type disagree")
        if self.control != (self.asked_relation is not None):
            raise ValueError("asked_relation is required for controls and only for controls")
        answer_fields = (self.expected_answer, self.wrong_answer, self.expected_answer_aliases, self.wrong_answer_aliases)
        if not self.control:
            if any(answer_fields):
                raise ValueError("expected/wrong answer fields are for controls only")
            return self
        if not all(answer_fields):
            raise ValueError("controls need expected_answer(_aliases) and wrong_answer(_aliases)")
        bigger, smaller = ((self.subject, self.object) if self.subject_value > self.object_value
                           else (self.object, self.subject))
        correct = bigger if self.asked_relation == "more_than" else smaller
        wrong = smaller if correct == bigger else bigger
        if (self.expected_answer, self.wrong_answer) != (correct, wrong):
            raise ValueError("expected_answer/wrong_answer do not follow from values and asked_relation")
        if self.expected_answer not in self.expected_answer_aliases or self.wrong_answer not in self.wrong_answer_aliases:
            raise ValueError("answer aliases must include the full entity name")
        return self


class TestCase(BaseModel):
    """One base test case loaded from data/test_cases/*.jsonl."""

    __test__ = False  # keep pytest from collecting this class
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    question: str
    context: Optional[str] = None
    # Phase 2: case subtype (e.g. "false_premise") and the instruction placed before the context.
    subtype: Optional[str] = None
    instruction: Optional[str] = None
    # Phase 3A: present only on derived cases (e.g. FG-013 reproduction variants).
    lineage: Optional[Lineage] = None
    # Phase 3B: present only on pre-registered held-out cases.
    heldout: Optional[HeldOut] = None
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class GenerationConfig(BaseModel):
    """Everything that controls decoding. Stored verbatim with every result."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=256, ge=1)
    do_sample: bool = False
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    # None = use the chat template default. True/False is passed to the template explicitly.
    enable_thinking: Optional[bool] = None


class ModelInfo(BaseModel):
    """Identity of the model actually used. Filled in by the adapter after loading."""

    adapter: str
    model_id: str
    requested_revision: Optional[str] = None
    resolved_revision: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    device: Optional[str] = None
    is_real_model: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class Generation(BaseModel):
    """Raw output of a single adapter call."""

    text: str
    raw_text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_s: Optional[float] = None
    peak_gpu_memory_gib: Optional[float] = None


class Reproducibility(BaseModel):
    runs: int = 1
    failures: int = 0
    rate: float = 0.0
    seeds: list[int] = Field(default_factory=list)


class TestResult(BaseModel):
    """One executed test. One JSON line in results.jsonl."""

    __test__ = False
    schema_version: str = SCHEMA_VERSION
    run_id: str
    test_id: str
    category: Category
    subtype: Optional[str] = None
    lineage: Optional[Lineage] = None
    # Phase 3B: present only on pre-registered held-out cases.
    heldout: Optional[HeldOut] = None
    model: str
    model_revision: Optional[str] = None
    timestamp: str = Field(default_factory=utc_now_iso)
    seed: int
    generation_config: GenerationConfig
    base_prompt: str
    mutated_prompt: Optional[str] = None
    mutation: Optional[dict[str, Any]] = None
    expected_behavior: dict[str, Any] = Field(default_factory=dict)
    response: str
    raw_response: Optional[str] = None
    usage: dict[str, Any] = Field(default_factory=dict)
    evaluator: Optional[str] = None
    checks: dict[str, Any] = Field(default_factory=dict)
    status: Status = Status.UNSCORED
    severity: Optional[Severity] = None
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    notes: Optional[str] = None


class RunMetadata(BaseModel):
    """Written once per run to metadata.json."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None
    command: str
    suite: Optional[str] = None
    test_suite_version: Optional[str] = None
    num_cases: int = 0
    seed: int
    generation_config: GenerationConfig
    model: Optional[ModelInfo] = None
    environment: dict[str, Any] = Field(default_factory=dict)
    stresslab_version: str
    notes: Optional[str] = None
