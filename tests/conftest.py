import pytest

from stresslab.schemas import GenerationConfig, TestCase


@pytest.fixture
def gen_config() -> GenerationConfig:
    return GenerationConfig(max_new_tokens=32)


@pytest.fixture
def cases() -> list[TestCase]:
    return [
        TestCase(id="T-001", category="smoke", question="Say hi."),
        TestCase(
            id="T-002",
            category="factual_grounding",
            context="Maria Keller was born in Bern in 1981.",
            question="Where was Maria Keller born?",
            expected={"contains_all": ["Bern"]},
        ),
    ]
