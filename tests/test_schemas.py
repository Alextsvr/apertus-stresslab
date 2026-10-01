import json

import pytest
from pydantic import ValidationError

from stresslab.schemas import GenerationConfig, RunMetadata, Status, TestCase, TestResult


def test_test_case_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TestCase(id="X", category="smoke", question="q", unexpected="boom")


def test_test_case_rejects_unknown_category():
    with pytest.raises(ValidationError):
        TestCase(id="X", category="not_a_category", question="q")


def test_generation_config_validation():
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValidationError):
        GenerationConfig(top_p=1.5)


def test_result_roundtrip_json(gen_config):
    r = TestResult(
        run_id="2026-10-01_120000",
        test_id="T-001",
        category="smoke",
        model="swiss-ai/Apertus-v1.5-8B",
        seed=42,
        generation_config=gen_config,
        base_prompt="Say hi.",
        response="Hi.",
    )
    data = json.loads(r.model_dump_json())
    assert data["status"] == "UNSCORED"
    assert data["generation_config"]["max_new_tokens"] == 32
    assert data["reproducibility"] == {"runs": 1, "failures": 0, "rate": 0.0, "seeds": []}
    again = TestResult.model_validate(data)
    assert again == r
    assert again.status is Status.UNSCORED


def test_run_metadata_roundtrip(gen_config):
    m = RunMetadata(run_id="r", command="cmd", seed=1, generation_config=gen_config,
                    stresslab_version="0.1.0", environment={"python_version": "3.11"})
    assert RunMetadata.model_validate_json(m.model_dump_json()) == m
