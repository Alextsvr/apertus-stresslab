import json

import pytest

from stresslab.cases import build_prompt, load_cases, suite_path
from stresslab.cli import main
from stresslab.config import DEFAULT_CASES_DIR
from stresslab.environment import collect_environment
from stresslab.models import ApertusAdapter
from stresslab.schemas import TestCase
from stresslab.storage import read_metadata, read_results


def test_bundled_smoke_suite_is_valid():
    cases = load_cases(suite_path("smoke", DEFAULT_CASES_DIR))
    assert 3 <= len(cases) <= 5
    assert len({c.id for c in cases}) == len(cases)


def test_load_cases_rejects_duplicates(tmp_path):
    line = json.dumps({"id": "D-1", "category": "smoke", "question": "q"})
    path = tmp_path / "dup.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_load_cases_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "A", "category": "smoke", "question": "q"}\n{"id": "B"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_cases(path)


def test_build_prompt_is_deterministic():
    case = TestCase(id="X", category="factual_grounding", context=" C. ", question=" Q? ")
    assert build_prompt(case) == build_prompt(case)
    assert "Context:\nC.\n\nQuestion:\nQ?" in build_prompt(case)
    assert build_prompt(TestCase(id="Y", category="smoke", question=" hi ")) == "hi"


def test_unknown_suite_lists_available():
    with pytest.raises(FileNotFoundError, match="smoke"):
        suite_path("does_not_exist", DEFAULT_CASES_DIR)


def test_environment_never_raises():
    env = collect_environment()
    assert env["python_version"]
    assert "packages" in env and "torch" in env


def test_cli_infer_with_echo_adapter(tmp_path, capsys):
    code = main(["infer", "--adapter", "echo", "--prompt", "Ping", "--seed", "5", "--results-dir", str(tmp_path)])
    assert code == 0
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    record = read_results(run_dirs[0])[0]
    assert record.response == "[echo seed=5] Ping"
    assert read_metadata(run_dirs[0]).model.adapter == "echo"
    assert "Stored 1 record" in capsys.readouterr().out


def test_cli_run_smoke_with_echo_adapter(tmp_path):
    code = main(["run", "--suite", "smoke", "--adapter", "echo", "--limit", "2", "--results-dir", str(tmp_path)])
    assert code == 0
    run_dir = next(tmp_path.iterdir())
    assert len(read_results(run_dir)) == 2
    assert read_metadata(run_dir).suite == "smoke"


def test_cli_env(capsys):
    assert main(["env"]) == 0
    assert "python_version" in json.loads(capsys.readouterr().out)


def test_apertus_adapter_construction_is_lazy():
    # Must not import torch or download anything until load()/generate() is called.
    adapter = ApertusAdapter(quantization="4bit")
    info = adapter.info()
    assert info.model_id == "swiss-ai/Apertus-v1.5-8B"
    assert info.quantization == "4bit"
    assert info.is_real_model is True
