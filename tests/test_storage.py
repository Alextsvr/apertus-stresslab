import json
from datetime import datetime

from stresslab.schemas import RunMetadata, TestResult
from stresslab.storage import (
    RESULTS_FILE,
    append_result,
    create_run_dir,
    new_run_id,
    read_metadata,
    read_results,
    write_metadata,
)


def _result(run_id: str, test_id: str, gen_config) -> TestResult:
    return TestResult(run_id=run_id, test_id=test_id, category="smoke", model="m", seed=42,
                      generation_config=gen_config, base_prompt="p", response="Grüezi – ✓")


def test_run_id_format():
    assert new_run_id(datetime(2026, 10, 1, 15, 30, 0)) == "2026-10-01_153000"


def test_create_run_dir_never_reuses(tmp_path):
    a = create_run_dir(tmp_path, "2026-10-01_153000")
    b = create_run_dir(tmp_path, "2026-10-01_153000")
    c = create_run_dir(tmp_path, "2026-10-01_153000")
    assert a.name == "2026-10-01_153000"
    assert b.name == "2026-10-01_153000_2"
    assert c.name == "2026-10-01_153000_3"
    assert len({a, b, c}) == 3


def test_jsonl_append_and_read(tmp_path, gen_config):
    run_dir = create_run_dir(tmp_path)
    append_result(run_dir, _result(run_dir.name, "A", gen_config))
    append_result(run_dir, _result(run_dir.name, "B", gen_config))
    lines = (run_dir / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["run_id"] == run_dir.name for line in lines)
    results = read_results(run_dir)
    assert [r.test_id for r in results] == ["A", "B"]
    assert results[0].response == "Grüezi – ✓"  # non-ASCII survives


def test_metadata_write_read(tmp_path, gen_config):
    run_dir = create_run_dir(tmp_path)
    meta = RunMetadata(run_id=run_dir.name, command="test", seed=7, generation_config=gen_config,
                       stresslab_version="0.1.0")
    write_metadata(run_dir, meta)
    assert read_metadata(run_dir) == meta


def test_read_results_missing_file(tmp_path):
    assert read_results(tmp_path) == []
