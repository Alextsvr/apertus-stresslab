from stresslab.models import EchoAdapter, ModelAdapter
from stresslab.runner import run_cases, single_prompt_case
from stresslab.schemas import GenerationConfig, Generation, ModelInfo, Status
from stresslab.storage import read_metadata, read_results

FAKE_ENV = {"python_version": "test"}


class RecordingAdapter(ModelAdapter):
    """Mock adapter: records calls, fails on demand."""

    name = "recording"

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[str, int]] = []
        self.loaded = False
        self.fail_on = fail_on

    def load(self):
        self.loaded = True

    def info(self):
        return ModelInfo(adapter=self.name, model_id="mock/model", resolved_revision="abc123", is_real_model=False)

    def generate(self, prompt, config, seed):
        self.calls.append((prompt, seed))
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("simulated failure")
        return Generation(text=f"answer to: {prompt[-20:]}", input_tokens=10, output_tokens=config.max_new_tokens)


def test_runner_persists_one_record_per_case(tmp_path, cases, gen_config):
    adapter = RecordingAdapter()
    outcome = run_cases(cases, adapter, gen_config, seed=42, results_root=tmp_path, command="test",
                        suite="unit", environment=FAKE_ENV)
    assert adapter.loaded
    assert len(adapter.calls) == 2
    assert all(seed == 42 for _, seed in adapter.calls)

    stored = read_results(outcome.run_dir)
    assert [r.test_id for r in stored] == ["T-001", "T-002"]
    assert all(r.status == Status.UNSCORED for r in stored)
    assert stored[0].model == "mock/model"
    assert stored[0].model_revision == "abc123"
    assert stored[0].reproducibility.seeds == [42]
    assert stored[0].usage["hit_max_new_tokens"] is True
    # grounded case: context is part of the stored prompt
    assert "Maria Keller was born in Bern in 1981." in stored[1].base_prompt
    assert stored[1].expected_behavior == {"contains_all": ["Bern"]}


def test_runner_writes_complete_metadata(tmp_path, cases, gen_config):
    outcome = run_cases(cases, RecordingAdapter(), gen_config, seed=7, results_root=tmp_path,
                        command="test", suite="unit", environment=FAKE_ENV)
    meta = read_metadata(outcome.run_dir)
    assert meta.run_id == outcome.run_dir.name
    assert meta.num_cases == 2
    assert meta.seed == 7
    assert meta.model.model_id == "mock/model"
    assert meta.finished_at is not None
    assert meta.environment == FAKE_ENV
    assert meta.generation_config == gen_config


def test_runner_records_errors_and_continues(tmp_path, cases, gen_config):
    outcome = run_cases(cases, RecordingAdapter(fail_on="Maria"), gen_config, seed=1, results_root=tmp_path,
                        command="test", environment=FAKE_ENV)
    stored = read_results(outcome.run_dir)
    assert [r.status for r in stored] == [Status.UNSCORED, Status.ERROR]
    assert "simulated failure" in stored[1].error
    assert outcome.errors == 1


def test_echo_adapter_is_marked_fake(tmp_path, gen_config):
    outcome = run_cases([single_prompt_case("Hello there")], EchoAdapter(), gen_config, seed=3,
                        results_root=tmp_path, command="test", environment=FAKE_ENV)
    meta = read_metadata(outcome.run_dir)
    assert meta.model.is_real_model is False
    record = read_results(outcome.run_dir)[0]
    assert record.response == "[echo seed=3] Hello there"
    assert "NOT A REAL MODEL" in record.notes


def test_two_runs_get_separate_directories(tmp_path, cases):
    cfg = GenerationConfig(max_new_tokens=8)
    a = run_cases(cases, RecordingAdapter(), cfg, 1, tmp_path, "t", environment=FAKE_ENV)
    b = run_cases(cases, RecordingAdapter(), cfg, 1, tmp_path, "t", environment=FAKE_ENV)
    assert a.run_dir != b.run_dir
    assert len(read_results(a.run_dir)) == 2 and len(read_results(b.run_dir)) == 2
