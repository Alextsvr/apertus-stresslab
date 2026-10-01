"""Offload planning and bitsandbytes config generation (pure Python; no torch, no model)."""

import pytest

from stresslab.cli import build_parser
from stresslab.models import ApertusAdapter, build_adapter
from stresslab.offload import (
    GIB,
    Block,
    OffloadPlanError,
    bnb_config_kwargs,
    collect_blocks,
    missing_modules,
    plan_device_map,
)

# ------------------------------------------------------------------ fake torch-like modules


class FakeTensor:
    def __init__(self, numel: int):
        self._n = numel

    def numel(self) -> int:
        return self._n


class FakeModule:
    def __init__(self, params=None, buffers=None, **children):
        self._parameters = dict(params or {})
        self._buffers = dict(buffers or {})
        self._children = children

    def named_children(self):
        return list(self._children.items())

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def named_parameters(self, prefix=""):
        for name, mod in self.named_modules(prefix):
            for p, t in mod._parameters.items():
                yield (f"{name}.{p}" if name else p), t

    def named_buffers(self, prefix=""):
        for name, mod in self.named_modules(prefix):
            for b, t in mod._buffers.items():
                yield (f"{name}.{b}" if name else b), t


class Linear(FakeModule):
    def __init__(self, n):
        super().__init__(params={"weight": FakeTensor(n)})


class DecoderLayer(FakeModule):
    pass


class Norm(FakeModule):
    def __init__(self, n=8):
        super().__init__(params={"weight": FakeTensor(n)})


def fake_apertus(num_layers=4, linear=1_000_000, vocab=2_000_000, tokenizer=500_000):
    """Mirrors Apertus1p5ForConditionalGeneration's module layout at toy scale."""
    layers = FakeModule(**{
        str(i): DecoderLayer(q_proj=Linear(linear), down_proj=Linear(linear), attention_layernorm=Norm())
        for i in range(num_layers)
    })
    language_model = FakeModule(
        embed_tokens=FakeModule(params={"weight": FakeTensor(vocab)}),
        layers=layers,
        norm=Norm(),
        rotary_emb=FakeModule(buffers={"inv_freq": FakeTensor(64)}),
    )
    model = FakeModule(
        language_model=language_model,
        vision_tokenizer=FakeModule(encoder=Linear(tokenizer)),
        audio_tokenizer=FakeModule(encoder=Linear(tokenizer)),
    )
    return FakeModule(model=model, lm_head=Linear(vocab))


def blocks_for(model, quantization="4bit"):
    return collect_blocks(
        model,
        quantization=quantization,
        is_linear=lambda m: isinstance(m, Linear),
        no_split_classes=["DecoderLayer"],
        skip_quantization=["lm_head"],
        keep_fp32_components=["vision_tokenizer", "audio_tokenizer"],
    )


# ------------------------------------------------------------------ bitsandbytes kwargs


def test_bnb_4bit_with_cpu_offload():
    kwargs = bnb_config_kwargs("4bit", compute_dtype="bf16", cpu_offload=True)
    assert kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
        "llm_int8_enable_fp32_cpu_offload": True,
    }


def test_bnb_4bit_without_offload_has_no_offload_flag():
    assert "llm_int8_enable_fp32_cpu_offload" not in bnb_config_kwargs("4bit", "bf16", cpu_offload=False)


def test_bnb_8bit_and_invalid():
    assert bnb_config_kwargs("8bit", None, cpu_offload=True) == {
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": True,
    }
    with pytest.raises(ValueError):
        bnb_config_kwargs("none", None, cpu_offload=False)


# ------------------------------------------------------------------ block collection


def test_collect_blocks_granularity_and_sizes():
    blocks = {b.name: b for b in blocks_for(fake_apertus())}
    # decoder layers are atomic; multimodal tokenizers are single forced-CPU blocks
    assert "model.language_model.layers.0" in blocks
    assert "model.language_model.layers.0.q_proj" not in blocks
    assert blocks["model.vision_tokenizer"].force_cpu and not blocks["model.vision_tokenizer"].executed_for_text
    assert blocks["model.audio_tokenizer"].force_cpu
    assert not blocks["lm_head"].force_cpu

    layer = blocks["model.language_model.layers.0"]
    assert layer.gpu_bytes == pytest.approx(2 * 1_000_000 * 0.52 + 8 * 2)  # 2 nf4 linears + bf16 norm
    assert layer.cpu_bytes == pytest.approx(2 * 1_000_000 * 2 + 8 * 2)  # unquantized on CPU
    assert blocks["lm_head"].gpu_bytes == 2_000_000 * 2  # lm_head is never quantized
    assert blocks["model.vision_tokenizer"].cpu_bytes == 500_000 * 4  # kept in fp32
    assert blocks["model.language_model.rotary_emb"].gpu_bytes == 64 * 2  # buffers are counted


def test_collect_blocks_without_quantization_is_bf16():
    blocks = {b.name: b for b in blocks_for(fake_apertus(), quantization="none")}
    layer = blocks["model.language_model.layers.0"]
    assert layer.gpu_bytes == layer.cpu_bytes


def test_tied_tensors_counted_once():
    shared = FakeTensor(1000)
    model = FakeModule(a=FakeModule(params={"weight": shared}), b=FakeModule(params={"weight": shared}))
    blocks = collect_blocks(model, "4bit", is_linear=lambda m: False, force_cpu_modules=())
    assert sum(b.gpu_bytes for b in blocks) == 1000 * 2


def test_missing_force_cpu_modules_reported():
    model = FakeModule(model=FakeModule(language_model=Norm()), lm_head=Linear(10))
    assert missing_modules(blocks_for(model)) == ["model.vision_tokenizer", "model.audio_tokenizer"]


# ------------------------------------------------------------------ planning


def test_plan_all_text_modules_fit_on_gpu():
    plan = plan_device_map(blocks_for(fake_apertus()), gpu_max_memory_gib=7.0, gpu_reserve_gib=1.0)
    assert plan.device_map["model.vision_tokenizer"] == "cpu"
    assert plan.device_map["model.audio_tokenizer"] == "cpu"
    assert plan.device_map["lm_head"] == 0
    assert all(plan.device_map[f"model.language_model.layers.{i}"] == 0 for i in range(4))
    assert plan.cpu_blocks_executed_for_text == []
    assert not plan.full_gpu  # tokenizers are on CPU, so never claim full-GPU
    assert "unused multimodal" in plan.execution_summary
    assert plan.disk_offload is False
    assert "disk" not in plan.device_map.values()


def test_plan_overflow_goes_to_cpu_not_disk():
    # Toy model scaled so that only part of it fits under a 2 GiB cap.
    model = fake_apertus(num_layers=8, linear=200_000_000, vocab=150_000_000)
    plan = plan_device_map(blocks_for(model), gpu_max_memory_gib=2.0, gpu_reserve_gib=0.5)
    assert plan.estimated_gpu_weights_gib + plan.gpu_headroom_gib <= 2.0
    assert plan.gpu_headroom_gib > 0.5  # grew to fit the largest streamed (unquantized) layer
    assert plan.gpu_blocks, "expected part of the model on the GPU"
    executed_on_cpu = plan.cpu_blocks_executed_for_text
    assert executed_on_cpu, "expected some decoder layers to be offloaded"
    assert set(plan.device_map.values()) <= {0, "cpu"}
    assert "copied to the GPU on every forward pass" in plan.warnings[0]
    assert "CPU offload" in plan.execution_summary
    meta = plan.to_metadata()
    assert meta["full_gpu"] is False and meta["sizes_are_estimates"] is True
    assert meta["device_map"] == plan.device_map


def test_plan_first_fit_keeps_later_small_blocks_on_gpu():
    blocks = [
        Block("big", gpu_bytes=1.0 * GIB, cpu_bytes=1.0 * GIB),
        Block("too_big", gpu_bytes=1.5 * GIB, cpu_bytes=1.5 * GIB),
        Block("small", gpu_bytes=0.05 * GIB, cpu_bytes=0.05 * GIB),
    ]
    plan = plan_device_map(blocks, gpu_max_memory_gib=3.0, gpu_reserve_gib=1.0)
    assert plan.device_map == {"big": 0, "too_big": "cpu", "small": 0}


def test_plan_raises_instead_of_disk_offload():
    blocks = [Block("huge", gpu_bytes=10 * GIB, cpu_bytes=30 * GIB, force_cpu=True, executed_for_text=False)]
    with pytest.raises(OffloadPlanError, match="Disk offload is intentionally not enabled"):
        plan_device_map(blocks, gpu_max_memory_gib=7.0, gpu_reserve_gib=1.0, cpu_max_memory_gib=24.0)


def test_plan_raises_when_streamed_block_cannot_fit_on_gpu():
    blocks = [Block("huge_layer", gpu_bytes=10 * GIB, cpu_bytes=10 * GIB)]
    with pytest.raises(OffloadPlanError, match="GPU cap is too small"):
        plan_device_map(blocks, gpu_max_memory_gib=7.0, gpu_reserve_gib=1.0)


def test_plan_rejects_bad_budgets():
    with pytest.raises(ValueError):
        plan_device_map([], gpu_max_memory_gib=1.0, gpu_reserve_gib=1.0)


def test_plan_headroom_grows_to_fit_largest_streamed_block():
    # The 2 GiB head must be copied to the GPU when executed, so 1 GiB reserve is not enough.
    blocks = [
        Block("layer", gpu_bytes=3.0 * GIB, cpu_bytes=11.0 * GIB),
        Block("embed", gpu_bytes=2.0 * GIB, cpu_bytes=2.0 * GIB),
        Block("lm_head", gpu_bytes=2.0 * GIB, cpu_bytes=2.0 * GIB),
    ]
    plan = plan_device_map(blocks, gpu_max_memory_gib=7.0, gpu_reserve_gib=1.0, stream_margin_gib=0.25)
    assert plan.gpu_headroom_gib == pytest.approx(2.25)
    # weights budget = 7 - 2.25 = 4.75 GiB -> only the layer stays resident
    assert plan.device_map == {"layer": 0, "embed": "cpu", "lm_head": "cpu"}
    assert plan.estimated_gpu_weights_gib + plan.gpu_headroom_gib <= 7.0
    assert plan.estimated_streamed_per_forward_gib == pytest.approx(4.0)


def test_plan_headroom_stays_at_reserve_without_streaming():
    plan = plan_device_map(blocks_for(fake_apertus()), gpu_max_memory_gib=7.0, gpu_reserve_gib=1.0)
    assert plan.gpu_headroom_gib == 1.0
    assert plan.estimated_streamed_per_forward_gib == 0
    assert plan.warnings == []


# ------------------------------------------------------------------ CLI / adapter wiring


def test_cli_parses_offload_flags():
    args = build_parser().parse_args(
        ["infer", "--prompt", "x", "--quantization", "4bit", "--cpu-offload", "--gpu-max-memory-gib", "6.5"]
    )
    assert args.cpu_offload is True
    assert args.gpu_max_memory_gib == 6.5
    assert args.gpu_reserve_gib == 1.0
    assert args.cpu_max_memory_gib == 24.0


def test_cli_defaults_keep_offload_off():
    args = build_parser().parse_args(["run", "--suite", "smoke"])
    assert args.cpu_offload is False


def test_build_adapter_passes_offload_settings():
    adapter = build_adapter("apertus", quantization="4bit", cpu_offload=True, gpu_max_memory_gib=6.0)
    assert isinstance(adapter, ApertusAdapter)
    assert adapter.cpu_offload is True and adapter.gpu_max_memory_gib == 6.0
    info = adapter.info()  # before load: no plan yet, no fake device claims
    assert info.extra["cpu_offload"] is True
    assert info.extra["offload_plan"] is None
    assert info.device is None


def test_plan_prefers_quantized_layers_over_unquantized_head():
    # A 4-bit layer saves ~4x streaming bytes per GPU byte; the bf16 head only 1x.
    blocks = [
        Block("embed", gpu_bytes=1.0 * GIB, cpu_bytes=1.0 * GIB),
        Block("layer", gpu_bytes=0.5 * GIB, cpu_bytes=1.9 * GIB),
        Block("lm_head", gpu_bytes=1.0 * GIB, cpu_bytes=1.0 * GIB),
    ]
    plan = plan_device_map(blocks, gpu_max_memory_gib=2.9, gpu_reserve_gib=1.0)
    # headroom = 1.0 GiB head + 0.25 margin -> 1.65 GiB for weights: layer first, then embed
    assert plan.device_map == {"embed": 0, "layer": 0, "lm_head": "cpu"}
    assert list(plan.device_map) == ["embed", "layer", "lm_head"]  # module order preserved
    assert plan.cpu_blocks_executed_for_text == ["lm_head"]
