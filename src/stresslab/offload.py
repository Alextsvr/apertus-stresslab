"""Explicit GPU + CPU offload planning for loading Apertus on small GPUs.

Why this exists: on an 8 GB laptop GPU the 4-bit Apertus 1.5 8B checkpoint does not fit
entirely in VRAM with `device_map="auto"`, and bitsandbytes refuses mixed GPU/CPU maps
unless `llm_int8_enable_fp32_cpu_offload=True` and an explicit device map are given.

How it works (deliberately simple and auditable):

1. Build the model skeleton on the meta device (no weights) and split it into placement
   blocks: whole decoder layers (never split), small leaf modules, and the multimodal
   tokenizers as single blocks.
2. Estimate each block's size *if placed on GPU* (4-bit/8-bit for quantizable Linear weights,
   bf16 otherwise, fp32 for modules that the model keeps in fp32) and *if placed on CPU*
   (unquantized: bitsandbytes does not quantize CPU-offloaded modules).
3. Force the image/audio tokenizers to CPU. They only run when image/audio inputs are
   given; StressLab sends text only, so they are never executed.
4. Fill the GPU up to `gpu_max_memory - headroom`, first-fit, taking blocks in order of
   "bytes saved from streaming per GPU byte" (cpu_bytes / gpu_bytes, ties keep module order).
   A 4-bit decoder layer would stream ~4x its GPU size from RAM on every forward pass, while
   an unquantized lm_head streams 1x, so quantized layers get GPU priority. Everything else
   goes to CPU RAM. Disk offload is never planned; exceeding the CPU budget is an error.
   `headroom` is the reserve, enlarged if needed so the largest streamed block fits.

CPU-offloaded blocks that *are* executed (e.g. some decoder layers) stay in system RAM and
are copied to the GPU for each forward pass by accelerate. That is correct but slow; the
plan says so in its metadata. This is not full-GPU inference.

The planning functions are pure Python (duck-typed modules), so they are unit-tested
without torch. Sizes are ESTIMATES, recorded as such.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

GIB = 1024**3

# Verified in the swiss-ai transformers fork (models/apertus1p5/modeling_apertus1p5.py):
# Apertus1p5ForConditionalGeneration -> model.{language_model, vision_tokenizer, audio_tokenizer}, lm_head.
# The two tokenizers are only called when pixel_values / input_features are passed.
DEFAULT_FORCE_CPU_MODULES = ("model.vision_tokenizer", "model.audio_tokenizer")

DEFAULT_GPU_MAX_MEMORY_GIB = 7.0
DEFAULT_GPU_RESERVE_GIB = 1.0  # activations, KV cache, CUDA context, streamed CPU blocks
DEFAULT_CPU_MAX_MEMORY_GIB = 24.0
DEFAULT_STREAM_MARGIN_GIB = 0.25  # activations on top of a block that is being streamed in

# Bytes per parameter on GPU for quantized nn.Linear weights (incl. ~3% scale overhead
# for nf4 + double quantization; int8 has one fp32 scale per row).
QUANTIZED_BYTES_PER_PARAM = {"4bit": 0.52, "8bit": 1.01}


class OffloadPlanError(RuntimeError):
    """The model cannot be placed within the GPU + CPU budgets without disk offload."""


@dataclass
class Block:
    name: str
    gpu_bytes: float
    cpu_bytes: float
    force_cpu: bool = False
    executed_for_text: bool = True


@dataclass
class OffloadPlan:
    device_map: dict[str, Any]
    gpu_device: int
    gpu_max_memory_gib: float
    gpu_reserve_gib: float
    gpu_headroom_gib: float
    cpu_max_memory_gib: float
    estimated_gpu_weights_gib: float
    estimated_cpu_weights_gib: float
    estimated_streamed_per_forward_gib: float
    gpu_blocks: list[str]
    cpu_blocks: list[str]
    cpu_blocks_executed_for_text: list[str]
    quantization: str
    disk_offload: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def full_gpu(self) -> bool:
        return not self.cpu_blocks

    @property
    def execution_summary(self) -> str:
        if self.full_gpu:
            return "full GPU"
        if self.cpu_blocks_executed_for_text:
            return "GPU + CPU offload (some executed layers are streamed from system RAM)"
        return "GPU for all text-path modules; unused multimodal tokenizers parked in CPU RAM"

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["full_gpu"] = self.full_gpu
        data["execution_summary"] = self.execution_summary
        data["sizes_are_estimates"] = True
        return data


def _place(blocks: list[Block], gpu_budget: float, gpu_device: int) -> dict[str, Any]:
    """Greedy first-fit, highest cpu_bytes/gpu_bytes first (ties keep module order)."""

    def priority(item: tuple[int, Block]) -> tuple[float, int]:
        idx, block = item
        ratio = block.cpu_bytes / block.gpu_bytes if block.gpu_bytes > 0 else float("inf")
        return (-ratio, idx)

    placement: dict[str, Any] = {}
    used = 0.0
    for _, block in sorted(enumerate(blocks), key=priority):
        if not block.force_cpu and used + block.gpu_bytes <= gpu_budget:
            placement[block.name] = gpu_device
            used += block.gpu_bytes
        else:
            placement[block.name] = "cpu"
    return placement


def plan_device_map(
    blocks: Iterable[Block],
    gpu_max_memory_gib: float = DEFAULT_GPU_MAX_MEMORY_GIB,
    gpu_reserve_gib: float = DEFAULT_GPU_RESERVE_GIB,
    cpu_max_memory_gib: float = DEFAULT_CPU_MAX_MEMORY_GIB,
    gpu_device: int = 0,
    quantization: str = "4bit",
    stream_margin_gib: float = DEFAULT_STREAM_MARGIN_GIB,
) -> OffloadPlan:
    """Place blocks on one GPU (resident weights) and CPU RAM (overflow), never on disk.

    GPU memory under the cap is split into resident weights and headroom. The headroom is
    `max(gpu_reserve, largest streamed block + stream_margin)`: a CPU-offloaded block that is
    executed is copied to the GPU during its forward pass, so it needs transient space. The
    plan is recomputed until the headroom covers the largest streamed block (fixed point).
    """
    if gpu_reserve_gib < 0 or gpu_max_memory_gib <= gpu_reserve_gib:
        raise ValueError("gpu_max_memory_gib must be larger than gpu_reserve_gib (both >= 0)")
    blocks = list(blocks)
    by_name = {b.name: b for b in blocks}

    headroom = gpu_reserve_gib * GIB
    for _ in range(len(blocks) + 1):
        placement = _place(blocks, gpu_max_memory_gib * GIB - headroom, gpu_device)
        streamed = [b for b in blocks if placement[b.name] == "cpu" and b.executed_for_text]
        largest = max((b.cpu_bytes for b in streamed), default=0.0)
        needed = max(gpu_reserve_gib * GIB, largest + stream_margin_gib * GIB) if streamed else headroom
        if needed <= headroom:
            break
        headroom = needed
    if headroom >= gpu_max_memory_gib * GIB:
        raise OffloadPlanError("GPU cap is too small to hold even one streamed block; raise --gpu-max-memory-gib.")

    gpu_used = sum(by_name[n].gpu_bytes for n, d in placement.items() if d != "cpu")
    cpu_used = sum(by_name[n].cpu_bytes for n, d in placement.items() if d == "cpu")
    if cpu_used > cpu_max_memory_gib * GIB:
        raise OffloadPlanError(
            f"Planned CPU offload needs ~{cpu_used / GIB:.1f} GiB but the CPU budget is "
            f"{cpu_max_memory_gib:.1f} GiB. Disk offload is intentionally not enabled; "
            "raise --cpu-max-memory-gib if you have the RAM, or use a larger GPU / Colab."
        )

    warnings: list[str] = []
    if streamed:
        streamed_gib = sum(b.cpu_bytes for b in streamed) / GIB
        warnings.append(
            f"{len(streamed)} executed block(s) (~{streamed_gib:.2f} GiB) stay in CPU RAM and are copied to the GPU "
            "on every forward pass (i.e. every generated token); generation is slower than full-GPU inference."
        )

    # Report everything in module order (easier to read and diff).
    return OffloadPlan(
        device_map={b.name: placement[b.name] for b in blocks},
        gpu_device=gpu_device,
        gpu_max_memory_gib=gpu_max_memory_gib,
        gpu_reserve_gib=gpu_reserve_gib,
        gpu_headroom_gib=round(headroom / GIB, 3),
        cpu_max_memory_gib=cpu_max_memory_gib,
        estimated_gpu_weights_gib=round(gpu_used / GIB, 3),
        estimated_cpu_weights_gib=round(cpu_used / GIB, 3),
        estimated_streamed_per_forward_gib=round(sum(b.cpu_bytes for b in streamed) / GIB, 3),
        gpu_blocks=[b.name for b in blocks if placement[b.name] != "cpu"],
        cpu_blocks=[b.name for b in blocks if placement[b.name] == "cpu"],
        cpu_blocks_executed_for_text=[b.name for b in streamed],
        quantization=quantization,
        warnings=warnings,
    )


# ------------------------------------------------------------------ skeleton -> blocks
def _has_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _has_component(name: str, components: Iterable[str]) -> bool:
    parts = name.split(".")
    return any(c in parts for c in components)


def collect_blocks(
    model: Any,
    quantization: str,
    is_linear: Callable[[Any], bool],
    no_split_classes: Iterable[str] = (),
    force_cpu_modules: Iterable[str] = DEFAULT_FORCE_CPU_MODULES,
    skip_quantization: Iterable[str] = (),
    keep_fp32_components: Iterable[str] = (),
    default_bytes: float = 2.0,
) -> list[Block]:
    """Split a (meta-device) module tree into placement blocks with size estimates.

    `model` only needs torch-like `named_children()`, `named_parameters()`,
    `named_buffers()`, `named_modules()` and `_parameters`; tensors only need `numel()`.
    """
    no_split = set(no_split_classes)
    force_cpu = tuple(force_cpu_modules)
    skip_q = tuple(skip_quantization)
    keep_fp32 = tuple(keep_fp32_components)
    q_bytes = QUANTIZED_BYTES_PER_PARAM.get(quantization)

    linear_modules = {name for name, mod in model.named_modules() if is_linear(mod)}
    blocks: list[Block] = []
    seen_tensors: set[int] = set()

    def tensor_bytes(full_name: str, tensor: Any, is_param: bool) -> tuple[float, float]:
        numel = float(tensor.numel())
        module_name, _, leaf = full_name.rpartition(".")
        if _has_component(full_name, keep_fp32):
            return numel * 4, numel * 4
        cpu = numel * default_bytes
        quantizable = (
            q_bytes is not None
            and is_param
            and leaf == "weight"
            and module_name in linear_modules
            and not _has_prefix(module_name, skip_q)
        )
        return (numel * q_bytes if quantizable else cpu), cpu

    def make_block(name: str, module: Any) -> Optional[Block]:
        gpu = cpu = 0.0
        tensors = [(n, t, True) for n, t in module.named_parameters()]
        tensors += [(n, t, False) for n, t in module.named_buffers()]
        for rel, tensor, is_param in tensors:
            if id(tensor) in seen_tensors:  # tied weights are counted once
                continue
            seen_tensors.add(id(tensor))
            g, c = tensor_bytes(f"{name}.{rel}" if name else rel, tensor, is_param)
            gpu += g
            cpu += c
        if not tensors:
            return None
        is_forced = _has_prefix(name, force_cpu)
        return Block(name=name, gpu_bytes=gpu, cpu_bytes=cpu, force_cpu=is_forced, executed_for_text=not is_forced)

    def visit(name: str, module: Any) -> None:
        children = list(module.named_children())
        atomic = (
            name in force_cpu
            or type(module).__name__ in no_split
            or not children
            or bool(getattr(module, "_parameters", {}))
        )
        if atomic and name:
            block = make_block(name, module)
            if block is not None:
                blocks.append(block)
            return
        for child_name, child in children:
            visit(f"{name}.{child_name}" if name else child_name, child)

    visit("", model)
    return blocks


def missing_modules(blocks: Iterable[Block], expected: Iterable[str] = DEFAULT_FORCE_CPU_MODULES) -> list[str]:
    """Expected force-CPU modules that do not exist in the model (reported as plan warnings)."""
    found = {b.name for b in blocks}
    return [m for m in expected if m not in found]


# ------------------------------------------------------------------ bitsandbytes config
def bnb_config_kwargs(quantization: str, compute_dtype: Any, cpu_offload: bool) -> dict[str, Any]:
    """Keyword arguments for transformers.BitsAndBytesConfig (kept torch-free for testing)."""
    if quantization == "4bit":
        kwargs: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": compute_dtype,
        }
    elif quantization == "8bit":
        kwargs = {"load_in_8bit": True}
    else:
        raise ValueError(f"No bitsandbytes config for quantization={quantization!r}")
    if cpu_offload:
        # Modules mapped to "cpu" are left unquantized and kept in system RAM.
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return kwargs
