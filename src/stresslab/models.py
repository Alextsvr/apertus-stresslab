"""Model adapters.

`ModelAdapter` is the only interface the runner knows about. Two implementations:

* `ApertusAdapter` - the real model via (forked) Hugging Face transformers. Heavy imports
  happen lazily inside `load()`, so the rest of the package works without torch installed.
* `EchoAdapter` - a deterministic stand-in for pipeline checks and unit tests. It is NOT a
  language model; its results are marked `is_real_model=False` and must never be reported
  as Apertus output.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal, Optional

from stresslab.config import DEFAULT_MODEL_ID, TRANSFORMERS_FORK_COMMIT
from stresslab.offload import (
    DEFAULT_CPU_MAX_MEMORY_GIB,
    DEFAULT_FORCE_CPU_MODULES,
    DEFAULT_GPU_MAX_MEMORY_GIB,
    DEFAULT_GPU_RESERVE_GIB,
    GIB,
    OffloadPlan,
    bnb_config_kwargs,
    collect_blocks,
    missing_modules,
    plan_device_map,
)
from stresslab.schemas import GenerationConfig, Generation, ModelInfo

Quantization = Literal["none", "8bit", "4bit"]
DtypeName = Literal["auto", "bfloat16", "float16", "float32"]


class ModelAdapter(ABC):
    name: str = "base"

    def load(self) -> None:  # noqa: B027 - optional hook
        """Load weights. Called once before the first `generate`."""

    @abstractmethod
    def info(self) -> ModelInfo:
        """Identity of the loaded model (call after `load`)."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig, seed: int) -> Generation:
        """Run one single-turn chat completion for `prompt`."""


class EchoAdapter(ModelAdapter):
    """Deterministic fake model: returns a fixed transformation of the prompt."""

    name = "echo"

    def info(self) -> ModelInfo:
        return ModelInfo(adapter=self.name, model_id="echo", is_real_model=False)

    def generate(self, prompt: str, config: GenerationConfig, seed: int) -> Generation:
        last_line = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        text = f"[echo seed={seed}] {last_line}"
        return Generation(text=text, raw_text=text, input_tokens=len(prompt.split()), output_tokens=len(text.split()))


class ApertusAdapter(ModelAdapter):
    """Official Apertus 1.5 checkpoint through transformers (swiss-ai fork)."""

    name = "apertus-transformers"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: Optional[str] = None,
        dtype: DtypeName = "auto",
        quantization: Quantization = "none",
        device_map: str = "auto",
        cpu_offload: bool = False,
        gpu_max_memory_gib: float = DEFAULT_GPU_MAX_MEMORY_GIB,
        gpu_reserve_gib: float = DEFAULT_GPU_RESERVE_GIB,
        cpu_max_memory_gib: float = DEFAULT_CPU_MAX_MEMORY_GIB,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dtype_name = dtype
        self.quantization = quantization
        self.device_map = device_map
        self.cpu_offload = cpu_offload
        self.gpu_max_memory_gib = gpu_max_memory_gib
        self.gpu_reserve_gib = gpu_reserve_gib
        self.cpu_max_memory_gib = cpu_max_memory_gib
        self._offload_plan: Optional[OffloadPlan] = None
        self._bnb_kwargs: Optional[dict] = None
        self._gpu_memory_after_load_gib: Optional[float] = None
        self._model = None
        self._processor = None
        self._torch_dtype = None
        self._resolved_revision: Optional[str] = None

    # ---------------------------------------------------------------- loading
    def _pick_dtype(self, torch):  # noqa: ANN001
        if self.dtype_name != "auto":
            return getattr(torch, self.dtype_name)
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.bfloat16  # CPU: halves memory vs float32; slow but workable

    def _resolve_revision(self) -> Optional[str]:
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415

            return HfApi().model_info(self.model_id, revision=self.revision).sha
        except Exception:  # noqa: BLE001 - metadata only, never fatal
            return None

    def _build_offload_plan(self, torch, model_cls, compute_dtype) -> OffloadPlan:  # noqa: ANN001
        """Plan an explicit GPU/CPU device map from a weightless (meta) skeleton of the model."""
        from accelerate import init_empty_weights  # noqa: PLC0415
        from transformers import AutoConfig  # noqa: PLC0415

        config = AutoConfig.from_pretrained(self.model_id, revision=self.revision)
        with init_empty_weights():
            try:
                skeleton = model_cls.from_config(config, dtype=compute_dtype)
            except TypeError:
                skeleton = model_cls.from_config(config)

        skip_quantization = {"lm_head"}
        try:
            from transformers.quantizers.base import get_keys_to_not_convert  # noqa: PLC0415

            skip_quantization.update(get_keys_to_not_convert(skeleton))
        except Exception:  # noqa: BLE001 - fall back to the conservative default
            pass
        keep_fp32 = set(getattr(skeleton, "_keep_in_fp32_modules_strict", None) or [])
        if compute_dtype == torch.float16:
            keep_fp32.update(getattr(skeleton, "_keep_in_fp32_modules", None) or [])

        blocks = collect_blocks(
            skeleton,
            quantization=self.quantization,
            is_linear=lambda m: isinstance(m, torch.nn.Linear),
            no_split_classes=getattr(skeleton, "_no_split_modules", None) or [],
            force_cpu_modules=DEFAULT_FORCE_CPU_MODULES,
            skip_quantization=sorted(skip_quantization),
            keep_fp32_components=sorted(keep_fp32),
        )
        plan = plan_device_map(
            blocks,
            gpu_max_memory_gib=self.gpu_max_memory_gib,
            gpu_reserve_gib=self.gpu_reserve_gib,
            cpu_max_memory_gib=self.cpu_max_memory_gib,
            gpu_device=0,
            quantization=self.quantization,
        )
        for name in missing_modules(blocks):
            plan.warnings.append(f"Expected multimodal module not found (not forced to CPU): {name}")
        total_gib = torch.cuda.get_device_properties(0).total_memory / GIB
        if self.gpu_max_memory_gib > total_gib:
            plan.warnings.append(
                f"--gpu-max-memory-gib {self.gpu_max_memory_gib} exceeds physical VRAM ({total_gib:.2f} GiB)."
            )
        del skeleton
        return plan

    @staticmethod
    def _print_plan(plan: OffloadPlan) -> None:
        import sys  # noqa: PLC0415

        print(
            f"[stresslab] offload plan: {plan.execution_summary}\n"
            f"[stresslab]   GPU cap {plan.gpu_max_memory_gib} GiB = ~{plan.estimated_gpu_weights_gib} GiB resident "
            f"weights ({len(plan.gpu_blocks)} blocks) + {plan.gpu_headroom_gib} GiB headroom\n"
            f"[stresslab]   CPU RAM: ~{plan.estimated_cpu_weights_gib} GiB: {', '.join(plan.cpu_blocks) or '-'}\n"
            f"[stresslab]   executed from CPU (copied to GPU each forward pass, "
            f"~{plan.estimated_streamed_per_forward_gib} GiB): {', '.join(plan.cpu_blocks_executed_for_text) or 'none'}\n"
            f"[stresslab]   (sizes are estimates)",
            file=sys.stderr,
        )
        for warning in plan.warnings:
            print(f"[stresslab]   WARNING: {warning}", file=sys.stderr)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            import transformers  # noqa: PLC0415
            from transformers import AutoProcessor  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Inference dependencies are missing. Install them with "
                "`pip install -r requirements-inference.txt` (see README)."
            ) from exc
        try:
            from transformers import AutoModelForMultimodalLM  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"Installed transformers {transformers.__version__} does not provide AutoModelForMultimodalLM. "
                "Apertus 1.5 needs the swiss-ai transformers fork pinned at commit "
                f"{TRANSFORMERS_FORK_COMMIT}. Run `pip install -r requirements-inference.txt`."
            ) from exc

        self._torch_dtype = self._pick_dtype(torch)
        kwargs: dict = {"device_map": self.device_map, "revision": self.revision}
        if self.quantization == "none":
            kwargs["dtype"] = self._torch_dtype
        else:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            compute_dtype = self._torch_dtype if self._torch_dtype != torch.float32 else torch.float16
            self._bnb_kwargs = bnb_config_kwargs(self.quantization, compute_dtype, self.cpu_offload)
            kwargs["quantization_config"] = BitsAndBytesConfig(**self._bnb_kwargs)
            kwargs["dtype"] = compute_dtype

        if self.cpu_offload:
            if not torch.cuda.is_available():
                raise RuntimeError("--cpu-offload needs a CUDA GPU (it splits the model between GPU and CPU RAM).")
            self._offload_plan = self._build_offload_plan(torch, AutoModelForMultimodalLM, kwargs["dtype"])
            self._print_plan(self._offload_plan)
            kwargs["device_map"] = dict(self._offload_plan.device_map)

        self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **kwargs).eval()
        self._resolved_revision = self._resolve_revision() or getattr(self._model.config, "_commit_hash", None)
        if torch.cuda.is_available():
            self._gpu_memory_after_load_gib = round(torch.cuda.memory_allocated() / GIB, 3)

    def _input_device(self):  # noqa: ANN202
        # With offloading, accelerate executes every module on the main GPU; feed inputs there.
        if self._offload_plan is not None:
            return f"cuda:{self._offload_plan.gpu_device}"
        return self._model.device

    # -------------------------------------------------------------- metadata
    def info(self) -> ModelInfo:
        device = None
        if self._model is not None:
            device = str(getattr(self._model, "device", None))
            device_map = getattr(self._model, "hf_device_map", None)
        else:
            device_map = None
        plan = self._offload_plan
        if plan is not None and not plan.full_gpu:
            device = f"cuda:{plan.gpu_device} + cpu offload"
        bnb = {k: (str(v).replace("torch.", "") if k.endswith("dtype") else v) for k, v in (self._bnb_kwargs or {}).items()}
        return ModelInfo(
            adapter=self.name,
            model_id=self.model_id,
            requested_revision=self.revision,
            resolved_revision=self._resolved_revision,
            dtype=str(self._torch_dtype).replace("torch.", "") if self._torch_dtype is not None else self.dtype_name,
            quantization=self.quantization,
            device=device,
            is_real_model=True,
            extra={
                "model_class": type(self._model).__name__ if self._model is not None else None,
                "device_map_devices": sorted({str(v) for v in device_map.values()}) if device_map else None,
                "cpu_offload": self.cpu_offload,
                "execution": plan.execution_summary if plan is not None else None,
                "offload_plan": plan.to_metadata() if plan is not None else None,
                "hf_device_map": {k: str(v) for k, v in device_map.items()} if device_map else None,
                "bitsandbytes_config": bnb or None,
                "gpu_memory_after_load_gib": self._gpu_memory_after_load_gib,
            },
        )

    # ------------------------------------------------------------ generation
    def generate(self, prompt: str, config: GenerationConfig, seed: int) -> Generation:
        self.load()
        import torch  # noqa: PLC0415
        from transformers import set_seed  # noqa: PLC0415

        set_seed(seed)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        template_kwargs = {}
        if config.enable_thinking is not None:
            template_kwargs["enable_thinking"] = config.enable_thinking
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self._input_device())

        gen_kwargs: dict = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample}
        if config.do_sample:
            if config.temperature is not None:
                gen_kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                gen_kwargs["top_p"] = config.top_p

        input_len = inputs["input_ids"].shape[-1]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - start

        new_tokens = output_ids[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        raw = self._processor.decode(new_tokens, skip_special_tokens=False)
        return Generation(
            text=text,
            raw_text=raw,
            input_tokens=int(input_len),
            output_tokens=int(new_tokens.shape[-1]),
            latency_s=round(latency, 3),
            peak_gpu_memory_gib=round(torch.cuda.max_memory_allocated() / GIB, 3) if torch.cuda.is_available() else None,
        )


def build_adapter(
    adapter: str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: Optional[str] = None,
    dtype: DtypeName = "auto",
    quantization: Quantization = "none",
    cpu_offload: bool = False,
    gpu_max_memory_gib: float = DEFAULT_GPU_MAX_MEMORY_GIB,
    gpu_reserve_gib: float = DEFAULT_GPU_RESERVE_GIB,
    cpu_max_memory_gib: float = DEFAULT_CPU_MAX_MEMORY_GIB,
) -> ModelAdapter:
    if adapter == "echo":
        return EchoAdapter()
    if adapter == "apertus":
        return ApertusAdapter(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            quantization=quantization,
            cpu_offload=cpu_offload,
            gpu_max_memory_gib=gpu_max_memory_gib,
            gpu_reserve_gib=gpu_reserve_gib,
            cpu_max_memory_gib=cpu_max_memory_gib,
        )
    raise ValueError(f"Unknown adapter {adapter!r}")
