"""Isolated Qwen adapter for one prospective study; frozen stresslab source stays untouched."""
from pathlib import Path
from types import SimpleNamespace
import time

from colab_quantization import require

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
OVERRIDES = {'max_new_tokens': 96, 'do_sample': False, 'repetition_penalty': 1.0}


class GenerateCapture:
    """Observe a single normal call without changing its arguments or result."""
    def __init__(self, original, expected_input=None):
        self.original = original
        self.expected_input = expected_input
        self.calls = 0
        self.input_ids = self.output_ids = self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        require(self.calls == 1 and not args and 'input_ids' in kwargs, 'Expected one normal generate call')
        self.input_ids = kwargs['input_ids'].detach().cpu().tolist()
        require(self.expected_input is None or self.input_ids==self.expected_input,
                'Generation inputs changed before forward')
        self.kwargs = {k:v for k,v in kwargs.items() if k not in {'input_ids','attention_mask'}}
        require(self.kwargs == OVERRIDES, 'Unexpected Qwen decoding overrides')
        result = self.original(**kwargs)
        self.output_ids = result.detach().cpu().tolist()
        return result


class QwenAdapter:
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot)
        self._model = None
        self._processor = None

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        require(self.snapshot.name == REVISION, 'Wrong Qwen snapshot')
        tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, device_map='cuda:0',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=False))
        require(type(model).__name__=='Qwen2ForCausalLM' and model.config.num_hidden_layers==28,
                'Unexpected Qwen architecture')
        require(model.generation_config.eos_token_id==[151645,151643] and
                tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643 and
                tokenizer.unk_token_id is None, 'Unexpected Qwen special tokens')
        model.eval()
        self._model = model
        self._processor = SimpleNamespace(tokenizer=tokenizer)

    def info(self):
        from stresslab.schemas import ModelInfo
        return ModelInfo(adapter='prospective_qwen', model_id=MODEL, requested_revision=REVISION,
                         resolved_revision=REVISION, dtype='float16_before_promotion', quantization='nf4_double',
                         device='cuda:0', extra={'trust_remote_code':False,'cpu_offload':False})

    def encode(self, prompt):
        return self._processor.tokenizer.apply_chat_template(
            [{'role':'user','content':prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt')

    def generate(self, prompt, config, seed):
        import torch
        from transformers import set_seed
        from stresslab.schemas import Generation
        require(config.max_new_tokens==96 and config.do_sample is False and config.enable_thinking is None,
                'Study decoding configuration changed')
        set_seed(seed)
        inputs = self.encode(prompt).to(self._model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(**inputs, **OVERRIDES)
        torch.cuda.synchronize()
        latency = time.perf_counter()-started
        new_ids = output[0, inputs['input_ids'].shape[-1]:]
        tokenizer = self._processor.tokenizer
        return Generation(text=tokenizer.decode(new_ids,skip_special_tokens=True).strip(),
                          raw_text=tokenizer.decode(new_ids,skip_special_tokens=False),
                          input_tokens=inputs['input_ids'].shape[-1], output_tokens=len(new_ids),
                          latency_s=latency, peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/2**30)
