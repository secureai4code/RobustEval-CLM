import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Union

import torch
from huggingface_hub.constants import HF_HOME
from torch import dtype
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

from .base_model import BaseModel

# Configure vLLM environment for deterministic behavior before importing
# Disable V1 multiprocessing for reproducibility
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Disable P2P for stability on multi-GPU setups without NVLink
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# Inject noise method to vLLM Worker class BEFORE importing vLLM
from .vllm_noise_injector import inject_noise_method

inject_noise_method()

from vllm import LLM, SamplingParams


@dataclass
class GenerationStrategy:
    """Configuration for different generation strategies"""
    num_return_sequences: int = 1
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    num_beams: int = 10
    use_beam_search: bool = False


@dataclass
class QuantizationConfig:
    method: Literal["bnb", "gptq", "awq"] = "bnb"  # Quantization method
    bits: Literal[4, 8] = 8  # Bits for quantization
    compute_dtype: torch.dtype = torch.float16  # Computation dtype
    quant_type: str = "nf4"  # For bnb 4-bit: "nf4" or "fp4"
    dataset: str = "c4"  # For GPTQ calibration


@dataclass
class DynamicQuantizationConfig:
    """Configuration for dynamic quantization of Large Language Models.
    
    Attributes:
        bits: Currently only supports 8-bit for dynamic quantization
        dtype: Quantization dtype (only qint8 supported)
        qconfig_preset: Backend configuration ('fbgemm' for x86)
        quantize_embeddings: Whether to quantize embedding layers
        modules_to_quantize: Set of specific modules to quantize in transformer
    """
    bits: int = 8  # Fixed to 8 for dynamic quantization
    dtype: dtype = torch.qint8
    qconfig_preset: str = 'fbgemm'
    quantize_embeddings: bool = False  # Usually keep embeddings in full precision
    modules_to_quantize: Set[torch.nn.Module] = None

    def __post_init__(self):
        # Set default modules for transformer architecture
        if self.modules_to_quantize is None:
            self.modules_to_quantize = {
                torch.nn.Linear,  # For attention & feed-forward layers
                torch.nn.LayerNorm  # For layer normalization
            }
        
        self._validate()

    def _validate(self):
        """Validate the configuration parameters."""
        if self.bits != 8:
            raise ValueError("Dynamic quantization only supports 8-bit quantization")
        
        if self.dtype != torch.qint8:
            raise ValueError("Dynamic quantization only supports torch.qint8 dtype")
            
        if self.qconfig_preset != 'fbgemm':
            raise ValueError("LLM dynamic quantization only supports 'fbgemm' backend")

    def get_qconfig(self) -> torch.quantization.QConfig:
        """Get the PyTorch quantization config."""
        return torch.quantization.get_default_qconfig('fbgemm')

    def __str__(self) -> str:
        """String representation of the config."""
        return (
            f"DynamicQuantizationConfig(\n"
            f"  bits={self.bits},\n"
            f"  dtype={self.dtype},\n"
            f"  qconfig_preset='{self.qconfig_preset}',\n"
            f"  quantize_embeddings={self.quantize_embeddings},\n"
            f"  modules_to_quantize={self.modules_to_quantize}\n"
            f")"
        )


class CodeLLaMAModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()
        self.gen_config = kwargs.get('generation_config')

    def load(self) -> None:
        print("Now loading original LLM model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        if strategy.num_return_sequences == 1:
            # Use greedy decoding for single sequence
            print("Using greedy decoding")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=1,
                do_sample=False,
            )
        elif strategy.use_beam_search:
            # Use beam search for multiple sequences
            print("Using beam search")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                num_beams=strategy.num_beams,
                do_sample=False,
            )
        else:
            # Use temperature sampling for multiple sequences
            print("Using temperature sampling")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
                do_sample=True,
            )
    
    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text (no function extraction)."""
        return full_text[len(prompt):]

    def generate(
        self,
        prompt: str,
    ) -> Union[str, List[str]]:
        """Generate completion(s) for a given prompt."""
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], batch_size: int = 16, **kwargs) -> List[str]:
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        generation_config = self._get_generation_config(strategy)
        results = []

        num_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Batch generating"):
            batch = prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for i, output in enumerate(outputs):
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(self._extract_completion(full_text, batch[i]))

        return results


class DynamicQuantizedModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        # Extract quantization config and generation config from kwargs
        try:
            self.quant_config = DynamicQuantizationConfig(**kwargs.get('quant_config'))
        except TypeError:
            raise ValueError("Quantization parameters not provided")
        self.gen_config = kwargs.get('generation_config')
        
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        """Load tokenizer and dynamically quantized model"""
        print(f"Loading model for dynamic quantization ({self.quant_config.bits} bits)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model in FP32 on CPU for dynamic quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        ).cpu()

        # Prepare modules to quantize based on config
        modules_to_quantize = self.quant_config.modules_to_quantize
        if not self.quant_config.quantize_embeddings:
            # Exclude embedding layers if specified
            print("Excluding embedding layers from quantization")
            exclude_modules = {torch.nn.Embedding}
            modules_to_quantize = {m for m in modules_to_quantize if m not in exclude_modules}

        # Apply dynamic quantization
        print("Applying dynamic quantization...")
        try:
            self.model = torch.quantization.quantize_dynamic(
                model,
                modules_to_quantize,
                dtype=self.quant_config.dtype
            )
            print("Dynamic quantization complete")
        except Exception as e:
            raise RuntimeError(f"Dynamic quantization failed: {str(e)}")

        print(self.get_memory_stats())

    def get_memory_stats(self) -> dict:
        """Get CPU memory statistics for dynamically quantized model"""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "total_cpu_memory": memory_info.rss / 1024**3,  # GB
            "model_size": sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024**3,  # GB
            "quantization_type": "dynamic",
            "bits": self.quant_config.bits,
            "device": "cpu"  # Dynamic quantization is CPU-only
        }

    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        if strategy.num_return_sequences == 1:
            # Use greedy decoding for single sequence
            print("Using greedy decoding")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=1,
                do_sample=False,
            )
        elif strategy.use_beam_search:
            # Use beam search for multiple sequences
            print("Using beam search")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                num_beams=strategy.num_beams,
                do_sample=False,
            )
        else:
            # Use temperature sampling for multiple sequences
            print("Using temperature sampling")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
                do_sample=True,
            )
    
    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text (no function extraction)."""
        return full_text[len(prompt):]

    def generate(
        self,
        prompt: str,
    ) -> Union[str, List[str]]:
        """Generate completions using the quantized model on CPU"""
        # Reuse your existing generation code but ensure CPU operation
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt")  # Already on CPU
        generation_config = self._get_generation_config(strategy)
        
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation - implemented as sequential for CPU operation"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class StaticQuantizedModel(BaseModel):
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        # Extract quantization config and generation config from kwargs
        try:
            self.quant_config = QuantizationConfig(**kwargs.get('quant_config'))
        except TypeError:
            raise ValueError("Quantization parameters not provided")
        self.gen_config = kwargs.get('generation_config')
        super().__init__(model_path, **kwargs)

        # Initialize model with a static quantization configuration
        self.load()

    def load(self) -> None:
        """Load tokenizer and quantized model based on configuration"""

        assert self.quant_config.bits in [4, 8], "Bits must be 4 or 8"
        assert self.quant_config.quant_type in ["nf4", "fp4"], "Quant type must be 'nf4' or 'fp4'"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if self.quant_config.method == "bnb":
            print(f"Loading model with BitsAndBytes quantization ({self.quant_config.bits} bits)")
            self._load_bnb_model()
        elif self.quant_config.method == "gptq":
            self._load_gptq_model()
        elif self.quant_config.method == "awq":
            self._load_awq_model()
        else:
            raise ValueError(f"Unsupported quantization method: {self.quant_config.method}")

    def _load_bnb_model(self) -> None:
        """Load model with bitsandbytes quantization"""
        if self.quant_config.bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.quant_config.compute_dtype,
                bnb_4bit_quant_type=self.quant_config.quant_type
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=self.quant_config.compute_dtype,
            device_map="auto"
        )

        print(self.get_memory_stats())

    def _load_gptq_model(self) -> None:
        """Load model with GPTQ quantization"""
        from transformers import GPTQConfig

        gptq_config = GPTQConfig(
            bits=self.quant_config.bits,
            dataset=self.quant_config.dataset,
            use_exllama=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=gptq_config,
            device_map="auto"
        )

    def _load_awq_model(self) -> None:
        """Load model with AWQ quantization"""
        if self.quant_config.bits != 8:
            raise ValueError("AWQ currently only supports 8-bit quantization")
            
        from awq import AutoAWQForCausalLM
        
        self.model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            bits=self.quant_config.bits,
            device_map="auto"
        )

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }

    def _get_generation_config(self, strategy: GenerationStrategy) -> GenerationConfig:
        """Create generation configuration based on strategy"""
        if strategy.num_return_sequences == 1:
            # Use greedy decoding for single sequence
            # print("Using greedy decoding")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=1,
                do_sample=False,
            )
        elif strategy.use_beam_search:
            # Use beam search for multiple sequences
            print("Using beam search")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                num_beams=strategy.num_beams,
                do_sample=False,
            )
        else:
            # Use temperature sampling for multiple sequences
            print("Using temperature sampling")
            return GenerationConfig(
                max_length=strategy.max_length,
                num_return_sequences=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
                do_sample=True,
            )
    
    def _extract_completion(self, full_text: str, prompt: str) -> str:
        """Extract only the completion part from the generated text (no function extraction)."""
        return full_text[len(prompt):]

    def generate(
        self,
        prompt: str,
    ) -> Union[str, List[str]]:
        """Generate completion(s) for a given prompt."""
        # Start with default strategy and update with any provided kwargs
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = self._get_generation_config(strategy)
        
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode all sequences
        decoded_outputs = [
            self._extract_completion(
                self.tokenizer.decode(output, skip_special_tokens=True),
                prompt
            )
            for output in outputs
        ]

        # Return single string if num_return_sequences=1, otherwise list
        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], batch_size: int = 16, **kwargs) -> List[str]:
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        generation_config = self._get_generation_config(strategy)
        results = []

        num_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(prompts), batch_size), total=num_batches, desc="Batch generating"):
            batch = prompts[start:start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for i, output in enumerate(outputs):
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(self._extract_completion(full_text, batch[i]))

        return results


class StarCoderModel(BaseModel):
    def __init__(self, model_path: str = "bigcode/starcoder", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()
    
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class CodeGenModel(BaseModel):
    def __init__(self, model_path: str = "Salesforce/codegen-350M-mono", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class DeepSeekModel(BaseModel):
    def __init__(self, model_path: str = "deepseek-ai/deepseek-coder-1.3b-base", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class InCoderModel(BaseModel):
    def __init__(self, model_path: str = "facebook/incoder-1B", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    

class MagicCoderModel(BaseModel):
    def __init__(self, model_path: str = "ise-uiuc/Magicoder-CL-7B", **kwargs):
        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=kwargs.get('max_length', 512),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', None),
            num_return_sequences=kwargs.get('num_return_sequences', 1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class VLLMModel(BaseModel):
    """vLLM-based model implementation for high-throughput inference"""
    
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        super().__init__(model_path, **kwargs)
        self.gen_config = kwargs.get('generation_config')
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.85)
        self.max_model_len = kwargs.get('max_model_len', 4096)
        self.seed = kwargs.get('seed', 42)
        self.load()

    def load(self) -> None:
        """Load model using vLLM engine"""
        self.SamplingParams = SamplingParams

        print(f"Loading model with vLLM: {self.model_path}")
        
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.seed,
            trust_remote_code=True,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print("vLLM model loaded successfully")

    def _get_sampling_params(self, strategy: GenerationStrategy) -> 'SamplingParams':
        """Create vLLM sampling parameters based on strategy"""
        if strategy.num_return_sequences == 1:
            print("Using greedy decoding")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=1,
                temperature=0.0,
                top_p=1.0,
            )
        elif strategy.use_beam_search:
            print("Using beam search")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=strategy.num_return_sequences,
                best_of=strategy.num_beams,
                temperature=0.0,
                use_beam_search=True,
            )
        else:
            print("Using temperature sampling")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
            )

    def _extract_completion(self, completion: str, prompt: str) -> str:
        """Return completion as-is (no function extraction; done in attack_framework)."""
        return completion

    def generate(self, prompt: str) -> Union[str, List[str]]:
        """Generate completion(s) for a given prompt using vLLM."""
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        sampling_params = self._get_sampling_params(strategy)

        print(f"Generating with prompt: {prompt}")
        outputs = self.model.generate([prompt], sampling_params)
        print("Finished generation")

        decoded_outputs = []
        for output in outputs[0].outputs:
            completion = output.text
            decoded_outputs.append(self._extract_completion(completion, prompt))

        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Efficient batch generation using vLLM."""
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        sampling_params = self._get_sampling_params(strategy)

        outputs = self.model.generate(prompts, sampling_params)

        results = []
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0].text
            results.append(self._extract_completion(completion, prompt))

        return results


class VLLMQuantizedModel(BaseModel):
    """vLLM-based model with BitsAndBytes quantization support"""
    
    def __init__(self, model_path: str = "codellama/CodeLlama-7b-hf", **kwargs):
        try:
            self.quant_config = QuantizationConfig(**kwargs.get('quant_config', {}))
        except TypeError:
            raise ValueError("Quantization parameters not provided")
        
        self.gen_config = kwargs.get('generation_config')
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        # BnB inflight quantization dequantizes weights at runtime (for MoE models,
        # whole per-layer expert tensors) in transient buffers that vLLM's memory
        # budget does not account for, so leave extra headroom by default.
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.88)
        self.max_model_len = kwargs.get('max_model_len', 4096)
        self.seed = kwargs.get('seed', 42)

        super().__init__(model_path, **kwargs)
        self.load()

    def load(self) -> None:
        """Load model using vLLM with BNB quantization"""
        self.SamplingParams = SamplingParams

        if self.quant_config.method != "bnb":
            raise ValueError(
                f"VLLMQuantizedModel only supports BNB quantization, got: {self.quant_config.method}"
            )

        print(f"Loading model with vLLM + BNB quantization ({self.quant_config.bits} bits)")

        if self.quant_config.bits == 4:
            self._load_4bit()
        elif self.quant_config.bits == 8:
            self._load_8bit()
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quant_config.bits}")

        self.tokenizer = self.model.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("vLLM quantized model loaded successfully")
        print(self.get_memory_stats())

    def _load_4bit(self) -> None:
        """Load model with vLLM's native 4-bit inflight BNB quantization."""
        self.model = LLM(
            model=self.model_path,
            quantization="bitsandbytes",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.seed,
            trust_remote_code=True,
        )

    @staticmethod
    def _copy_missing_processor_files(model_path: str, cache_dir: str) -> None:
        """Copy preprocessor/processor configs from the original HF model snapshot.

        Multimodal models (e.g. Gemma-3) ship with preprocessor_config.json and
        processor_config.json that ``save_pretrained`` on the *model* object does
        not persist.  vLLM expects these files when it resolves the model, so we
        copy any that are missing from the original HF hub snapshot.
        """
        import shutil

        from huggingface_hub import snapshot_download

        extra_files = [
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.json",
        ]
        needed = [f for f in extra_files if not os.path.exists(os.path.join(cache_dir, f))]
        if not needed:
            return

        try:
            snapshot_dir = snapshot_download(model_path, allow_patterns=needed)
        except Exception:
            return

        for fname in needed:
            src = os.path.join(snapshot_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(cache_dir, fname))
                print(f"Copied {fname} from original model into cache")

    def _load_8bit(self) -> None:
        """Load 8-bit BNB model for vLLM.

        vLLM does not support 8-bit inflight quantization, so we:
        1. Quantize the model to 8-bit using HuggingFace Transformers + BNB
        2. Save the pre-quantized checkpoint under the Hugging Face cache root
        3. Free the HF model from GPU memory
        4. Copy any missing processor configs from the original model
        5. Load the pre-quantized checkpoint into vLLM
        """
        import gc

        cache_dir = os.path.join(
            HF_HOME, "vllm_bnb8_cache", self.model_path.replace("/", "_")
        )
        config_path = os.path.join(cache_dir, "config.json")

        if os.path.exists(config_path):
            print(f"Found cached 8-bit checkpoint at {cache_dir}, skipping quantization")
        else:
            print("Quantizing model to 8-bit with HuggingFace Transformers + BNB...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                torch_dtype=self.quant_config.compute_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            print(f"Saving 8-bit checkpoint to {cache_dir}...")
            os.makedirs(cache_dir, exist_ok=True)
            hf_model.save_pretrained(cache_dir)
            tokenizer.save_pretrained(cache_dir)

            del hf_model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Freed HuggingFace model from memory")

        self._copy_missing_processor_files(self.model_path, cache_dir)

        print("Loading 8-bit checkpoint into vLLM...")
        self.model = LLM(
            model=cache_dir,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.seed,
            trust_remote_code=True,
        )

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "quantization_method": self.quant_config.method,
            "quantization_bits": self.quant_config.bits
        }

    def _get_sampling_params(self, strategy: GenerationStrategy) -> 'SamplingParams':
        """Create vLLM sampling parameters based on strategy"""
        if strategy.num_return_sequences == 1:
            print("Using greedy decoding")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=1,
                temperature=0.0,
                top_p=1.0,
            )
        elif strategy.use_beam_search:
            print("Using beam search")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=strategy.num_return_sequences,
                best_of=strategy.num_beams,
                temperature=0.0,
                use_beam_search=True,
            )
        else:
            print("Using temperature sampling")
            return self.SamplingParams(
                max_tokens=strategy.max_length,
                n=strategy.num_return_sequences,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
            )

    def _extract_completion(self, completion: str, prompt: str) -> str:
        """Return completion as-is (no function extraction; done in attack_framework)."""
        return completion

    def generate(self, prompt: str) -> Union[str, List[str]]:
        """Generate completion(s) for a given prompt using vLLM with quantization."""
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        sampling_params = self._get_sampling_params(strategy)

        print(f"Generating with prompt: {prompt}")
        outputs = self.model.generate([prompt], sampling_params)
        print("Finished generation")

        decoded_outputs = []
        for output in outputs[0].outputs:
            completion = output.text
            decoded_outputs.append(self._extract_completion(completion, prompt))

        return decoded_outputs[0] if strategy.num_return_sequences == 1 else decoded_outputs

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Efficient batch generation using vLLM with quantization."""
        strategy = GenerationStrategy()
        if self.gen_config:
            strategy_dict = strategy.__dict__.copy()
            strategy_dict.update(self.gen_config)
            strategy = GenerationStrategy(**strategy_dict)

        sampling_params = self._get_sampling_params(strategy)

        outputs = self.model.generate(prompts, sampling_params)

        results = []
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0].text
            results.append(self._extract_completion(completion, prompt))

        return results
