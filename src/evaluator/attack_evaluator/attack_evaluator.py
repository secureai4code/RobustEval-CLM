"""Main attack evaluator class."""

import json
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Tuple, Union

import fire

from src.core.models import Models
from src.core.models.model_config import ModelConfigManager
from src.evaluator.attack_evaluator.attack_config import (
    AttackConfig,
    EvaluationConfig,
    GenerationConfig,
    QuantizationConfig,
)
from src.evaluator.attack_evaluator.attack_registry import AttackRegistry
from src.evaluator.attack_evaluator.framework.attack_framework import AttackFramework
from src.evaluator.utils import visualizer


class AttackEvaluator:
    """Main class for conducting adversarial attacks on code generation models."""
    
    def __init__(self):
        """Initialize the attack evaluator."""
        self.config_manager = ModelConfigManager()
    
    def evaluate(
        self,
        model_path: str,
        model_type: str = "codellama",
        quantized_type: Optional[str] = None,
        dataset: str = "mbpp",
        attack_method: str = "synonym",
        save_prompts: Optional[str] = None,
        save_results: Optional[str] = None,
        visualization: bool = False,
        # Attack parameters
        replacement_probability: float = 0.15,
        max_synonyms: int = 3,
        char_change_probability: float = 0.5,
        max_char_changes: int = 15,
        translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        attack_model: str = "gpt-4o",
        attack_type: str = "paraphrase",
        adv_temperature: float = 0.7,
        adv_max_tokens: int = 150,
        api_path: str = "",
        input_type: str = None,
        noise_type: str = "gaussian",
        noise_level: float = 1e-3,
        seed: Optional[int] = None,
        # Quantization parameters
        quant_method: Literal["bnb", "gptq", "awq"] = "bnb",
        quant_bits: Literal[4, 8] = 8,
        quant_type: Literal["nf4", "fp4"] = "nf4",
        quantize_embeddings: bool = False,
        # Generation parameters
        num_return_sequences: int = 1,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_beams: int = 10,
        use_beam_search: bool = False,
        # VLLM parameters
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        # Other parameters
        gen_ori: bool = False,
        original_results: str = None,
    ) -> Dict[str, Any]:
        """
        Run adversarial attack evaluation on code generation models.
        
        Args:
            # Base parameters
            model_path: Path to the model,
            model_type: Type of model (codellama, starcoder, etc.)
            quantized_type: Type of quantization (None, "static", or "dynamic").
            dataset: Dataset to use, choices=["mbpp", "humaneval"].
            attack_method: Type of attack, choices=["synonym", "char", "translate", "llm_attack"].
            save_prompts: Path to save prompts.
            save_results: Path to save results. Required if visualization is True.
            
            # Attack parameters
            replacement_probability: Probability of replacement.
            max_synonyms: Maximum number of synonyms.
            char_change_probability: Probability of changing character case.
            max_char_changes: Maximum number of character changes.
            translation_model: Translation model for translation attack.
            attack_model: Model for LLM-based attack. Now only support ChatGPT.
            attack_type: Type of attack for LLM-based attack. 
                        Choices=["paraphrase", "constraint_change", "scope_expansion", "semantic_preserve"].
            adv_temperature: Temperature for LLM-based attack.
            adv_max_tokens: Maximum tokens for LLM-based attack.
            input_type: Type of input, decided by the dataset.
            noise_type: Type of noise. Choices=["gaussian", "uniform"].
            noise_level: Level of noise.
            seed: Random seed for reproducibility.
            
            # Quantization parameters
            quant_method: Static quantization method. Choices=["bnb", "gptq", "awq"].
            quant_bits: Number of bits for quantization.
            quant_type: Quantization type for 4-bit static quantization. Choices=["nf4", "nf4_2", "nf4_3"].
            quantize_embeddings: Whether to quantize embeddings (for dynamic).
            
            # Generation parameters
            num_return_sequences: Number of responses to generate.
            max_length: Maximum generation length.
            temperature: Temperature for sampling.
            top_p: Top-p for sampling, generally used with temperature.
            num_beams: Number of beams for beam search.
            use_beam_search: Whether to use beam search.
            
            # VLLM parameters
            tensor_parallel_size: Number of GPUs for tensor parallelism. If None, uses all available GPUs.
            gpu_memory_utilization: GPU memory fraction (e.g. 0.85). Lower if OOM during sampler warmup.
            
        Returns:
            Dictionary containing evaluation results.
        """
        # Create config objects from parameters  
        attack_config = AttackConfig(
            replacement_probability=replacement_probability,
            max_synonyms=max_synonyms,
            char_change_probability=char_change_probability,
            max_char_changes=max_char_changes,
            translation_model=translation_model,
            attack_model=attack_model,
            attack_type=attack_type,
            adv_temperature=adv_temperature,
            adv_max_tokens=adv_max_tokens,
            api_path=api_path,
            input_type=input_type,
            noise_type=noise_type,
            noise_level=noise_level,
            seed=seed,
        )
        
        quantization_config = QuantizationConfig(
            method=quant_method,
            bits=quant_bits,
            quant_type=quant_type,
            quantize_embeddings=quantize_embeddings,
        )
        
        generation_config = GenerationConfig(
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            use_beam_search=use_beam_search,
        )
        
        config = EvaluationConfig(
            model_path=model_path,
            model_type=model_type,
            quantized_type=quantized_type,
            dataset=dataset,
            attack_method=attack_method,
            save_prompts=save_prompts,
            save_results=save_results,
            visualization=visualization,
            gen_ori=gen_ori,
            original_results=original_results,
            attack_config=attack_config,
            quantization_config=quantization_config,
            generation_config=generation_config,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        return self.evaluate_with_config(config)
    
    def evaluate_with_config(self, config: EvaluationConfig) -> Dict[str, Any]:
        """
        Run adversarial attack evaluation using configuration objects.

        The attacker is initialised *before* the main model so that GPU-heavy
        attack models (e.g. the translation back-translation pipeline) can
        claim their memory first.

        Args:
            config: Complete evaluation configuration.

        Returns:
            Dictionary containing evaluation results.
        """
        self._validate_parameters(config.visualization, config.save_results)

        # ------------------------------------------------------------------
        # 1. Build the attack config dict and derive input_type from dataset
        #    so the attacker can be instantiated without the dataset loader.
        # ------------------------------------------------------------------
        attack_config_dict = asdict(config.attack_config)
        dataset_lower = config.dataset.lower()
        if dataset_lower == "humaneval":
            attack_config_dict["input_type"] = "code"
        elif dataset_lower == "mbpp":
            attack_config_dict["input_type"] = "prompt"
        elif dataset_lower == "canitedit":
            attack_config_dict["input_type"] = "instruction"
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}. Choose 'humaneval', 'mbpp', or 'canitedit'")

        # ------------------------------------------------------------------
        # 2. Instantiate the attacker BEFORE the main model is loaded so that
        #    GPU resources are reserved in the correct order.
        # ------------------------------------------------------------------
        attack_class = AttackRegistry.get(config.attack_method)
        attacker = attack_class(attack_config_dict)

        # ------------------------------------------------------------------
        # 3. Load the main (generation) model.
        # ------------------------------------------------------------------
        actual_model_type = self._determine_model_type(config.model_type, config.quantized_type)

        model_config = self._setup_model_config(
            config.model_type,
            config.quantized_type,
            config.quantization_config.method,
            config.quantization_config.bits,
            config.quantization_config.quant_type,
            config.quantization_config.quantize_embeddings,
            config.generation_config.num_return_sequences,
            config.generation_config.max_length,
            config.generation_config.temperature,
            config.generation_config.top_p,
            config.generation_config.num_beams,
            config.generation_config.use_beam_search,
            config.tensor_parallel_size,
        )
        if config.model_type == "vllm" and config.gpu_memory_utilization is not None:
            model_config["gpu_memory_utilization"] = config.gpu_memory_utilization

        model = Models.load(actual_model_type, config.model_path, **model_config)

        # ------------------------------------------------------------------
        # 4. Create the framework with the pre-built attacker and run.
        # ------------------------------------------------------------------
        framework = AttackFramework(
            model=model,
            attack_method=config.attack_method,
            attack_config=attack_config_dict,
            dataset=config.dataset,
            attacker=attacker,
        )

        results = framework.run_attack(
            save_prompts=config.save_prompts,
            save_results=config.save_results,
            gen_ori=config.gen_ori,
        )

        if config.visualization:
            visualizer.visualize_results(results, config.save_results)

        return results
    
    def _validate_parameters(self, visualization: bool, save_results: Optional[str]) -> None:
        """Validate input parameters."""
        if visualization and save_results is None:
            raise ValueError("save_results must be provided when visualization is enabled")
    
    def _determine_model_type(self, model_type: str, quantized_type: Optional[str]) -> str:
        """
        Determine the actual model type to load based on model_type and quantized_type.
        
        Args:
            model_type: Base model type (e.g., "vllm", "codellama")
            quantized_type: Quantization type (e.g., "static", "dynamic")
            
        Returns:
            Actual model type to load from registry
        """
        # For VLLM models, determine if quantized or not
        if model_type == "vllm":
            if quantized_type is not None:
                return "vllm_quantized"
            else:
                return "vllm"
        
        # For other models, use quantized_type if specified, otherwise use model_type
        if quantized_type is not None:
            return quantized_type
        else:
            return model_type
    
    def _setup_model_config(
        self, model_type, quantized_type, quant_method, quant_bits, quant_type,
        quantize_embeddings, num_return_sequences, max_length, temperature,
        top_p, num_beams, use_beam_search, tensor_parallel_size
    ) -> Dict[str, Any]:
        """Set up model configuration including quantization and generation parameters."""
        quant_params = {
            "method": quant_method,
            "bits": quant_bits,
            "quant_type": quant_type,
            "quantize_embeddings": quantize_embeddings
        }
        
        gen_params = {
            "num_return_sequences": num_return_sequences,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "use_beam_search": use_beam_search
        }
        
        model_config = self.config_manager.create_model_config(
            model_type=model_type,
            quantized_type=quantized_type,
            quant_params=quant_params,
            gen_params=gen_params
        )
        
        # Add VLLM-specific parameters
        if model_type == "vllm":
            # Auto-detect GPU count if tensor_parallel_size not specified
            if tensor_parallel_size is None:
                import torch
                if torch.cuda.is_available():
                    tensor_parallel_size = torch.cuda.device_count()
                else:
                    tensor_parallel_size = 1
            
            model_config["tensor_parallel_size"] = tensor_parallel_size
        
        return model_config
    
    def _generate_visualization(
        self, results, gen_ori: bool, original_results: Optional[str], 
        model_path: str, save_results: str
    ) -> None:
        """Generate visualization of results."""
        model_name = model_path.rsplit('/', 1)[-1]
        
        if gen_ori:
            visualizer(results[0], results[1], model_name, save_results)
        else:
            if original_results is None:
                raise ValueError("original_results must be provided when gen_ori is False")
            with open(original_results, "r") as f:
                ori_results = json.load(f)
            visualizer(ori_results, results, model_name, save_results) 


class AttackEvaluatorCLI:
    """Command-line interface for running adversarial attack evaluations."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.evaluator = AttackEvaluator()
    
    def attack(self, **kwargs):
        """
        Run adversarial attack evaluation on code generation models.
        
        Accepts all parameters supported by AttackEvaluator.evaluate().
        Use 'reval attack --help' to see all available parameters.
        """
        self.evaluator.evaluate(**kwargs)

    def attack_with_config(self, config_path: str):
        """
        Run attack evaluation using a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        import json
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config from dictionary
        config = EvaluationConfig(**config_dict)
        
        return self.evaluator.evaluate_with_config(config)


def main():
    """Main entry point for the CLI."""
    fire.Fire(AttackEvaluatorCLI)


if __name__ == "__main__":
    main()
