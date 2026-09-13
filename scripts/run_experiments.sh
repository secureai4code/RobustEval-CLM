#!/usr/bin/env bash
# Batch driver for the paper's full experiment sweep:
#   models x datasets x (adversarial attacks + weight noise) x quantization levels.
#
# Run inside the project environment (`source .venv/bin/activate` or prefix with
# `uv run`), from the repository root. Comment out array entries to run a subset.
#
# Backend / quantization rules used below:
#   - Most dense models run on the vLLM backend; StarCoder2 and CodeGen use the
#     HF backend (--model_type codellama).
#   - MoE models support only 4-bit quantization (--quant_bits 4).
#   - vLLM + quantization runs use --tensor_parallel_size 1: determinism
#     requires a single GPU.
set -u

# Deterministic vLLM behavior (also set in src/core/models/model_implementations.py).
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NCCL_P2P_DISABLE=1
# Optionally pin the GPU used for the single-GPU quantized runs:
# export CUDA_VISIBLE_DEVICES=0

MODELS=(
    # DeepSeek
    "deepseek-ai/deepseek-coder-1.3b-base"
    "deepseek-ai/deepseek-coder-6.7b-base"
    "deepseek-ai/deepseek-coder-33b-base"
    # LLaMA
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    # CodeGen (no CanItEdit: samples exceed the 2,048-token context window)
    "Salesforce/codegen-350M-mono"
    "Salesforce/codegen-2B-mono"
    "Salesforce/codegen-6B-mono"
    # StarCoder2
    "bigcode/starcoder2-3b"
    "bigcode/starcoder2-7b"
    "bigcode/starcoder2-15b"
    # NextCoder
    "microsoft/NextCoder-7B"
    "microsoft/NextCoder-14B"
    "microsoft/NextCoder-32B"
    # Gemma
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    # MoE
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
)

DATASETS=(
    "mbpp"
    "humaneval"
    "canitedit"
)

ATTACKS=(
    "char"
    "synonym"
    "translate"
    "destructure"
)

NOISE_TYPES=(
    "gaussian"
    "uniform"
)

NOISE_LEVELS=(
    "0.0"
    "1e-4"
    "1e-3"
    "3e-3"
    "5e-3"
    "1e-2"
)

model_type_for() {
    # StarCoder2 and CodeGen require the HF backend; everything else uses vLLM.
    case "$1" in
        bigcode/* | Salesforce/*) echo "codellama" ;;
        *) echo "vllm" ;;
    esac
}

is_moe() {
    case "$1" in
        Qwen/Qwen3-Coder-30B-A3B-Instruct | deepseek-ai/DeepSeek-Coder-V2-Lite-Base) return 0 ;;
        *) return 1 ;;
    esac
}

for model in "${MODELS[@]}"; do
    model_type=$(model_type_for "$model")
    if is_moe "$model"; then
        quant_bits_adv=(4)
        quant_bits_noise=(4)
    else
        quant_bits_adv=(8)
        quant_bits_noise=(8 4)
    fi

    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" == "canitedit" ]; then
            MAX_LENGTH=4096
        else
            MAX_LENGTH=512
        fi

        for attack in "${ATTACKS[@]}"; do
            echo ">>> $model | $dataset | $attack"
            results_folder="outputs/$dataset/$model/$attack"
            reval attack \
                --model_path "$model" \
                --model_type "$model_type" \
                --dataset "$dataset" \
                --attack_method "$attack" \
                --save_prompts "$results_folder/base" \
                --save_results "$results_folder/base" \
                --max_length "$MAX_LENGTH" \
                --seed 42
            for bits in "${quant_bits_adv[@]}"; do
                reval attack \
                    --model_path "$model" \
                    --model_type "$model_type" \
                    --quantized_type static \
                    --quant_bits "$bits" \
                    --dataset "$dataset" \
                    --attack_method "$attack" \
                    --save_prompts "$results_folder/bnb$bits" \
                    --save_results "$results_folder/bnb$bits" \
                    --max_length "$MAX_LENGTH" \
                    --tensor_parallel_size 1 \
                    --seed 42
            done
        done

        for noise_type in "${NOISE_TYPES[@]}"; do
            for noise_level in "${NOISE_LEVELS[@]}"; do
                echo ">>> $model | $dataset | noise $noise_type $noise_level"
                results_folder="outputs/$dataset/$model/noise/$noise_type/$noise_level"
                reval attack \
                    --model_path "$model" \
                    --model_type "$model_type" \
                    --dataset "$dataset" \
                    --attack_method noise \
                    --noise_type "$noise_type" \
                    --noise_level "$noise_level" \
                    --save_prompts "$results_folder/base" \
                    --save_results "$results_folder/base" \
                    --max_length "$MAX_LENGTH" \
                    --seed 42
                for bits in "${quant_bits_noise[@]}"; do
                    reval attack \
                        --model_path "$model" \
                        --model_type "$model_type" \
                        --quantized_type static \
                        --quant_bits "$bits" \
                        --dataset "$dataset" \
                        --attack_method noise \
                        --noise_type "$noise_type" \
                        --noise_level "$noise_level" \
                        --save_prompts "$results_folder/bnb$bits" \
                        --save_results "$results_folder/bnb$bits" \
                        --max_length "$MAX_LENGTH" \
                        --tensor_parallel_size 1 \
                        --seed 42
                done
            done
        done
    done
done
