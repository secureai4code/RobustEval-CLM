# RobustEval-CLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Replication package for the paper:

> **Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**
> Sen Fang, Weiyuan Ding, Antonio Mastropaolo, and Bowen Xu
> *IEEE Transactions on Software Engineering (TSE)*

A framework for comparing the robustness of **original (full-precision) and
quantized** Code Language Models (CLMs) under two kinds of perturbation:

1. **Prompt perturbation** — character-case flips, synonym substitution,
   back-translation, and LLM-based paraphrase applied to the natural-language
   input the model receives.
2. **Noise perturbation** — additive Gaussian / uniform noise injected directly
   into model weights, supported on both HF Transformers models and vLLM
   workers (via collective RPC).

For each (model, dataset, perturbation) triple the framework runs the same
evaluation at multiple precisions — full precision, 8-bit, and 4-bit — so that
robustness can be compared across quantization settings. Quantization is done
with [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (NF4 / int8),
loaded natively inside vLLM (or through HF Transformers for non-vLLM backends).

Inference uses [vLLM](https://github.com/vllm-project/vllm) for high
throughput; evaluation reuses [EvalPlus](https://github.com/evalplus/evalplus)
for HumanEval+/MBPP+ and the official
[nuprl/CanItEdit](https://github.com/nuprl/CanItEdit) container for code
editing.

## Supported Models

Any HuggingFace causal-LM checkpoint can be used. The experiments in the paper
cover the following families (see `scripts/metadata.py` for the full list):

| Family         | Example checkpoints |
|----------------|---------------------|
| StarCoder2     | `bigcode/starcoder2-{3b,7b,15b}` |
| CodeGen-mono   | `Salesforce/codegen-{350M,2B,6B}-mono` |
| DeepSeek-Coder | `deepseek-ai/deepseek-coder-{1.3b,6.7b,33b}-base`, `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` |
| Llama-3.x      | `meta-llama/Llama-3.2-{1B,3B}`, `meta-llama/Llama-3.1-8B` |
| Gemma-3 (it)   | `google/gemma-3-{4b,12b,27b}-it` |
| NextCoder      | `microsoft/NextCoder-{7B,14B,32B}` |
| Qwen3-Coder    | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE) |

## Datasets

- **HumanEval / HumanEval+** — loaded via `evalplus`.
- **MBPP / MBPP+** — loaded via `evalplus`.
- **CanItEdit** — loaded from `nuprl/CanItEdit` on HuggingFace. Executes
  completions inside the official `ghcr.io/nuprl/canitedit` container, so
  **Docker or Podman is required** for CanItEdit evaluation.

## Attack Methods

Registered names used with `--attack_method`:

The paper evaluates four adversarial attacks — `char` (character-level),
`synonym` (word-level), `translate` (sentence-level), `destructure`
(structure-level) — plus the `noise` weight-perturbation method.

| Name            | What it perturbs | Notes |
|-----------------|------------------|-------|
| `char`          | Prompt           | Random character-case flips. |
| `synonym`       | Prompt           | WordNet-based synonym substitution. |
| `translate`     | Prompt           | Back-translation through mBART-50 (en → de → en). |
| `destructure`   | Prompt           | Deterministically strips structural formatting cues (assertion keywords, doctest markers, markdown headers), collapsing the prompt into flat text. |
| `noise`         | Model weights    | Additive Gaussian / uniform noise (`--noise_type`, `--noise_level`). Prompt is unchanged. |
| `llm_paraphrase` | Prompt          | Precomputed LLM paraphrases loaded from a dataset (`LLM_PARAPHRASE_PATH` / `LLM_PARAPHRASE_REPO`). |
| `llm_attack`    | Prompt           | GPT-4o-based paraphrase / semantic-preserving rewrite. Requires an OpenAI API key at `--api_path`. |
| `natural_noise`, `semantic`, `structural`, `structured` | Prompt | Experimental. |

For CanItEdit, code blocks in the instruction are masked before the attack is
applied and restored afterwards, so that only natural-language instructions are
perturbed.

## Installation

Python 3.10+ and a CUDA-capable GPU are required. Dependencies are managed
with [uv](https://github.com/astral-sh/uv); a single command creates the
virtualenv, resolves the lockfile, installs all runtime dependencies (including
vLLM and BitsAndBytes), and installs `reval` itself in editable mode:

```bash
git clone https://github.com/secureai4code/RobustEval-CLM
cd RobustEval-CLM
uv sync
source .venv/bin/activate
```

Optional extras:

```bash
uv sync --extra scripts   # matplotlib / pandas / scipy for result plotting
uv sync --extra dev       # pytest
```

The NLTK corpora required by the synonym attack (`wordnet`, `punkt`, ...) are
fetched lazily on first import, so no post-install step is needed.

For the CanItEdit dataset, install Docker or Podman — the official
`ghcr.io/nuprl/canitedit` container is used to execute generated edits.

## Quick Start

The CLI entry point is `reval attack`:

```bash
reval attack \
    --model_path deepseek-ai/deepseek-coder-1.3b-base \
    --model_type vllm \
    --dataset humaneval \
    --attack_method synonym \
    --save_prompts outputs/humaneval/deepseek-ai/deepseek-coder-1.3b-base/synonym/base \
    --save_results outputs/humaneval/deepseek-ai/deepseek-coder-1.3b-base/synonym/base \
    --max_length 512 \
    --seed 42
```

### Backend and quantization constraints

- Most dense models run on the vLLM backend (`--model_type vllm`), but the
  **StarCoder2 and CodeGen families must use the HF backend**
  (`--model_type codellama`).
- **MoE models** (`Qwen/Qwen3-Coder-30B-A3B-Instruct`,
  `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`) **support only 4-bit
  quantization** (`--quant_bits 4`); 8-bit is not available for them.
- For **vLLM + quantization**, deterministic generation requires a **single
  GPU**: pass `--tensor_parallel_size 1` (optionally pinning the device via
  `CUDA_VISIBLE_DEVICES`).

`scripts/run_experiments.sh` drives the paper's full sweep
(models × datasets × attacks × quantization levels) and encodes all three
rules.

## CLI Arguments

**Model & backend**

| Flag | Description |
|------|-------------|
| `--model_path` | HuggingFace repo id or local path. **Required.** |
| `--model_type` | `vllm` (recommended), `codellama`, `starcoder`, `codegen`, `deepseek`, `incoder`, `magicoder`. Default `codellama`. |
| `--quantized_type` | `static` to apply BNB quantization, or unset for full precision. |
| `--tensor_parallel_size` | Tensor-parallel size for vLLM. Auto = all visible GPUs. |
| `--gpu_memory_utilization` | vLLM memory fraction, e.g. `0.85`. Lower if OOM during sampler warmup. |

**Quantization**

| Flag | Description |
|------|-------------|
| `--quant_bits` | `4` or `8` (default `8`). |

**Dataset & attack**

| Flag | Description |
|------|-------------|
| `--dataset` | `mbpp`, `humaneval`, `canitedit`. |
| `--attack_method` | See table above. |
| `--noise_type`, `--noise_level` | For the `noise` attack. |
| `--replacement_probability`, `--max_synonyms` | For `synonym`. |
| `--char_change_probability`, `--max_char_changes` | For `char`. |
| `--translation_model` | Back-translation model. Default `facebook/mbart-large-50-many-to-many-mmt`. |
| `--attack_model`, `--attack_type`, `--adv_temperature`, `--adv_max_tokens`, `--api_path` | For `llm_attack`. |

**Generation**

| Flag | Description |
|------|-------------|
| `--max_length` | Max generated tokens. Use `4096` for CanItEdit, `512` otherwise. |
| `--num_return_sequences` | Number of samples per prompt. |
| `--temperature`, `--top_p` | Sampling params (used when `num_return_sequences > 1` and no beam search). |
| `--num_beams`, `--use_beam_search` | Beam search params. |
| `--seed` | Random seed. |

**Output**

| Flag | Description |
|------|-------------|
| `--save_prompts` | Directory for `{original,adversarial}_prompts.jsonl`. |
| `--save_results` | Directory for `{original,adversarial}_results.json` and `pass_rates.json`. |
| `--gen_ori` | Also generate outputs for the original (unperturbed) prompts. |
| `--visualization` | Write Venn-diagram visualizations. Requires `--save_results`. |

Both `--save_prompts` and `--save_results` are resumable: existing entries
with a non-null `solution` field are reused on rerun.

## Examples

### 1. Full-precision vLLM, synonym attack on MBPP

```bash
reval attack \
    --model_path deepseek-ai/deepseek-coder-6.7b-base \
    --model_type vllm \
    --dataset mbpp \
    --attack_method synonym \
    --save_prompts outputs/mbpp/deepseek-ai/deepseek-coder-6.7b-base/synonym/base \
    --save_results outputs/mbpp/deepseek-ai/deepseek-coder-6.7b-base/synonym/base \
    --seed 42
```

### 2. StarCoder2 on the HF backend (`codellama`), destructure attack

StarCoder2 and CodeGen models must use `--model_type codellama` instead of
`vllm`:

```bash
reval attack \
    --model_path bigcode/starcoder2-7b \
    --model_type codellama \
    --dataset mbpp \
    --attack_method destructure \
    --save_prompts outputs/mbpp/bigcode/starcoder2-7b/destructure/base \
    --save_results outputs/mbpp/bigcode/starcoder2-7b/destructure/base \
    --seed 42
```

### 3. 4-bit BNB + Gaussian weight noise on HumanEval (single GPU for determinism)

```bash
reval attack \
    --model_path meta-llama/Llama-3.1-8B \
    --model_type vllm \
    --quantized_type static \
    --quant_bits 4 \
    --dataset humaneval \
    --attack_method noise \
    --noise_type gaussian \
    --noise_level 1e-3 \
    --save_prompts outputs/humaneval/meta-llama/Llama-3.1-8B/noise/gaussian/1e-3/bnb4 \
    --save_results outputs/humaneval/meta-llama/Llama-3.1-8B/noise/gaussian/1e-3/bnb4 \
    --tensor_parallel_size 1 \
    --seed 42
```

### 4. Quantized MoE on CanItEdit with back-translation (Docker required)

MoE models only support 4-bit quantization:

```bash
reval attack \
    --model_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --model_type vllm \
    --quantized_type static \
    --quant_bits 4 \
    --dataset canitedit \
    --attack_method translate \
    --max_length 4096 \
    --save_prompts outputs/canitedit/Qwen/Qwen3-Coder-30B-A3B-Instruct/translate/bnb4 \
    --save_results outputs/canitedit/Qwen/Qwen3-Coder-30B-A3B-Instruct/translate/bnb4 \
    --tensor_parallel_size 1 \
    --seed 42
```

## Output Layout

Results are written under the hierarchy consumed by
`scripts/generate_all.py`:

```
outputs/<dataset>/<org>/<model>/<attack>[/<noise_type>/<noise_level>]/<quant>/
├── original_prompts.jsonl        # if --gen_ori
├── adversarial_prompts.jsonl
├── original_results.json         # if --gen_ori
├── adversarial_results.json
└── pass_rates.json               # {"base": <pass@1>, "plus": <pass@1>}
```

Where `<quant>` is one of `base`, `bnb8`, `bnb4`. For CanItEdit, intermediate
completions are placed in `original_completions/` and `adversarial_completions/`
subfolders and evaluated by the Docker/Podman container.

## Reproducing Paper Results

All experimental results — the pass@1 summaries plus the raw data (adversarial
prompts, model generations, and per-sample evaluation results) — are archived
on Zenodo (~630 MB compressed):

> **DOI**: [10.5281/zenodo.22737766](https://zenodo.org/records/22737766)

Download `RobustEval-CLM-raw-results.tar.gz` and extract it at the repository
root, where it unpacks as `outputs_public/`:

```bash
wget https://zenodo.org/records/22737766/files/RobustEval-CLM-raw-results.tar.gz
tar xzf RobustEval-CLM-raw-results.tar.gz    # -> outputs_public/
```

The tree holds 2,436 experiment folders — 20 models × 3 datasets ×
(4 adversarial attacks + 2 noise types × 6 intensity levels) × quantization
levels — each with a `pass_rates.json` (`{"base": <pass@1>, "plus": <pass@1>}`:
the base and plus test-set variants for HumanEval/MBPP, or the Descriptive and
Lazy prompt variants for CanItEdit) next to the raw prompts and generations.
The clean (unattacked) baseline for a model+quantization is the noise entry at
level `0.0`. CodeGen models have no CanItEdit results (their samples exceed
the 2,048-token context window), so those table cells render as `-`.

With the archive in place, all tables and figures regenerate directly:

```bash
uv sync --extra scripts    # matplotlib / scipy
python scripts/generate_all.py
```

`generate_all.py` writes into `statistic_results/`: `passat1.tex`,
`passrate_rl.tex`, `passat1_noise.tex`, `noise_results.csv`, `rrs.tex`, and
`noise_robustness_figure.{pdf,png}`.

Additional analysis scripts (all read `outputs_public/` the same way):

| Script | Output |
|--------|--------|
| `scripts/stat_analysis.py` | Wilcoxon signed-rank tests + bootstrap CIs (statistical validation). |
| `scripts/plot_noise_combined.py` | `plots/RQ3.{pdf,png}` — noise robustness grid. |
| `scripts/plot_rrs_translate_bar.py` | `plots/RQ3_adv.{pdf,png}` — RRS under the translate attack. |
| `scripts/plot_noise_per_model.py` | `noise_plots/<model>.{pdf,png}` — per-model noise curves. |
| `scripts/stat_attack_drop.py` | Per-attack performance-drop CSV + plot. |
| `scripts/avg_quant_drop.py` | Average clean pass@1 cost of 8-bit / 4-bit quantization. |

## Repository Layout

```
src/
├── core/
│   ├── datasets/          # adversarial dataset wrapper (LLM-based attacks)
│   └── models/            # HF, vLLM, and quantized model implementations
│       └── vllm_noise_injector.py  # RPC-based noise injection for vLLM workers
├── evaluator/
│   ├── attack_evaluator/
│   │   ├── attack_config.py
│   │   ├── attack_evaluator.py     # CLI entry point (reval)
│   │   ├── attack_registry.py
│   │   ├── attacks/                # char, synonym, translate, noise, chatgpt, ...
│   │   └── framework/              # end-to-end orchestration
│   └── utils/                      # evaluation, translation helpers, visualization
└── utils/                          # content masking, function extraction
scripts/
├── metadata.py                     # models / datasets / attacks / quantization metadata
├── generate_all.py                 # regenerate all paper tables and figures
├── run_experiments.sh              # batch driver: models x datasets x attacks x quant
└── ...                             # statistics and plotting scripts
outputs_public/                     # extracted Zenodo results archive (gitignored)
statistic_results/                  # generated tables / figures (gitignored)
```

## Citation

```bibtex
@article{fang2026smaller,
  title   = {Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation},
  author  = {Fang, Sen and Ding, Weiyuan and Mastropaolo, Antonio and Xu, Bowen},
  journal = {IEEE Transactions on Software Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## Contributing

Contributions are welcome — please open an issue or PR. Questions:
<fangsen1996@gmail.com> / <sfang9@ncsu.edu>.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) — inference engine.
- [EvalPlus](https://github.com/evalplus/evalplus) — HumanEval+/MBPP+ evaluation.
- [nuprl/CanItEdit](https://github.com/nuprl/CanItEdit) — code-editing benchmark and evaluator image.
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4/8-bit quantization.
- [HuggingFace Transformers](https://github.com/huggingface/transformers) / [datasets](https://github.com/huggingface/datasets).

## License

MIT — see [LICENSE](LICENSE).
