from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def outputs_dir() -> Path:
    """Return the results directory used by all analysis scripts.

    Prefers the committed pass@1-only tree ``outputs_public/`` (shipped with the
    repository and matching the paper's experiment scope); falls back to the
    full raw ``outputs/`` tree produced by local experiment runs.
    """
    public = ROOT / "outputs_public"
    return public if public.exists() else ROOT / "outputs"


MODELS = {
    'DeepSeek': [
        'deepseek-ai/deepseek-coder-1.3b-base',
        'deepseek-ai/deepseek-coder-6.7b-base',
        'deepseek-ai/deepseek-coder-33b-base',
    ],
    'LLaMA': [
        'meta-llama/Llama-3.2-1B',
        'meta-llama/Llama-3.2-3B',
        'meta-llama/Llama-3.1-8B',
    ],
    'CodeGen': [
        'Salesforce/codegen-350M-mono',
        'Salesforce/codegen-2B-mono',
        'Salesforce/codegen-6B-mono',
    ],
    'StarCoder': [
        'bigcode/starcoder2-3b',
        'bigcode/starcoder2-7b',
        'bigcode/starcoder2-15b',
    ],
    'NextCoder': [
        'microsoft/NextCoder-7B',
        'microsoft/NextCoder-14B',
        'microsoft/NextCoder-32B',
    ],
    'Gemma': [
        'google/gemma-3-4b-it',
        'google/gemma-3-12b-it',
        'google/gemma-3-27b-it',
    ],
    'MOE': [
        'Qwen/Qwen3-Coder-30B-A3B-Instruct',
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Base',
    ]
}

DATASETS = [
    'mbpp',
    'humaneval',
    'canitedit',
]


ATTACKS = [
    'char',
    'synonym',
    'translate',
    'destructure',
]

NOISE_TYPES = [
    'gaussian',
    'uniform',
]

NOISE_LEVELS = [
    '0.0',
    '1e-4',
    '1e-3',
    '3e-3',
    '5e-3',
    '1e-2',
]

QUANTIZED_TYPES = {
    'adversarial': ['base', 'bnb8'],
    'noise': ['base', 'bnb8', 'bnb4'],
    'moe': ['base', 'bnb4']
}
