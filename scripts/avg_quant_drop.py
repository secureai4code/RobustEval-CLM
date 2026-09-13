"""Calculate average pass@1 drop for 8-bit and 4-bit vs base at gaussian noise level 0.0 on MBPP+."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metadata import MODELS, outputs_dir

OUTPUT_DIR = outputs_dir()
DATASET = "mbpp"
NOISE_TYPE = "gaussian"
NOISE_LEVEL = "0.0"

DENSE_FAMILIES = ["LLaMA", "DeepSeek", "CodeGen", "StarCoder", "NextCoder", "Gemma"]


def load_pass_at_1(model: str, quant: str) -> float | None:
    fpath = OUTPUT_DIR / DATASET / model / "noise" / NOISE_TYPE / NOISE_LEVEL / quant / "pass_rates.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
        return data["plus"]
    except (KeyError, json.JSONDecodeError):
        return None


def main():
    drops_8bit = []
    drops_4bit = []

    print(f"{'Model':<45} {'Base':>8} {'8-bit':>8} {'Drop8':>8} {'4-bit':>8} {'Drop4':>8}")
    print("-" * 95)

    for family in DENSE_FAMILIES:
        for model in MODELS[family]:
            base = load_pass_at_1(model, "base")
            bnb8 = load_pass_at_1(model, "bnb8")
            bnb4 = load_pass_at_1(model, "bnb4")

            if base is None:
                continue

            drop8 = (base - bnb8) if bnb8 is not None else None
            drop4 = (base - bnb4) if bnb4 is not None else None

            if drop8 is not None:
                drops_8bit.append(drop8)
            if drop4 is not None:
                drops_4bit.append(drop4)

            print(f"{model:<45} {base:>8.4f}"
                  f" {bnb8:>8.4f}" if bnb8 is not None else f"{'N/A':>8}",
                  f" {drop8:>8.4f}" if drop8 is not None else f"{'N/A':>8}",
                  f" {bnb4:>8.4f}" if bnb4 is not None else f"{'N/A':>8}",
                  f" {drop4:>8.4f}" if drop4 is not None else f"{'N/A':>8}")

    print("-" * 95)
    if drops_8bit:
        print(f"Average pass@1 drop (8-bit): {sum(drops_8bit)/len(drops_8bit):.4f}  (n={len(drops_8bit)})")
    if drops_4bit:
        print(f"Average pass@1 drop (4-bit): {sum(drops_4bit)/len(drops_4bit):.4f}  (n={len(drops_4bit)})")


if __name__ == "__main__":
    main()
