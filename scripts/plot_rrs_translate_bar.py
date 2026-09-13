"""Bar chart of RRS (translate attack, MBPP+) for dense models, grouped by family."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from metadata import MODELS, outputs_dir

_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?[bBmMkK])", re.IGNORECASE)


def _extract_size_label(model_name: str) -> str:
    """Extract a human-readable size label (e.g. '1B', '350M') from model name."""
    match = _SIZE_PATTERN.search(model_name)
    if match:
        raw = match.group(1)
        suffix = raw[-1].upper()
        number_str = raw[:-1]
        try:
            number = float(number_str)
            if number == int(number):
                return f"{int(number)}{suffix}"
            return f"{number}{suffix}"
        except ValueError:
            return raw.upper()
    return model_name

OUTPUT_DIR = outputs_dir()
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DATASET = "mbpp"
ATTACK = "translate"
DENSE_FAMILIES = ["LLaMA", "DeepSeek", "CodeGen", "StarCoder", "NextCoder", "Gemma"]


def _load_plus(folder: Path) -> float | None:
    fpath = folder / "pass_rates.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath) as f:
            return json.load(f)["plus"]
    except (KeyError, json.JSONDecodeError):
        return None


def compute_rrs(model: str, quant: str) -> float | None:
    """Compute RRS = |fp_before - fp_after| / |quant_before - quant_after|."""
    org, name = model.split("/", 1)
    base_dir = OUTPUT_DIR / DATASET / org / name

    fp_before = _load_plus(base_dir / "noise" / "gaussian" / "0.0" / "base")
    fp_after = _load_plus(base_dir / ATTACK / "base")
    q_before = _load_plus(base_dir / "noise" / "gaussian" / "0.0" / quant)
    q_after = _load_plus(base_dir / ATTACK / quant)

    if any(v is None for v in [fp_before, fp_after, q_before, q_after]):
        return None

    denom = abs(q_before - q_after)
    if denom == 0.0:
        return float("inf")
    return abs(fp_before - fp_after) / denom


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
    })

    # Collect data
    labels = []          # size labels
    rrs_8bit = []
    rrs_4bit = []
    family_boundaries = []  # (start_idx, family_name)

    idx = 0
    for family in DENSE_FAMILIES:
        family_boundaries.append((idx, family))
        for model in MODELS[family]:
            short = model.split("/")[-1]
            labels.append(_extract_size_label(short))

            r8 = compute_rrs(model, "bnb8")
            r4 = compute_rrs(model, "bnb4")
            # Cap inf at a visible sentinel for plotting
            rrs_8bit.append(r8 if r8 is not None and r8 != float("inf") else None)
            rrs_4bit.append(r4 if r4 is not None and r4 != float("inf") else None)
            idx += 1

    # Print RRS table
    print(f"\n{'Family':<12} {'Size':<8} {'8-bit RRS':>10} {'4-bit RRS':>10}")
    print("-" * 42)
    fi = 0
    for i, (label, r8, r4) in enumerate(zip(labels, rrs_8bit, rrs_4bit)):
        if fi < len(family_boundaries) and family_boundaries[fi][0] == i:
            family_name = family_boundaries[fi][1]
            fi += 1
        r8_str = f"{r8:.2f}" if r8 is not None else "N/A"
        r4_str = f"{r4:.2f}" if r4 is not None else "N/A"
        print(f"{family_name:<12} {label:<8} {r8_str:>10} {r4_str:>10}")
    print()

    # Build x positions: tight within family, gap between families
    n = len(labels)
    bar_width = 0.5
    group_gap = 1.0
    x = []
    family_ranges = []  # (x_start, x_end, family_name)
    pos = 0.0
    for i, (start, family) in enumerate(family_boundaries):
        end = family_boundaries[i + 1][0] if i + 1 < len(family_boundaries) else n
        count = end - start
        if start > 0:
            pos += group_gap
        x_start = pos
        for j in range(count):
            x.append(pos)
            pos += 1.0
        family_ranges.append((x_start, pos - 1.0, family))
    x = np.array(x)

    fig, ax = plt.subplots(figsize=(9, 3.5))

    vals_8 = [v if v is not None else 0 for v in rrs_8bit]
    vals_4 = [v if v is not None else 0 for v in rrs_4bit]

    color_blue = (135 / 255, 197 / 255, 229 / 255)
    color_green = (150 / 255, 232 / 255, 141 / 255)
    ax.bar(x - bar_width / 2, vals_8, bar_width, color=color_blue, label="8-bit RRS", zorder=3)
    ax.bar(x + bar_width / 2, vals_4, bar_width, color=color_green, label="4-bit RRS", zorder=3)

    # RRS = 1 reference line
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="RRS = 1")

    # X-axis: horizontal size labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("RRS")
    ax.set_title("RRS under Translate Attack (MBPP+, Dense Models)", fontsize=10, fontweight="bold")

    # Family group separators and labels
    for i, (x_start, x_end, family) in enumerate(family_ranges):
        if i > 0:
            prev_end = family_ranges[i - 1][1]
            ax.axvline(x=(prev_end + x_start) / 2, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        mid = (x_start + x_end) / 2
        ax.text(mid, -0.15, family, ha="center", va="top", fontsize=7, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7)

    out_path = PLOT_DIR / "RQ3_adv.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
