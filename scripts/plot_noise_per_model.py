"""Generate individual noise robustness plots for each dense model on MBPP."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from metadata import MODELS, NOISE_LEVELS, NOISE_TYPES, outputs_dir

OUTPUT_DIR = outputs_dir()
PLOT_DIR = ROOT / "noise_plots"
PLOT_DIR.mkdir(exist_ok=True)

DATASET = "mbpp"
NOISE_LABELS = {"gaussian": "Gaussian", "uniform": "Uniform"}
QUANT_STYLES = {"base": ("-", "o", "Base"), "bnb8": ("--", "s", "8-bit"), "bnb4": (":", "^", "4-bit")}
QUANT_COLORS = {"base": "#1f77b4", "bnb8": "#ff7f0e", "bnb4": "#2ca02c"}

DENSE_FAMILIES = ["LLaMA", "DeepSeek", "CodeGen", "StarCoder", "NextCoder", "Gemma"]


def _load_pass_at_1(model: str, noise_type: str, noise_level: str, quant: str) -> float | None:
    fpath = OUTPUT_DIR / DATASET / model / "noise" / noise_type / noise_level / quant / "pass_rates.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
        return data["plus"]
    except (KeyError, json.JSONDecodeError):
        return None


def _safe_filename(model: str) -> str:
    return model.replace("/", "_")


def plot_model(model: str) -> None:
    """Generate a 1x2 plot (Gaussian | Uniform) for a single model."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 0.7,
        "lines.markersize": 3,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.85, top=0.85, bottom=0.18, wspace=0.08)

    x_positions = np.arange(len(NOISE_LEVELS))
    all_vals = []

    for col_idx, nt in enumerate(NOISE_TYPES):
        ax = axes[col_idx]
        for quant, (ls, marker, qlabel) in QUANT_STYLES.items():
            vals, xs = [], []
            for nl_idx, nl in enumerate(NOISE_LEVELS):
                v = _load_pass_at_1(model, nt, nl, quant)
                if v is not None:
                    vals.append(v)
                    xs.append(nl_idx)
                    all_vals.append(v)
            if not vals:
                continue
            ax.plot(xs, vals, linestyle=ls, color=QUANT_COLORS[quant], marker=marker, label=qlabel)

        ax.set_title(NOISE_LABELS[nt])
        ax.set_xticks(x_positions)
        ax.set_xticklabels(NOISE_LEVELS, rotation=45, ha="right")
        ax.set_xlim(-0.3, len(NOISE_LEVELS) - 0.7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        margin = (y_max - y_min) * 0.08 if y_max > y_min else 0.05
        for ax in axes:
            ax.set_ylim(y_min - margin, y_max + margin)

    axes[0].set_ylabel("pass@1")
    fig.supxlabel("Noise Level", fontsize=7)

    short_name = model.split("/")[-1]
    fig.suptitle(short_name, fontsize=9, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    out_path = PLOT_DIR / f"{_safe_filename(model)}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def main():
    for family in DENSE_FAMILIES:
        models = MODELS[family]
        for model in models:
            print(f"Plotting {model} ...")
            plot_model(model)
    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
