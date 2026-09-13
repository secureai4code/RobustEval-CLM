"""Generate a combined 2x4 noise robustness plot for selected model/noise-type pairs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from metadata import NOISE_LEVELS, outputs_dir

OUTPUT_DIR = outputs_dir()
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DATASET = "mbpp"
NOISE_LABELS = {"gaussian": "Gaussian", "uniform": "Uniform"}
QUANT_STYLES = {"base": ("-", "o", "Base"), "bnb8": ("--", "s", "8-bit"), "bnb4": (":", "^", "4-bit")}
QUANT_COLORS = {"base": "#1f77b4", "bnb8": "#ff7f0e", "bnb4": "#2ca02c"}

# (model_id, noise_type) — row-major order for 2x4 grid
SUBPLOTS = [
    ("deepseek-ai/deepseek-coder-1.3b-base", "gaussian"),
    ("deepseek-ai/deepseek-coder-33b-base", "gaussian"),
    ("meta-llama/Llama-3.2-3B", "gaussian"),
    ("meta-llama/Llama-3.1-8B", "uniform"),
    ("Salesforce/codegen-6B-mono", "uniform"),
    ("bigcode/starcoder2-7b", "gaussian"),
    ("microsoft/NextCoder-14B", "gaussian"),
    ("google/gemma-3-4b-it", "gaussian"),
]


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


def main():
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

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.14, wspace=0.25, hspace=0.35)

    x_positions = np.arange(len(NOISE_LEVELS))

    for idx, (model, noise_type) in enumerate(SUBPLOTS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for quant, (ls, marker, qlabel) in QUANT_STYLES.items():
            vals, xs = [], []
            for nl_idx, nl in enumerate(NOISE_LEVELS):
                v = _load_pass_at_1(model, noise_type, nl, quant)
                if v is not None:
                    vals.append(v)
                    xs.append(nl_idx)
            if not vals:
                continue
            ax.plot(xs, vals, linestyle=ls, color=QUANT_COLORS[quant], marker=marker, label=qlabel)

        short_name = model.split("/")[-1]
        ax.set_title(f"{short_name} ({NOISE_LABELS[noise_type]})", fontsize=7, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(NOISE_LEVELS, rotation=45, ha="right")
        ax.set_xlim(-0.3, len(NOISE_LEVELS) - 0.7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if col == 0:
            ax.set_ylabel("pass@1")

    fig.supxlabel("Noise Level", fontsize=8)

    # Legend inside the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="best", frameon=True, framealpha=0.9, fontsize=6)

    out_path = PLOT_DIR / "RQ3.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
