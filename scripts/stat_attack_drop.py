"""Statistic the pass@1 drop under an adversarial attack (default: destructure).

An exp case is one attack x model x quant x dataset x score (base|plus). For each
case the drop is the clean pass@1 (noise/gaussian/0.0) minus the attacked pass@1,
in percentage points. The script writes a CSV of all cases to statistic_results/
and a summary figure (distribution, means, and a per-case heatmap) to plots/.

Usage:
    uv run python scripts/stat_attack_drop.py [--attack destructure]
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from metadata import DATASETS, MODELS, QUANTIZED_TYPES, outputs_dir

OUTPUTS_DIR = outputs_dir()

DATASET_LABELS = {"mbpp": "MBPP", "humaneval": "HumanEval", "canitedit": "CanItEdit"}
QUANT_LABELS = {"base": "FP", "bnb8": "8-bit", "bnb4": "4-bit"}
SCORES = ["base", "plus"]
SCORE_LABELS = {"base": "Base", "plus": "Plus"}

# Ordinal one-hue ramp for precision (light -> dark = FP -> 4-bit).
PREC_COLORS = {"FP": "#86b6ef", "8-bit": "#2a78d6", "4-bit": "#104281"}
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
MISSING_FILL = "#e9e8e3"

# Diverging: blue (attack helped) <- gray 0 -> red (attack hurt).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "drop_diverging",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a2a1", "#d03b3b", "#7a1f1f"],
)


def load_pass_rates(folder: Path) -> Optional[dict]:
    """Load pass_rates.json from a results folder, or None if missing/unreadable."""
    fpath = folder / "pass_rates.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def quant_types_for(family: str) -> list:
    """Quant options evaluated under adversarial attacks for a model family."""
    return QUANTIZED_TYPES["moe"] if family == "MOE" else QUANTIZED_TYPES["adversarial"]


def collect_cases(attack: str) -> list:
    """Collect one record per exp case, including cases with missing results.

    Returns:
        List of dicts with keys: dataset, family, model, quant, prec, score,
        clean, attacked, drop (drop in percentage points; None when either side
        of the comparison is missing).
    """
    cases = []
    for dataset in DATASETS:
        for family, models in MODELS.items():
            for model_path in models:
                model_dir = OUTPUTS_DIR / dataset / model_path
                for quant in quant_types_for(family):
                    attacked = load_pass_rates(model_dir / attack / quant)
                    clean = load_pass_rates(model_dir / "noise" / "gaussian" / "0.0" / quant)
                    for score in SCORES:
                        clean_v = clean.get(score) if clean else None
                        att_v = attacked.get(score) if attacked else None
                        drop = (clean_v - att_v) * 100 if clean_v is not None and att_v is not None else None
                        cases.append({
                            "dataset": dataset,
                            "family": family,
                            "model": model_path.split("/")[-1],
                            "quant": quant,
                            "prec": QUANT_LABELS[quant],
                            "score": score,
                            "clean": clean_v,
                            "attacked": att_v,
                            "drop": drop,
                        })
    return cases


def write_csv(cases: list, out_path: Path) -> None:
    """Write all exp cases (including missing ones) to a CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "family", "model", "prec", "score", "clean", "attacked", "drop"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow(case)


def print_summary(cases: list, attack: str) -> None:
    """Print per-dataset/precision drop statistics and missing-case counts."""
    valid = [c for c in cases if c["drop"] is not None]
    missing = len(cases) - len(valid)
    print(f"Attack: {attack} — {len(valid)} cases with results, {missing} missing")
    header = f"{'dataset':<10} {'prec':<6} {'n':>3} {'mean':>7} {'median':>7} {'min':>7} {'max':>7}"
    print(header)
    print("-" * len(header))
    groups = [("all", "all", valid)]
    for dataset in DATASETS:
        for prec in PREC_COLORS:
            sub = [c for c in valid if c["dataset"] == dataset and c["prec"] == prec]
            if sub:
                groups.append((dataset, prec, sub))
    for dataset, prec, sub in groups:
        drops = np.array([c["drop"] for c in sub])
        print(f"{dataset:<10} {prec:<6} {len(drops):>3} {drops.mean():>7.2f} "
              f"{np.median(drops):>7.2f} {drops.min():>7.2f} {drops.max():>7.2f}")


def _text_ink(rgba: tuple) -> str:
    """Pick black or white cell text based on the background luminance."""
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#ffffff" if lum < 0.45 else INK


def draw_distribution(ax, cases: list) -> None:
    """Panel (a): box plots + jittered points of drops per dataset and precision."""
    rng = np.random.default_rng(0)
    precs = list(PREC_COLORS)
    offsets = {"FP": -0.26, "8-bit": 0.0, "4-bit": 0.26}
    for ds_idx, dataset in enumerate(DATASETS):
        for prec in precs:
            drops = [c["drop"] for c in cases
                     if c["drop"] is not None and c["dataset"] == dataset and c["prec"] == prec]
            if not drops:
                continue
            pos = ds_idx + offsets[prec]
            color = PREC_COLORS[prec]
            box = ax.boxplot(
                [drops], positions=[pos], widths=0.2, patch_artist=True,
                showfliers=False, zorder=2,
                medianprops={"color": INK, "linewidth": 1.2},
                whiskerprops={"color": color, "linewidth": 1},
                capprops={"color": color, "linewidth": 1},
            )
            box["boxes"][0].set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1)
            jitter = rng.uniform(-0.06, 0.06, size=len(drops))
            ax.scatter(pos + jitter, drops, s=9, color=color, alpha=0.75,
                       edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("pass@1 drop (pp)")
    ax.set_title("(a) Drop distribution per dataset", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.legend(handles=[Patch(facecolor=c, label=p) for p, c in PREC_COLORS.items()],
              frameon=False, loc="upper left", title="Precision", title_fontsize=7)


def draw_means(ax, cases: list) -> None:
    """Panel (b): mean drop per dataset, grouped by precision, direct-labeled."""
    precs = list(PREC_COLORS)
    offsets = {"FP": -0.26, "8-bit": 0.0, "4-bit": 0.26}
    for ds_idx, dataset in enumerate(DATASETS):
        for prec in precs:
            drops = [c["drop"] for c in cases
                     if c["drop"] is not None and c["dataset"] == dataset and c["prec"] == prec]
            if not drops:
                continue
            mean = float(np.mean(drops))
            pos = ds_idx + offsets[prec]
            ax.bar(pos, mean, width=0.22, color=PREC_COLORS[prec], zorder=2)
            ax.text(pos, mean + (0.4 if mean >= 0 else -0.4), f"{mean:.1f}",
                    ha="center", va="bottom" if mean >= 0 else "top",
                    fontsize=6.5, color=INK)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("mean pass@1 drop (pp)")
    ax.set_title("(b) Mean drop per dataset", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)


def draw_heatmap(ax, cases: list) -> None:
    """Panel (c): per-case heatmap — rows are model x precision, columns dataset x score."""
    lookup = {(c["dataset"], c["model"], c["prec"], c["score"]): c["drop"] for c in cases}
    rows = []  # (family, model, prec)
    family_bounds = []  # row index where each family ends
    for family, models in MODELS.items():
        for model_path in models:
            model = model_path.split("/")[-1]
            for quant in quant_types_for(family):
                rows.append((family, model, QUANT_LABELS[quant]))
        family_bounds.append(len(rows))
    cols = [(ds, score) for ds in DATASETS for score in SCORES]

    grid = np.full((len(rows), len(cols)), np.nan)
    for i, (_, model, prec) in enumerate(rows):
        for j, (ds, score) in enumerate(cols):
            drop = lookup.get((ds, model, prec, score))
            if drop is not None:
                grid[i, j] = drop

    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = DIVERGING_CMAP.copy()
    cmap.set_bad(MISSING_FILL)
    mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                         edgecolors="white", linewidth=1.2)
    ax.invert_yaxis()

    for i in range(len(rows)):
        for j in range(len(cols)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center",
                        fontsize=5.5, color=MUTED)
            else:
                ax.text(j + 0.5, i + 0.5, f"{v:+.1f}", ha="center", va="center",
                        fontsize=5.5, color=_text_ink(cmap(norm(v))))

    for bound in family_bounds[:-1]:
        ax.axhline(bound, color=INK, linewidth=1.4)
    for j in range(len(SCORES), len(cols), len(SCORES)):
        ax.axvline(j, color=INK, linewidth=1.4)

    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{model} ({prec})" for _, model, prec in rows], fontsize=5.8)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([SCORE_LABELS[score] for _, score in cols], fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for ds_idx, dataset in enumerate(DATASETS):
        ax.text(ds_idx * len(SCORES) + len(SCORES) / 2, -1.1, DATASET_LABELS[dataset],
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(c) Drop per exp case (pp)", loc="left", fontweight="bold", pad=28)

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, aspect=45)
    cbar.set_label("pass@1 drop (pp)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)


def build_figure(cases: list, attack: str, out_stem: Path) -> None:
    """Compose the three panels into one figure and save PNG + PDF."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
    })
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.85, 1.0], height_ratios=[1, 1],
                          left=0.06, right=0.97, top=0.9, bottom=0.05,
                          hspace=0.28, wspace=0.42)
    draw_distribution(fig.add_subplot(gs[0, 0]), cases)
    draw_means(fig.add_subplot(gs[1, 0]), cases)
    draw_heatmap(fig.add_subplot(gs[:, 1]), cases)
    fig.suptitle(f"pass@1 drop under {attack} attack (clean − attacked, percentage points)",
                 fontsize=10, fontweight="bold")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_stem}.png and {out_stem}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistic pass@1 drop under an attack.")
    parser.add_argument("--attack", default="destructure",
                        help="Attack folder name under each model dir (default: destructure)")
    args = parser.parse_args()

    cases = collect_cases(args.attack)
    csv_path = ROOT / "statistic_results" / f"{args.attack}_drop.csv"
    write_csv(cases, csv_path)
    print(f"Saved per-case CSV to {csv_path}")
    print_summary(cases, args.attack)
    build_figure(cases, args.attack, ROOT / "plots" / f"{args.attack}_drop")


if __name__ == "__main__":
    main()
