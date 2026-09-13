"""Statistical analysis of adversarial and noise attack results.

Part 1 – Adversarial attacks:
    For each (model, dataset, attack, metric), compute:
        delta_fp   = |pass@1_clean_fp   - pass@1_attack_fp|
        delta_quant = |pass@1_clean_quant - pass@1_attack_quant|
    Then test whether median(delta_fp - delta_quant) > 0 via one-sided
    Wilcoxon signed-rank test, with bootstrap CI on the median.

Part 2 – Noise robustness:
    For each noise level, collect paired (FP pass@1, 8-bit pass@1) across all
    models/datasets/noise-types, run a paired Wilcoxon signed-rank test, and
    report where the difference is statistically significant.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metadata import (
    ATTACKS,
    DATASETS,
    MODELS,
    NOISE_LEVELS,
    NOISE_TYPES,
    QUANTIZED_TYPES,
    outputs_dir,
)

OUTPUTS_DIR = outputs_dir()


# ── Shared helpers ────────────────────────────────────────────────────────────


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.median,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1-D array of observations.
        stat_fn: Statistic function (default: median).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (two-sided).
        seed: Random seed.

    Returns:
        (lower, upper) bounds of the (1 - alpha) CI.
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.array(
        [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── Part 1: Adversarial delta analysis ────────────────────────────────────────


def _load_pass_rates(folder: Path) -> tuple[float, float] | None:
    """Load (base, plus) pass@1 from pass_rates.json.

    Args:
        folder: Directory containing pass_rates.json.

    Returns:
        (base, plus) as floats, or None if unavailable.
    """
    path = folder / "pass_rates.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        base_val = data.get("base")
        plus_val = data.get("plus")
        if base_val is None or plus_val is None:
            return None
        return float(base_val), float(plus_val)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _collect_deltas() -> np.ndarray:
    """Collect all (delta_fp - delta_quant) values across models/datasets/attacks.

    For each combination, both base and plus metrics contribute one sample each.

    Returns:
        1-D array of (delta_fp - delta_quant) values.
    """
    diffs = []
    for family, model_paths in MODELS.items():
        quant_key = "bnb4" if family == "MOE" else "bnb8"
        for model_path in model_paths:
            org, model_name = model_path.split("/", 1)
            for dataset in DATASETS:
                model_dir = OUTPUTS_DIR / dataset / org / model_name
                # Clean baseline: noise/gaussian/0.0/{quant}
                clean_fp = _load_pass_rates(model_dir / "noise" / "gaussian" / "0.0" / "base")
                clean_q = _load_pass_rates(model_dir / "noise" / "gaussian" / "0.0" / quant_key)
                if clean_fp is None or clean_q is None:
                    continue

                for attack in ATTACKS:
                    atk_fp = _load_pass_rates(model_dir / attack / "base")
                    atk_q = _load_pass_rates(model_dir / attack / quant_key)
                    if atk_fp is None or atk_q is None:
                        continue

                    # base metric (idx 0) and plus metric (idx 1)
                    for idx in range(2):
                        delta_fp = abs(clean_fp[idx] - atk_fp[idx])
                        delta_q = abs(clean_q[idx] - atk_q[idx])
                        diffs.append(delta_fp - delta_q)

    return np.array(diffs)


def analyse_adversarial() -> None:
    """Run Wilcoxon signed-rank test on delta_fp - delta_quant."""
    print("=" * 70)
    print("Part 1: Adversarial – Wilcoxon signed-rank test")
    print("  delta_fp - delta_quant, H1: median > 0")
    print("  (delta = |pass@1_clean - pass@1_attack|)")
    print("=" * 70)

    diffs = _collect_deltas()
    print(f"Total paired samples: {len(diffs)}")

    median_d = float(np.median(diffs))
    mean_d = float(np.mean(diffs))
    ci_lo, ci_hi = bootstrap_ci(diffs)

    # Filter zeros for Wilcoxon
    diffs = np.round(diffs, 9)
    nonzero = diffs[diffs != 0.0]
    print(f"Non-zero differences: {len(nonzero)}")

    # One-sided Wilcoxon signed-rank test: H1: median > 0
    stat, p_val = stats.wilcoxon(nonzero, alternative="greater")

    print(f"\nMedian (Δ_fp - Δ_quant): {median_d:.6f}")
    print(f"Mean   (Δ_fp - Δ_quant): {mean_d:.6f}")
    print(f"95% Bootstrap CI:        [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Wilcoxon stat:           {stat:.1f}")
    print(f"p-value (H1: median > 0): {p_val:.6f}")
    if p_val < 0.05:
        print("→ Reject H0 at α=0.05: FP models degrade more than quantized under attack.")
    else:
        print("→ Fail to reject H0 at α=0.05.")
    print()


# ── Part 2: Noise attack paired analysis ─────────────────────────────────────




def analyse_noise() -> None:
    """For each noise level, run paired Wilcoxon test: FP vs 8-bit pass@1."""
    print("=" * 70)
    print("Part 2: Noise attack – Paired Wilcoxon test (FP vs 8-bit pass@1)")
    print("=" * 70)
    print("For each noise level, we pair FP and 8-bit (or 4-bit for MOE)")
    print("pass@1 values across all models × datasets × noise types.\n")

    # Skip level 0.0 as baseline — test on each non-zero level
    for level in NOISE_LEVELS:
        fp_vals = []
        quant_vals = []

        for family, model_paths in MODELS.items():
            quant_key = "bnb4" if family == "MOE" else "bnb8"
            for model_path in model_paths:
                org, model_name = model_path.split("/", 1)
                for dataset in DATASETS:
                    # Skip canitedit for CodeGen (no results)
                    for noise_type in NOISE_TYPES:
                        fp_dir = (
                            OUTPUTS_DIR / dataset / org / model_name
                            / "noise" / noise_type / level / "base"
                        )
                        q_dir = (
                            OUTPUTS_DIR / dataset / org / model_name
                            / "noise" / noise_type / level / quant_key
                        )
                        fp_pair = _load_pass_rates(fp_dir)
                        q_pair = _load_pass_rates(q_dir)
                        fp_val = fp_pair[0] if fp_pair else None
                        q_val = q_pair[0] if q_pair else None
                        if fp_val is not None and q_val is not None:
                            fp_vals.append(fp_val)
                            quant_vals.append(q_val)

        fp_arr = np.array(fp_vals)
        q_arr = np.array(quant_vals)
        diff = fp_arr - q_arr

        n_pairs = len(fp_vals)
        if n_pairs == 0:
            print(f"Level {level:>5s}: no paired data")
            continue

        median_diff = float(np.median(diff))
        mean_fp = float(np.mean(fp_arr))
        mean_q = float(np.mean(q_arr))

        # Paired Wilcoxon signed-rank test (two-sided)
        # Filter out zero differences (Wilcoxon requires non-zero)
        nonzero = diff[diff != 0.0]
        if len(nonzero) < 2:
            print(
                f"Level {level:>5s}: n={n_pairs:3d}  "
                f"mean FP={mean_fp:.4f}  mean 8bit={mean_q:.4f}  "
                f"median Δ={median_diff:.4f}  (too few non-zero diffs for test)"
            )
            continue

        stat, p_val = stats.wilcoxon(nonzero)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        print(
            f"Level {level:>5s}: n={n_pairs:3d}  "
            f"mean FP={mean_fp:.4f}  mean 8bit={mean_q:.4f}  "
            f"median Δ={median_diff:.4f}  "
            f"W={stat:.0f}  p={p_val:.6f}  {sig}"
        )

    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run both analyses."""
    analyse_adversarial()
    analyse_noise()


if __name__ == "__main__":
    main()
