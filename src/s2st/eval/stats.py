"""Statistical tests: Bootstrap 95% CIs, paired Wilcoxon signed-rank, and Spearman rank correlation."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def bootstrap_ci_mean(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute sample mean, SD, and non-parametric percentile bootstrap confidence interval."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if len(arr) == 1:
        val = float(arr[0])
        return {"mean": val, "std": 0.0, "ci_low": val, "ci_high": val, "n": 1}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(boot_means, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "n": int(len(arr)),
    }


def paired_wilcoxon_and_effect_size(
    x: Sequence[float], y: Sequence[float]
) -> Dict[str, float | None]:
    """Run two-sided paired Wilcoxon signed-rank test and compute rank-biserial effect size."""
    from scipy.stats import wilcoxon

    pairs: List[Tuple[float, float]] = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b))
    ]
    if len(pairs) < 5:
        return {"n_pairs": len(pairs), "p_value": None, "mean_diff": None, "rank_biserial_r": None}

    arr_x = np.array([p[0] for p in pairs], dtype=np.float64)
    arr_y = np.array([p[1] for p in pairs], dtype=np.float64)
    diffs = arr_x - arr_y
    if np.allclose(diffs, 0.0):
        return {"n_pairs": len(pairs), "p_value": 1.0, "mean_diff": 0.0, "rank_biserial_r": 0.0}

    res = wilcoxon(arr_x, arr_y, zero_method="wilcox", alternative="two-sided")
    n_non_zero = int(np.sum(diffs != 0))
    total_ranks = n_non_zero * (n_non_zero + 1) / 2.0
    rank_biserial = float(1.0 - (2.0 * res.statistic) / max(total_ranks, 1.0))
    return {
        "n_pairs": len(pairs),
        "mean_diff": round(float(diffs.mean()), 4),
        "p_value": round(float(res.pvalue), 6),
        "rank_biserial_r": round(rank_biserial, 4),
    }


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float | None]:
    """Compute Spearman rank correlation rho and p-value."""
    from scipy.stats import spearmanr

    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b))
    ]
    if len(pairs) < 4:
        return {"n": len(pairs), "rho": None, "p_value": None}
    ax = [p[0] for p in pairs]
    ay = [p[1] for p in pairs]
    rho, pval = spearmanr(ax, ay)
    return {"n": len(pairs), "rho": round(float(rho), 4), "p_value": round(float(pval), 6)}
