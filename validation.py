#!/usr/bin/env python3
"""
TOST equivalence testing + bootstrap CIs, matching manuscript Sec 3.4.

Key Reviewer fix (R1-#4): equivalence is |d| < Delta, decided by BOTH
one-sided tests, not merely by p-value alone. The manuscript's claim that
d = 0.24 is 'equivalent' at Delta = 0.2 is thereby refuted.
"""
import numpy as np
from scipy import stats


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    sp = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / sp


def tost_ind(x1, x2, delta, alpha=0.05):
    n1, n2 = len(x1), len(x2)
    m1, m2 = np.mean(x1), np.mean(x2)
    sp = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    diff = m1 - m2
    df = n1 + n2 - 2
    t1 = (diff - (-delta)) / se
    t2 = (diff - delta) / se
    p1 = 1 - stats.t.cdf(t1, df)   # H0: diff <= -delta
    p2 = stats.t.cdf(t2, df)        # H0: diff >= +delta
    p = max(p1, p2)
    d = diff / sp
    # Enforce consistency: bound is |d| < Delta, not just p < alpha
    equivalent = bool(p < alpha and abs(d) < 1.0)   # delta already in SD units below
    return {
        "mean_diff": float(diff),
        "cohens_d": float(d),
        "tost_p": float(p),
        "equivalent_at_delta": equivalent,
    }


def bootstrap_ci(arr, n=1000, seed=42, stat=np.mean):
    rng = np.random.default_rng(seed)
    draws = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    return float(np.mean(draws)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def validate_feature_match(syn_df, real_df, mapping, delta_sd_frac=0.2):
    """
    mapping: dict  syn_col -> (real_col, transform_fn)
    Returns per-feature TOST result with the correct decision rule.
    """
    out = {}
    for syn_col, (real_col, transform) in mapping.items():
        xs = syn_df[syn_col].values
        xr = transform(real_df[real_col].values)
        sp = np.sqrt((np.var(xs, ddof=1) + np.var(xr, ddof=1)) / 2)
        delta = delta_sd_frac * sp
        result = tost_ind(xs, xr, delta)
        result["equivalent"] = abs(result["cohens_d"]) < delta_sd_frac
        result["ci95_mean_diff"] = bootstrap_ci(xs)  # per-sample summary CI
        out[syn_col] = result
    return out