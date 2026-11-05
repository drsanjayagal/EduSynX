#!/usr/bin/env python3
"""
TOST equivalence testing with the CORRECT decision rule.

Reviewer 1 (#4): the manuscript reports d=0.24 at Δ=0.2 as "equivalent",
which cannot hold under TOST. The correct rule is:

    equivalent  iff  (p_tost < alpha)  AND  (|Cohen's d| < delta_frac)

Both conditions must hold. Also reports the CI for the mean difference.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    sp = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) +
                  (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / sp if sp > 0 else 0.0


def tost(x1, x2, delta_frac=0.2, alpha=0.05):
    """Return dict with mean_diff, cohens_d, p_tost, equivalent (correct rule)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
    if len(x1) < 3 or len(x2) < 3:
        return {"mean_diff": np.nan, "cohens_d": np.nan, "p_tost": np.nan,
                "equivalent": False, "note": "insufficient samples"}

    n1, n2 = len(x1), len(x2)
    m1, m2 = np.mean(x1), np.mean(x2)
    sp = np.sqrt(((n1 - 1) * np.var(x1, ddof=1) +
                  (n2 - 1) * np.var(x2, ddof=1)) / (n1 + n2 - 2))
    se = sp * np.sqrt(1/n1 + 1/n2)
    df = n1 + n2 - 2
    diff = m1 - m2
    delta = delta_frac * sp  # bound in raw units

    # Two one-sided t-tests
    t_lower = (diff - (-delta)) / se   # H0: diff <= -delta  (want to reject)
    t_upper = (diff -    delta) / se   # H0: diff >= +delta  (want to reject)
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)

    d = diff / sp if sp > 0 else 0.0
    # CORRECTED decision rule: BOTH conditions must hold
    equivalent = bool(p_tost < alpha and abs(d) < delta_frac)

    # 95% CI on mean difference
    crit = stats.t.ppf(1 - alpha/2, df)
    ci_lo, ci_hi = diff - crit*se, diff + crit*se

    return {
        "n_syn": int(n1), "n_real": int(n2),
        "mean_syn": float(m1), "mean_real": float(m2),
        "mean_diff": float(diff),
        "ci95_diff_lo": float(ci_lo), "ci95_diff_hi": float(ci_hi),
        "cohens_d": float(d),
        "p_tost": float(p_tost),
        "equivalent": equivalent,
        "delta_bound_raw": float(delta),
        "note": ("equiv requires p<alpha AND |d|<delta_frac"
                 if equivalent else
                 f"NOT equivalent: |d|={abs(d):.3f} vs bound {delta_frac}"
                 if abs(d) >= delta_frac else
                 f"NOT equivalent: p={p_tost:.4f} >= alpha"),
    }


def proportion_tost(p1, n1, p2, n2, delta=0.05, alpha=0.05):
    """TOST for proportions (e.g. categorical studytime)."""
    diff = p1 - p2
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    if se == 0:
        return {"p_tost": 1.0, "equivalent": False, "diff": 0.0}
    z_l = (diff + delta) / se
    z_u = (diff - delta) / se
    p_l = 1 - stats.norm.cdf(z_l)
    p_u = stats.norm.cdf(z_u)
    p_tost = max(p_l, p_u)
    equivalent = bool(p_tost < alpha and abs(diff) < delta)
    return {"diff": float(diff), "p_tost": float(p_tost),
            "equivalent": equivalent,
            "note": ("ok" if equivalent else f"|diff|={abs(diff):.3f} vs {delta}")}


# ---------------------------------------------------------------------------
# Feature mappings (manuscript Table 5)
# ---------------------------------------------------------------------------

def load_uci(path_dir):
    """Return concatenated student-mat + student-por."""
    d = Path(path_dir)
    frames = []
    for f in ["student-mat.csv", "student-por.csv"]:
        p = d / f
        if p.exists():
            frames.append(pd.read_csv(p, sep=";"))
    if not frames:
        raise SystemExit(f"No student-mat.csv / student-por.csv in {d}")
    return pd.concat(frames, ignore_index=True)


def map_uci(df):
    """Map UCI columns to our synthetic feature names.
    Returns DataFrame with: study_hours_cat (1-4), attendance (0-100),
    previous_gpa (0-10), final_grade (0-20)."""
    out = pd.DataFrame()
    out["study_hours_cat"] = df["studytime"].astype(int)          # 1-4 already
    out["attendance"] = np.clip(100 - df["absences"].astype(float), 0, 100)
    out["previous_gpa"] = ((df["G1"] + df["G2"]) / 2) / 20 * 10    # avg -> 0-10
    out["final_grade"] = df["G3"].astype(float)
    return out


def map_oulad(oulad_dir):
    """Return per-student: lms_clicks (log), submission_delay_days,
    completion_rate, final_result_3class."""
    d = Path(oulad_dir)
    info = pd.read_csv(d / "studentInfo.csv")
    ass = pd.read_csv(d / "studentAssessment.csv")
    assessments = pd.read_csv(d / "assessments.csv")
    vle = pd.read_csv(d / "studentVle.csv")

    # LMS clicks per student
    clicks = vle.groupby("id_student")["sum_click"].sum().rename("lms_clicks")

    # --- FIX: coerce date columns to numeric, drop "NA" strings ----------
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    ass["date_submitted"] = pd.to_numeric(ass["date_submitted"], errors="coerce")

    merged = ass.merge(assessments[["id_assessment", "date"]],
                       on="id_assessment", how="left")
    merged["delay_days"] = merged["date_submitted"] - merged["date"]
    merged = merged.dropna(subset=["delay_days"])
    delay = merged.groupby("id_student")["delay_days"].mean().rename("mean_delay_days")

    # Completion rate: submissions / total assessments in that module
    n_assessments = assessments.groupby("code_module").size().rename("n_assess")
    n_submitted = ass.groupby(["id_student"]).size().rename("n_sub")
    sub = n_submitted.to_frame().join(info.set_index("id_student")[["code_module"]])
    sub = sub.join(n_assessments, on="code_module")
    sub["completion_rate"] = (sub["n_sub"] / sub["n_assess"]).clip(0, 1)

    out = info.set_index("id_student").join(clicks).join(delay).join(
        sub["completion_rate"])
    out = out.reset_index()
    out["lms_clicks_log"] = np.log1p(out["lms_clicks"].fillna(0))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--syn-data", default="data/student_chunk_00000.csv")
    p.add_argument("--uci-dir", default="data/uci")
    p.add_argument("--oulad-dir", default="data/oulad")
    p.add_argument("--out", default="validation/tost_results.csv")
    p.add_argument("--delta", type=float, default=0.2,
                   help="equivalence bound in SD units")
    return p.parse_args()


def main():
    a = parse_args()
    syn = pd.read_csv(a.syn_data, nrows=500_000)

    rows = []

    # ---------- UCI ----------
    print("=== UCI Student Performance ===")
    try:
        uci_raw = load_uci(a.uci_dir)
        uci = map_uci(uci_raw)
        print(f"  loaded {len(uci)} UCI rows")

        # Continuous: attendance
        r = tost(syn["attendance"].values, uci["attendance"].values,
                 delta_frac=a.delta)
        r.update({"dataset": "UCI", "feature": "attendance",
                  "comparison_type": "continuous"})
        rows.append(r); print(f"  attendance: {r['note']}")

        # Continuous: previous_gpa
        r = tost(syn["previous_gpa"].values, uci["previous_gpa"].values,
                 delta_frac=a.delta)
        r.update({"dataset": "UCI", "feature": "previous_gpa",
                  "comparison_type": "continuous"})
        rows.append(r); print(f"  previous_gpa: {r['note']}")

        # Categorical: studytime proportions per category
        for cat in [1, 2, 3, 4]:
            p_syn = (syn.get("study_hours_cat",
                             pd.Series([])).eq(cat).mean()
                     if "study_hours_cat" in syn.columns else None)
            # Approximate synthetic studytime category from hours
            bins = [0, 2, 5, 10, np.inf]
            syn_cat = pd.cut(syn["study_hours"], bins=bins, labels=[1,2,3,4])
            p_syn = (syn_cat == cat).mean()
            p_uci = (uci["study_hours_cat"] == cat).mean()
            r = proportion_tost(p_syn, len(syn), p_uci, len(uci),
                               delta=0.05)
            r.update({"dataset": "UCI",
                      "feature": f"study_hours_cat={cat}",
                      "comparison_type": "proportion",
                      "mean_syn": float(p_syn), "mean_real": float(p_uci)})
            rows.append(r)
    except SystemExit as e:
        print(f"  SKIP UCI: {e}")

    # ---------- OULAD ----------
    print("\n=== OULAD ===")
    try:
        oulad = map_oulad(a.oulad_dir)
        print(f"  loaded {len(oulad)} OULAD rows")

        # LMS usage (log)
        r = tost(np.log1p(syn["lms_usage_hours"].values),
                 oulad["lms_clicks_log"].values, delta_frac=a.delta)
        r.update({"dataset": "OULAD", "feature": "lms_usage_log",
                  "comparison_type": "continuous"})
        rows.append(r); print(f"  lms_usage_log: {r['note']}")

        # Submission delay
        r = tost(syn["submission_delay"].values,
                 oulad["mean_delay_days"].values, delta_frac=a.delta)
        r.update({"dataset": "OULAD", "feature": "submission_delay",
                  "comparison_type": "continuous"})
        rows.append(r); print(f"  submission_delay: {r['note']}")

        # Completion rate
        r = tost(syn["assignment_completion_rate"].values,
                 oulad["completion_rate"].dropna().values, delta_frac=a.delta)
        r.update({"dataset": "OULAD", "feature": "completion_rate",
                  "comparison_type": "continuous"})
        rows.append(r); print(f"  completion_rate: {r['note']}")
    except SystemExit as e:
        print(f"  SKIP OULAD: {e}")
    except Exception as e:
        print(f"  OULAD failed: {e}")

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nSaved {len(df)} TOST results to {a.out}")

    # Summary
    if not df.empty:
        eq_count = df["equivalent"].sum()
        print(f"\nEquivalence summary: {eq_count}/{len(df)} features pass at "
              f"Δ={a.delta}. Note the CORRECT decision rule requires "
              f"|d| < Δ AND p < alpha.")


if __name__ == "__main__":
    main()