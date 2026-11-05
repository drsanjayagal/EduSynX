#!/usr/bin/env python3
"""Produce validation/statistical_summary.csv and class_distribution.csv.

These are the ACTUAL statistics from the generated data, with explicit
deltas against the manuscript's Table 4 — the honest replacement requested
by Reviewer 1 (comments 3-4) and Reviewer 2 (text vs figure correlation).

Nothing is fabricated. If the manuscript and data disagree, the disagreement
is written to the CSV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Manuscript Table 4 (from article .tex)
MANUSCRIPT_TABLE4 = {
    "study_hours":     {"mean": 10.02, "std": 4.21,  "min": 0.00,  "max": 28.34},
    "attendance":      {"mean": 69.97, "std": 8.89,  "min": 42.15, "max": 98.76},
    "internal_marks":  {"mean": 50.12, "std": 12.34, "min": 8.56,  "max": 92.34},
    "previous_gpa":    {"mean": 6.01,  "std": 0.87,  "min": 3.21,  "max": 8.99},
    "stress_level":    {"mean": 4.98,  "std": 1.42,  "min": 1.00,  "max": 9.87},
    "sleep_hours":     {"mean": 6.99,  "std": 0.89,  "min": 4.12,  "max": 9.87},
    "lms_usage_hours": {"mean": 5.01,  "std": 1.89,  "min": 0.12,  "max": 14.56},
}

# Manuscript-reported correlations (Reviewer 2 flagged 0.42/0.35 in text
# vs 0.35/0.27 in Fig 12 — both must be compared to reality).
MANUSCRIPT_CORRELATIONS = {
    ("study_hours", "internal_marks"): 0.42,
    ("study_hours", "attendance"):     0.38,
    ("study_hours", "previous_gpa"):   0.35,
}

# Manuscript target distribution (abstract claim)
MANUSCRIPT_CLASS_DIST = {
    "Distinction": 0.20, "Pass": 0.40, "Fail": 0.25, "Dropout": 0.15,
}

# Manuscript completion-rate claim (Reviewer 1 comment 4: 0.78 vs 0.70)
MANUSCRIPT_COMPLETION_RATE_MEAN = 0.78


def load_all_chunks(data_dir, max_rows=None):
    files = sorted(Path(data_dir).glob("student_chunk_*.csv"))
    if not files:
        single = Path(data_dir) / "student_data.csv"
        if single.exists():
            files = [single]
        else:
            raise SystemExit(f"No chunk files in {data_dir}")
    dfs, total = [], 0
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        total += len(df)
        if max_rows and total >= max_rows:
            break
    full = pd.concat(dfs, ignore_index=True)
    if max_rows and len(full) > max_rows:
        full = full.sample(n=max_rows, random_state=42)
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap for huge datasets (None = all)")
    p.add_argument("--out-dir", default="validation")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    df = load_all_chunks(args.data_dir, args.max_rows)
    print(f"  loaded {len(df):,} rows, {len(df.columns)} columns")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---- Statistical summary --------------------------------------------
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        row = {
            "feature": col,
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "q50": float(s.quantile(0.50)),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }
        if col in MANUSCRIPT_TABLE4:
            m = MANUSCRIPT_TABLE4[col]
            row["manuscript_mean"] = m["mean"]
            row["manuscript_std"] = m["std"]
            row["manuscript_min"] = m["min"]
            row["manuscript_max"] = m["max"]
            row["delta_mean"] = row["mean"] - m["mean"]
            row["delta_std"] = row["std"] - m["std"]
            row["std_matches_manuscript"] = bool(abs(row["delta_std"]) < 0.5)
        else:
            row["manuscript_mean"] = np.nan
            row["manuscript_std"] = np.nan
            row["manuscript_min"] = np.nan
            row["manuscript_max"] = np.nan
            row["delta_mean"] = np.nan
            row["delta_std"] = np.nan
            row["std_matches_manuscript"] = np.nan
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out / "statistical_summary.csv", index=False)
    print(f"  -> {out / 'statistical_summary.csv'}")

    # ---- Correlation structure (Reviewer 2 fix) ------------------------
    corr_rows = []
    for (a, b), reported_r in MANUSCRIPT_CORRELATIONS.items():
        if a in df.columns and b in df.columns:
            actual_r = float(df[[a, b]].corr().iloc[0, 1])
            corr_rows.append({
                "pair": f"{a}__{b}",
                "manuscript_r": reported_r,
                "actual_r": actual_r,
                "delta": actual_r - reported_r,
                "matches_within_0.05": bool(abs(actual_r - reported_r) < 0.05),
            })
    pd.DataFrame(corr_rows).to_csv(out / "correlation_check.csv", index=False)
    print(f"  -> {out / 'correlation_check.csv'}")

    # ---- Class distribution --------------------------------------------
    if "final_result" in df.columns:
        counts = df["final_result"].value_counts()
        props = df["final_result"].value_counts(normalize=True)
        class_rows = []
        for cls, m_prop in MANUSCRIPT_CLASS_DIST.items():
            class_rows.append({
                "class": cls,
                "actual_count": int(counts.get(cls, 0)),
                "actual_proportion": float(props.get(cls, 0.0)),
                "manuscript_proportion": m_prop,
                "delta": float(props.get(cls, 0.0)) - m_prop,
                "matches_within_0.03": bool(abs(props.get(cls, 0.0) - m_prop) < 0.03),
            })
        pd.DataFrame(class_rows).to_csv(out / "class_distribution.csv", index=False)
        print(f"  -> {out / 'class_distribution.csv'}")

    # ---- Completion rate (Reviewer 1 comment 4) ------------------------
    if "assignment_completion_rate" in df.columns:
        s = df["assignment_completion_rate"].dropna()
        # If stored as percentage (0-100), convert to 0-1 for comparison
        mean_raw = float(s.mean())
        mean_01 = mean_raw / 100.0 if mean_raw > 1.5 else mean_raw
        pd.DataFrame([{
            "feature": "assignment_completion_rate",
            "mean_raw": mean_raw,
            "mean_normalised_0_1": mean_01,
            "manuscript_mean": MANUSCRIPT_COMPLETION_RATE_MEAN,
            "delta": mean_01 - MANUSCRIPT_COMPLETION_RATE_MEAN,
            "note": ("Manuscript reports 0.78. Table 1 equation "
                     "(0.7 + 0.1*C + 0.05*M) implies mean 0.70. "
                     "Reviewer 1 (comment 4) flagged this. This row is the "
                     "actual value from the regenerated data."),
        }]).to_csv(out / "completion_rate_check.csv", index=False)
        print(f"  -> {out / 'completion_rate_check.csv'}")

    # ---- Console summary of discrepancies ------------------------------
    print("\n=== Manuscript vs actual (Table 4 features) ===")
    for _, row in stats_df[stats_df["manuscript_mean"].notna()].iterrows():
        flag = "MATCH" if row["std_matches_manuscript"] else "MISMATCH"
        print(f"  {row['feature']:<20} "
              f"mean {row['mean']:>7.3f} (manu {row['manuscript_mean']:>7.3f}, "
              f"delta {row['delta_mean']:>+7.3f})  "
              f"std {row['std']:>6.3f} (manu {row['manuscript_std']:>6.3f}, "
              f"delta {row['delta_std']:>+6.3f})  [{flag}]")

    print("\n=== Correlations ===")
    for r in corr_rows:
        flag = "MATCH" if r["matches_within_0.05"] else "MISMATCH"
        print(f"  {r['pair']:<35} "
              f"actual {r['actual_r']:>+6.3f}  (manuscript {r['manuscript_r']:>+6.3f}, "
              f"delta {r['delta']:>+6.3f})  [{flag}]")

    print(f"\nAll files written to {out}/")


if __name__ == "__main__":
    main()