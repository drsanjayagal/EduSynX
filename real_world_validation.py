#!/usr/bin/env python3
"""
Real-world transfer (Reviewer 2). Trains on synthetic, evaluates on real.

Explicitly reports two regimes:
  (a) MATCHED-ONLY  : train on the 3 features that overlap with UCI/OULAD
  (b) ALL-WITH-IMPUTATION : train on all 44 synthetic features, zero-impute
                            the 41 features that do not exist in the real
                            dataset (Reviewer 2 flagged the ambiguity)

Both are honest; the paper must state which is reported.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from tost_validation import load_uci, map_uci, map_oulad


def bootstrap_metrics(y_true, y_pred, n_boot=1000, seed=42):
    """95% bootstrap CIs for accuracy and macro F1."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    accs, f1s = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        accs.append(accuracy_score(y_true[s], y_pred[s]))
        f1s.append(f1_score(y_true[s], y_pred[s], average="macro", zero_division=0))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci_lo": float(np.percentile(accs, 2.5)),
        "accuracy_ci_hi": float(np.percentile(accs, 97.5)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_ci_lo": float(np.percentile(f1s, 2.5)),
        "f1_macro_ci_hi": float(np.percentile(f1s, 97.5)),
    }


def discretize_grade(grades, quartiles_from=None):
    """Map continuous grades to 4 classes matching the synthetic schema."""
    q = np.quantile(grades, [0.25, 0.50, 0.75]) if quartiles_from is None \
        else np.quantile(quartiles_from, [0.25, 0.50, 0.75])
    out = np.select([grades >= q[2], grades >= q[1], grades >= q[0]],
                    ["Distinction", "Pass", "Fail"], default="Dropout")
    return out


def run_one(dataset_name, syn_df, real_df, matched_features,
            feature_imputation_default=0.0):
    """Train on synthetic, eval on real, for both transfer regimes."""
    from sklearn.preprocessing import OrdinalEncoder
    results = []

    # Build common labels
    y_syn = syn_df["final_result"].values
    y_real = real_df["final_result"].values

    # ---- Regime A: MATCHED-ONLY ----------------------------
    syn_A = syn_df[matched_features].astype(float).values
    real_A = real_df[matched_features].astype(float).values

    model_A = LGBMClassifier(n_estimators=300, num_leaves=63,
                             random_state=42, verbose=-1, class_weight="balanced")
    model_A.fit(syn_A, y_syn)
    pred_A = model_A.predict(real_A)
    metrics_A = bootstrap_metrics(y_real, pred_A)
    metrics_A.update({"dataset": dataset_name, "regime": "matched_only",
                      "n_features": len(matched_features)})
    results.append(metrics_A)

    # ---- Regime B: ALL 44 WITH IMPUTATION ----------------
    syn_features = [c for c in syn_df.columns
                    if c not in ("student_id", "final_result")]

    # Ordinal-encode categoricals so LightGBM receives numeric input
    syn_enc = syn_df[syn_features].copy()
    cat_cols = syn_enc.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        syn_enc[cat_cols] = oe.fit_transform(syn_enc[cat_cols])
    syn_B = syn_enc.astype(float).values

    # Build real_B with imputation default, then copy matched features
    real_B = pd.DataFrame(
        np.full((len(real_df), len(syn_features)), feature_imputation_default),
        columns=syn_features)
    for f in matched_features:
        real_B[f] = real_df[f].astype(float).values
    real_B_arr = real_B.astype(float).values

    model_B = LGBMClassifier(n_estimators=300, num_leaves=63,
                             random_state=42, verbose=-1, class_weight="balanced")
    model_B.fit(syn_B, y_syn)
    pred_B = model_B.predict(real_B_arr)
    metrics_B = bootstrap_metrics(y_real, pred_B)
    metrics_B.update({"dataset": dataset_name,
                      "regime": "all_with_imputation",
                      "n_features": len(syn_features)})
    results.append(metrics_B)

    # ---- Regime C: REAL-ONLY (baseline upper bound) ------
    X_tr, X_te, y_tr, y_te = train_test_split(
        real_A, y_real, test_size=0.3, stratify=y_real, random_state=42)
    model_C = LGBMClassifier(n_estimators=300, num_leaves=63,
                             random_state=42, verbose=-1, class_weight="balanced")
    model_C.fit(X_tr, y_tr)
    pred_C = model_C.predict(X_te)
    metrics_C = bootstrap_metrics(y_te, pred_C)
    metrics_C.update({"dataset": dataset_name, "regime": "real_only",
                      "n_features": len(matched_features)})
    results.append(metrics_C)

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--syn-data", default="data/student_chunk_00000.csv")
    p.add_argument("--uci-dir", default="data/uci")
    p.add_argument("--oulad-dir", default="data/oulad")
    p.add_argument("--out", default="validation/real_world_validation.csv")
    return p.parse_args()


def main():
    a = parse_args()
    syn = pd.read_csv(a.syn_data, nrows=500_000)
    rows = []

    # ---------- UCI ----------
    try:
        uci = map_uci(load_uci(a.uci_dir))
        uci["final_result"] = discretize_grade(uci["final_grade"].values)
        # Map UCI studytime category to a continuous proxy for synthetic
        print(f"\n=== UCI ({len(uci)} rows) ===")
        r = run_one("UCI", syn, uci,
                    matched_features=["attendance", "previous_gpa"])
        for x in r:
            print(f"  {x['regime']:<22} acc={x['accuracy_mean']:.3f} "
                  f"[{x['accuracy_ci_lo']:.3f},{x['accuracy_ci_hi']:.3f}] "
                  f"f1={x['f1_macro_mean']:.3f}")
        rows.extend(r)
    except SystemExit as e:
        print(f"SKIP UCI: {e}")

    # ---------- OULAD ----------
    try:
        oulad = map_oulad(a.oulad_dir)
        # OULAD final_result has 3 classes: Pass, Fail, Withdrawn
        oulad["final_result"] = oulad["final_result"].replace({
            "Withdrawn": "Dropout",
            "Pass": "Pass", "Fail": "Fail",
        })
        # Add pseudo-"Distinction" via quartile of final_grade if available
        # (OULAD has no numeric final grade, so we use 3-class -> we drop
        #  "Distinction" class from synthetic when evaluating)
        print(f"\n=== OULAD ({len(oulad)} rows) ===")
        matched_features = ["lms_clicks_log", "mean_delay_days", "completion_rate"]
        # Rename synthetic columns to OULAD equivalents before comparison
        syn_oulad = pd.DataFrame({
            "lms_clicks_log": np.log1p(syn["lms_usage_hours"].values),
            "mean_delay_days": syn["submission_delay"].values,
            "completion_rate": syn["assignment_completion_rate"].values,
            "final_result": syn["final_result"].replace({
                "Distinction": "Pass"  # collapse to 3-class OULAD
            }).values,
        })
        oulad_3 = oulad.dropna(subset=matched_features + ["final_result"])

        # Regime A only (matched features)
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, f1_score
        X_syn = syn_oulad[matched_features].values
        y_syn = syn_oulad["final_result"].values
        X_real = oulad_3[matched_features].values
        y_real = oulad_3["final_result"].values

        model = LGBMClassifier(n_estimators=300, num_leaves=63,
                               random_state=42, verbose=-1, class_weight="balanced")
        model.fit(X_syn, y_syn)
        pred = model.predict(X_real)
        m = bootstrap_metrics(y_real, pred)
        m.update({"dataset": "OULAD", "regime": "matched_only",
                  "n_features": len(matched_features)})
        rows.append(m)
        print(f"  matched_only           acc={m['accuracy_mean']:.3f} "
              f"[{m['accuracy_ci_lo']:.3f},{m['accuracy_ci_hi']:.3f}] "
              f"f1={m['f1_macro_mean']:.3f}")

        # Real-only baseline
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_real, y_real, test_size=0.3, stratify=y_real, random_state=42)
        model_r = LGBMClassifier(n_estimators=300, num_leaves=63,
                                 random_state=42, verbose=-1, class_weight="balanced")
        model_r.fit(X_tr, y_tr)
        pred_r = model_r.predict(X_te)
        m = bootstrap_metrics(y_te, pred_r)
        m.update({"dataset": "OULAD", "regime": "real_only",
                  "n_features": len(matched_features)})
        rows.append(m)
        print(f"  real_only              acc={m['accuracy_mean']:.3f} "
              f"[{m['accuracy_ci_lo']:.3f},{m['accuracy_ci_hi']:.3f}] "
              f"f1={m['f1_macro_mean']:.3f}")
    except SystemExit as e:
        print(f"SKIP OULAD: {e}")
    except Exception as e:
        print(f"OULAD failed: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()