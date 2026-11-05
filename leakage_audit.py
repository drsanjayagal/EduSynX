#!/usr/bin/env python3
"""Leakage diagnostic: can predictors reconstruct the target?
If linear R^2 or linear-classifier accuracy is very high, non-circularity
fails (Reviewer 2)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import config


def audit(df, seed=42):
    X = df[config.FEATURE_COLS].copy()
    cats = X.select_dtypes(include=["object"]).columns.tolist()
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cats] = enc.fit_transform(X[cats])
    X = X.values.astype(float)

    y_map = {"Dropout": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
    y_num = df[config.TARGET].map(y_map).values
    y = df[config.TARGET].values

    Xtr, Xte, ytr_n, yte_n, ytr, yte = train_test_split(
        X, y_num, y, test_size=0.2, random_state=seed, stratify=y)

    lr = LinearRegression().fit(Xtr, ytr_n)
    r2 = float(r2_score(yte_n, lr.predict(Xte)))

    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))

    # Correlation of each predictor with target (continuous proxy)
    corr_with_target = {}
    for i, col in enumerate(config.FEATURE_COLS):
        c = np.corrcoef(X[:, i], y_num)[0, 1]
        corr_with_target[col] = float(c) if not np.isnan(c) else 0.0

    return {
        "linear_r2_target_proxy": r2,
        "linear_classifier_accuracy": acc,
        "max_abs_feature_target_corr": float(max(abs(v) for v in corr_with_target.values())),
        "top5_correlated_features": sorted(corr_with_target.items(),
                                           key=lambda kv: abs(kv[1]), reverse=True)[:5],
        "interpretation": (
            "If R^2 > 0.85 or linear accuracy > 0.85, target is trivially recoverable. "
            "For a genuine prediction task with a four-class target and nonlinear "
            "generating mechanism, expect R^2 < 0.60 and accuracy < 0.60."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="validation/leakage_audit.csv")
    p.add_argument("--n", type=int, default=200_000)
    args = p.parse_args()

    df = pd.read_csv(args.data, nrows=args.n)
    print(f"Loaded {len(df)} rows from {args.data}")
    result = audit(df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "metric": k, "value": v if not isinstance(v, list) else str(v),
    } for k, v in result.items()]).to_csv(args.out, index=False)

    print("\n=== Leakage Audit ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()