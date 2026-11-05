#!/usr/bin/env python3
"""Evaluate trained models. Saves metrics + raw matrices for figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

import config
from preprocessing import load_object


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    return p.parse_args()


def main():
    a = parse_args()
    rd = Path(a.results_dir)
    plots = rd / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    test_path = rd / "test_raw.csv"
    for f in [test_path, rd / "preprocessor.joblib", rd / "label_encoder.joblib"]:
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run train_model.py first.")

    df = pd.read_csv(test_path)
    X_raw, y_raw = df[config.FEATURE_COLS], df[config.TARGET]
    pre = load_object(rd / "preprocessor.joblib")
    le = load_object(rd / "label_encoder.joblib")
    X = pre.transform(X_raw)
    y = le.transform(y_raw)

    # Correlation matrix (needed for R2 text-figure consistency)
    numeric_df = X_raw.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    corr.to_csv(rd / "correlation_matrix.csv")

    model_files = sorted((rd / "models").glob("model_*.joblib"))
    if not model_files:
        raise SystemExit("No models found.")

    classes = list(le.classes_)
    n_classes = len(classes)
    y_bin = label_binarize(y, classes=range(n_classes))

    rows, roc_rows = [], []
    for mf in model_files:
        name = mf.stem.replace("model_", "")
        print(f"Evaluating {name}...")
        model = load_object(mf)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        acc = accuracy_score(y, y_pred)
        pm, rm, fm, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
        try:
            roc_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            roc_auc = float("nan")

        rows.append({"model": name, "accuracy": acc,
                     "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
                     "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
                     "roc_auc_macro": roc_auc})

        cm = confusion_matrix(y, y_pred)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(rd / f"cm_{name}.csv")

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            for f_, t_ in zip(fpr, tpr):
                roc_rows.append({"model": name, "class": cls, "fpr": f_, "tpr": t_})

    pd.DataFrame(rows).to_csv(rd / "metrics.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(rd / "roc_data.csv", index=False)
    print(f"\nSaved: metrics.csv, cm_*.csv, correlation_matrix.csv, roc_data.csv in {rd}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()