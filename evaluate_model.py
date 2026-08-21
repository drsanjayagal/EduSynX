#!/usr/bin/env python3
"""Evaluate trained models and generate metrics and plots."""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config
from preprocessing import load_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained EduSynX models.")
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR))
    parser.add_argument("--models", nargs="*", default=None, help="Model names (e.g., RandomForest XGBoost LightGBM)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    test_path = results_dir / "test_raw.csv"
    preprocessor_path = results_dir / "preprocessor.joblib"
    label_encoder_path = results_dir / "label_encoder.joblib"
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for path in [test_path, preprocessor_path, label_encoder_path]:
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}. Run train_model.py first.")

    df = pd.read_csv(test_path)
    X_raw = df[config.FEATURE_COLS]
    y_raw = df[config.TARGET]

    preprocessor = load_object(preprocessor_path)
    label_encoder = load_object(label_encoder_path)

    y_true = label_encoder.transform(y_raw)
    X_trans = preprocessor.transform(X_raw)

    model_files = sorted(models_dir.glob("model_*.joblib"))
    if args.models:
        model_files = [f for f in model_files if any(name in f.stem for name in args.models)]

    if not model_files:
        raise SystemExit("No trained models found.")

    n_classes = len(label_encoder.classes_)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    metrics_rows = []

    for model_file in model_files:
        model_name = model_file.stem.replace("model_", "")
        print(f"Evaluating {model_name}...")
        model = load_object(model_file)

        y_pred = model.predict(X_trans)
        y_proba = model.predict_proba(X_trans) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        roc_auc = np.nan
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            except ValueError:
                roc_auc = np.nan

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_weighted": prec_weighted,
                "recall_weighted": rec_weighted,
                "f1_weighted": f1_weighted,
                "roc_auc_macro": roc_auc,
            }
        )

        # Confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{model_name} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(n_classes)
        plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, label_encoder.classes_)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{model_name}.png", dpi=150)
        plt.close()

        # ROC curves
        if y_proba is not None:
            plt.figure(figsize=(6, 5))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc_class = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    label=f"{label_encoder.classes_[i]} (AUC={roc_auc_class:.2f})",
                )
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC Curves")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(plots_dir / f"roc_curve_{model_name}.png", dpi=150)
            plt.close()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    print("Metrics saved to", results_dir / "metrics.csv")

    # Model comparison bar chart
    if not metrics_df.empty:
        metrics_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(
            x="model", kind="bar", figsize=(10, 6)
        )
        plt.title("Model Comparison (Macro Metrics)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_comparison.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()