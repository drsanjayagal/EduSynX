#!/usr/bin/env python3
"""Generate SHAP-based global and local explanations for trained models."""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from preprocessing import load_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations.")
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR))
    parser.add_argument("--model-name", type=str, default="XGBoost")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--class-name", type=str, default="Distinction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    test_path = results_dir / "test_raw.csv"
    preprocessor_path = results_dir / "preprocessor.joblib"
    label_encoder_path = results_dir / "label_encoder.joblib"
    model_path = results_dir / "models" / f"model_{args.model_name}.joblib"

    for path in [test_path, preprocessor_path, label_encoder_path, model_path]:
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}. Run train_model.py first.")

    df = pd.read_csv(test_path)
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    X_raw = df[config.FEATURE_COLS]
    preprocessor = load_object(preprocessor_path)
    label_encoder = load_object(label_encoder_path)
    model = load_object(model_path)

    X_trans = preprocessor.transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())
    feature_names = [str(f) for f in feature_names]  # ensure strings

    print(f"Computing SHAP values for {args.model_name} on {len(X_trans)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Handle different SHAP return types
    class_names = list(label_encoder.classes_)
    if args.class_name in class_names:
        class_idx = class_names.index(args.class_name)
    else:
        class_idx = 0

    if isinstance(shap_values, list):
        # List of 2D arrays: one per class
        shap_values_class = shap_values[class_idx]
    else:
        # Could be 3D array (samples, features, classes) or 2D (single class)
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            shap_values_class = arr[:, :, class_idx]
        else:
            shap_values_class = arr

    # Sanity check: dimensions must match
    if shap_values_class.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name / SHAP shape mismatch: {len(feature_names)} names vs "
            f"{shap_values_class.shape[1]} SHAP columns"
        )

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values_class, X_trans, feature_names=feature_names, show=False)
    plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(
        shap_values_class, X_trans, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.savefig(plots_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save feature importance CSV
    importance = np.abs(shap_values_class).mean(axis=0).flatten()
    feature_names_flat = np.asarray(feature_names).flatten()

    importance_df = pd.DataFrame(
        {"feature": feature_names_flat, "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(results_dir / "shap_importance.csv", index=False)

    print("SHAP outputs saved to", plots_dir)


if __name__ == "__main__":
    main()