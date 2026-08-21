#!/usr/bin/env python3
"""Generate actionable decision support insights from a trained model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import config
from preprocessing import load_object

# Features for which an increase typically improves success probability
ACTIONABLE_HIGHER = {
    "study_hours",
    "attendance",
    "assignments_score",
    "LMS_usage_time",
    "login_frequency",
    "forum_participation",
    "online_learning_hours",
    "library_usage_hours",
    "sleep_hours",
    "physical_activity_hours",
    "motivation_score",
    "self_efficacy_score",
    "family_support_score",
    "peer_support_score",
    "tutoring_sessions",
}

# Features for which a decrease typically improves success probability
ACTIONABLE_LOWER = {
    "submission_delay",
    "stress_level",
    "anxiety_score",
    "part_time_job_hours",
    "backlog_count",
    "campus_distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate decision support insights.")
    parser.add_argument("--results-dir", type=str, default=str(config.RESULTS_DIR))
    parser.add_argument("--model-name", type=str, default="XGBoost")
    parser.add_argument("--num-students", type=int, default=5)
    parser.add_argument("--output-file", type=str, default=str(config.RESULTS_DIR / "insights.txt"))
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
    sample_df = df.sample(n=min(args.num_students, len(df)), random_state=42)

    X_raw = sample_df[config.FEATURE_COLS]
    y_raw = sample_df[config.TARGET]

    preprocessor = load_object(preprocessor_path)
    label_encoder = load_object(label_encoder_path)
    model = load_object(model_path)

    X_trans = preprocessor.transform(X_raw)
    feature_names_out = preprocessor.get_feature_names_out()
    feature_names = [str(f) for f in feature_names_out]

    # Map transformed feature names back to original feature names
    original_names = []
    for name in feature_names:
        if "__" in name:
            original_names.append(name.split("__")[-1])
        else:
            original_names.append(name)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    class_names = list(label_encoder.classes_)
    distinction_idx = class_names.index("Distinction") if "Distinction" in class_names else 0

    # Extract SHAP values for the target class "Distinction"
    if isinstance(shap_values, list):
        shap_values_class = shap_values[distinction_idx]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            shap_values_class = arr[:, :, distinction_idx]
        else:
            shap_values_class = arr

    # Sanity check
    if shap_values_class.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name / SHAP shape mismatch: {len(feature_names)} names vs "
            f"{shap_values_class.shape[1]} SHAP columns"
        )

    insights = []

    for i, (_, row) in enumerate(sample_df.iterrows()):
        proba = model.predict_proba(X_trans[i : i + 1])[0]
        distinction_prob = proba[distinction_idx]

        insights.append(
            f"Student {i + 1} (actual: {y_raw.iloc[i]}) - Predicted Distinction probability: "
            f"{distinction_prob:.3f}"
        )

        # Top negative SHAP features (hindering Distinction)
        shap_row = shap_values_class[i]
        neg_indices = np.argsort(shap_row)[:5]

        suggestions = []
        for idx in neg_indices:
            orig = original_names[idx]
            current_value = row[orig] if orig in X_raw.columns else "N/A"
            if orig in ACTIONABLE_HIGHER:
                suggestions.append(f"  - Increase {orig} (current value: {current_value})")
            elif orig in ACTIONABLE_LOWER:
                suggestions.append(f"  - Reduce {orig} (current value: {current_value})")
            else:
                suggestions.append(f"  - Address {orig} (current value: {current_value})")

        if suggestions:
            insights.append("  Actionable suggestions:")
            insights.extend(suggestions)
        else:
            insights.append("  No actionable suggestions based on top negative SHAP features.")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(insights), encoding="utf-8")

    print("\n".join(insights))
    print(f"\nInsights saved to {output_file}")


if __name__ == "__main__":
    main()