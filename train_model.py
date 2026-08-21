#!/usr/bin/env python3
"""Train RandomForest, XGBoost, and LightGBM models on sampled data."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config
from preprocessing import create_preprocessor, encode_target, load_sample, save_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models on synthetic student data.")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR))
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--output-dir", type=str, default=str(config.RESULTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading sample data...")
    df = load_sample(Path(args.data_dir), max_rows=args.max_train_rows, seed=args.seed)
    if df.empty:
        raise SystemExit("No data found. Run generate_data.py first.")

    X = df[config.FEATURE_COLS]
    y_raw = df[config.TARGET]

    # Encode target labels
    y, label_encoder = encode_target(y_raw)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    print("Fitting preprocessor...")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Save preprocessor and label encoder for later use
    save_object(preprocessor, output_dir / "preprocessor.joblib")
    save_object(label_encoder, output_dir / "label_encoder.joblib")

    # Save raw test set (unprocessed) for evaluation/explainability
    test_raw = X_test.copy()
    test_raw[config.TARGET] = label_encoder.inverse_transform(y_test)
    test_raw.to_csv(output_dir / "test_raw.csv", index=False)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            eval_metric="mlogloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.1,
            random_state=args.seed,
            class_weight="balanced",
            verbose=-1,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_t, y_train)
        save_object(model, models_dir / f"model_{name}.joblib")
        print(f"Saved model_{name}.joblib")

    print("Training complete. Models and preprocessor saved to", output_dir)


if __name__ == "__main__":
    main()