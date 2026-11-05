#!/usr/bin/env python3
"""
Train RF/XGB/LGBM/Stacking with hyperparameter tuning.

v2 fixes:
  - avoids n_jobs oversubscription (n_jobs=1 on estimators,
    n_jobs=-1 on GridSearchCV only)
  - verbose=2 prints each CV fit as it completes
  - --quick flag for a 5-minute smoke test
  - XGBoost tree_method='hist', the fast path
"""
import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import config
from preprocessing import create_preprocessor, encode_target, load_sample, save_object


FULL_GRIDS = {
    "RandomForest": {"n_estimators": [200, 300],
                     "max_depth": [20, 30],
                     "min_samples_split": [5]},
    "XGBoost":      {"n_estimators": [200, 300],
                     "max_depth": [6, 9],
                     "learning_rate": [0.05, 0.1]},
    "LightGBM":     {"n_estimators": [200, 300],
                     "num_leaves": [63, 127],
                     "learning_rate": [0.05, 0.1]},
}

QUICK_GRIDS = {
    "RandomForest": {"n_estimators": [100], "max_depth": [20], "min_samples_split": [5]},
    "XGBoost":      {"n_estimators": [100], "max_depth": [6],  "learning_rate": [0.1]},
    "LightGBM":     {"n_estimators": [100], "num_leaves": [63],"learning_rate": [0.1]},
}


def build_model(name, seed):
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
    if name == "XGBoost":
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="mlogloss",
                             tree_method="hist")
    if name == "LightGBM":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1,
                              class_weight="balanced")
    raise ValueError(name)


def tuned_search(name, grid, X, y, seed):
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"\n[{name}] grid search: {n_combos} configs x 5 folds = {n_combos*5} fits",
          flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(build_model(name, seed), grid, cv=cv,
                      scoring="f1_macro", n_jobs=-1, verbose=2, refit=True)
    gs.fit(X, y)
    print(f"[{name}] best cv f1_macro = {gs.best_score_:.4f}  params = {gs.best_params_}",
          flush=True)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--max-train-rows", type=int, default=200_000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    p.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 50k rows, 1 config per model, no stacking")
    return p.parse_args()


def main():
    a = parse_args()
    if a.quick:
        a.max_train_rows = min(a.max_train_rows, 50_000)
        grids = QUICK_GRIDS
    else:
        grids = FULL_GRIDS

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    print(f"Loading sample: max_rows={a.max_train_rows}, quick={a.quick}", flush=True)
    df = load_sample(Path(a.data_dir), max_rows=a.max_train_rows, seed=a.seed)
    if df.empty:
        raise SystemExit("No data. Run generate_data.py first.")
    print(f"  loaded {len(df)} rows, {len(df.columns)} columns", flush=True)

    X, y_raw = df[config.FEATURE_COLS], df[config.TARGET]
    y, le = encode_target(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, stratify=y, random_state=a.seed)
    print(f"  train={len(X_train)}  test={len(X_test)}", flush=True)

    pre = create_preprocessor().fit(X_train)
    Xt_tr = pre.transform(X_train)
    Xt_te = pre.transform(X_test)
    print(f"  preprocessed shape: {Xt_tr.shape}", flush=True)

    save_object(pre, out / "preprocessor.joblib")
    save_object(le, out / "label_encoder.joblib")
    test_raw = X_test.copy()
    test_raw[config.TARGET] = le.inverse_transform(y_test)
    test_raw.to_csv(out / "test_raw.csv", index=False)

    tuning_log = {}
    models = {}
    for name, grid in grids.items():
        est, params, score = tuned_search(name, grid, Xt_tr, y_train, a.seed)
        tuning_log[name] = {"best_params": params, "cv_macro_f1": score}
        save_object(est, out / "models" / f"model_{name}.joblib")
        models[name] = est

    if not a.quick:
        print("\n[Stacking] training ...", flush=True)
        stack = StackingClassifier(
            estimators=[(n, models[n]) for n in ["RandomForest", "XGBoost", "LightGBM"]],
            final_estimator=LogisticRegression(max_iter=1000),
            n_jobs=1, passthrough=False,
        )
        stack.fit(Xt_tr, y_train)
        save_object(stack, out / "models" / "model_Stacking.joblib")
        print("[Stacking] done")

    (out / "tuning_log.json").write_text(json.dumps(tuning_log, indent=2))
    print(f"\nSaved models to {out / 'models'}")


if __name__ == "__main__":
    main()