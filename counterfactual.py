#!/usr/bin/env python3
"""Fast counterfactual generation (Reviewer 2 A3, Reviewer 3 causal framing).

Changes vs previous version:
  - default model is LightGBM (Stacking is ~100x slower per predict_proba)
  - n_restarts=3, maxiter=200 (was 10 x 1000)
  - per-student progress printing
  - wall-clock timeout per student
  - early-exit once target class reached
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from preprocessing import load_object


MODIFIABLE = ["study_hours", "attendance", "lms_usage_hours",
              "submission_delay", "sleep_hours", "tutoring_sessions",
              "forum_participation", "assignment_completion_rate"]

CAVEAT = ("Model scenario only. Describes what the model would predict if "
          "features changed. Not a causal recommendation.")


def _obj(delta, x_base, mod_idx, model, target_class, lam):
    x_cf = x_base.copy()
    x_cf[mod_idx] = x_cf[mod_idx] + delta
    proba = model.predict_proba(x_cf.reshape(1, -1))[0]
    return lam * np.sum(delta ** 2) - np.log(proba[target_class] + 1e-10)


def generate_one(x_trans, mod_idx, model, target_class, lo, hi,
                 lam=0.1, n_restarts=3, maxiter=200, seed=42,
                 student_timeout_s=15):
    rng = np.random.default_rng(seed)
    deadline = time.time() + student_timeout_s
    best = None

    for r in range(n_restarts):
        if time.time() > deadline:
            break
        d0 = rng.uniform(lo - x_trans[mod_idx], hi - x_trans[mod_idx])
        try:
            res = minimize(
                _obj, d0,
                args=(x_trans, mod_idx, model, target_class, lam),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
            )
        except Exception:
            continue

        x_cf = x_trans.copy()
        x_cf[mod_idx] += res.x
        pred = model.predict(x_cf.reshape(1, -1))[0]
        if pred == target_class:
            mag = float(np.sum(res.x ** 2))
            if best is None or mag < best["magnitude"]:
                best = {"delta": res.x.copy(), "magnitude": mag}
                break  # early exit — a valid counterfactual found

    if best is None:
        return None
    return {"magnitude": best["magnitude"], "delta": best["delta"]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--model-name", default="LightGBM",
                   help="RandomForest / XGBoost / LightGBM / Stacking")
    p.add_argument("--n-students", type=int, default=10)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--timeout-s", type=int, default=15,
                   help="per-student wall-clock cap")
    p.add_argument("--out", default="validation/counterfactuals.csv")
    return p.parse_args()


def main():
    a = parse_args()
    rd = Path(a.results_dir)
    model_path = rd / "models" / f"model_{a.model_name}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Train this model first.")

    print(f"Loading {a.model_name} ...")
    model = load_object(model_path)
    pre = load_object(rd / "preprocessor.joblib")
    le = load_object(rd / "label_encoder.joblib")
    df = pd.read_csv(rd / "test_raw.csv").sample(n=a.n_students, random_state=42)

    X = pre.transform(df[config.FEATURE_COLS])
    feat_names = [str(n).split("__")[-1] for n in pre.get_feature_names_out()]
    mod_idx = np.array([i for i, n in enumerate(feat_names) if n in MODIFIABLE])
    lo = np.full(len(mod_idx), -3.0)
    hi = np.full(len(mod_idx), 3.0)

    print(f"Modifiable features: {[feat_names[i] for i in mod_idx]}")
    print(f"n_students={a.n_students}, n_restarts={a.n_restarts}, "
          f"maxiter={a.maxiter}, timeout={a.timeout_s}s/student")
    print("-" * 60)

    rows = []
    t_start = time.time()
    for i, pred in enumerate(model.predict(X)):
        t_stu = time.time()
        pred_cls = le.inverse_transform([pred])[0]

        # pick a target class different from the prediction
        candidates = [c for c in le.classes_ if c != pred_cls]
        cf = None
        chosen = None
        for tc_str in candidates:
            tc = le.transform([tc_str])[0]
            cf = generate_one(X[i], mod_idx, model, tc, lo, hi,
                              n_restarts=a.n_restarts, maxiter=a.maxiter,
                              student_timeout_s=a.timeout_s)
            if cf is not None:
                chosen = tc_str
                break

        if cf is not None:
            changes = {feat_names[j]: float(cf["delta"][k])
                       for k, j in enumerate(mod_idx) if abs(cf["delta"][k]) > 1e-3}
            rows.append({
                "student_idx": i,
                "original_class": pred_cls,
                "target_class": chosen,
                "magnitude": cf["magnitude"],
                "changes": str(changes),
                "label": "model_scenario",
            })
            status = f"OK -> {chosen} (mag={cf['magnitude']:.2f})"
        else:
            status = "no counterfactual found (timeout or infeasible)"

        print(f"[{i+1:>2}/{a.n_students}] pred={pred_cls:<12} "
              f"{status:<45} ({time.time()-t_stu:.1f}s)")

    print("-" * 60)
    print(f"Total: {time.time()-t_start:.1f}s   "
          f"counterfactuals generated: {len(rows)}/{a.n_students}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nCaveat: {CAVEAT}")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()