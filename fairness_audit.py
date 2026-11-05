#!/usr/bin/env python3
"""Fairness audit with bootstrap CIs (Reviewer 2)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from preprocessing import load_object


def dp(y_pred, ma, mb):
    return abs(y_pred[ma].mean() - y_pred[mb].mean())


def eo(y_true, y_pred, ma, mb):
    def tpr(m):
        pos = m & (y_true == 1)
        return y_pred[pos].mean() if pos.any() else np.nan
    def fpr(m):
        neg = m & (y_true == 0)
        return y_pred[neg].mean() if neg.any() else np.nan
    return max(abs(tpr(ma) - tpr(mb)), abs(fpr(ma) - fpr(mb)))


def pp(y_true, y_pred, ma, mb):
    def ppv(m):
        pos = m & (y_pred == 1)
        return y_true[pos].mean() if pos.any() else np.nan
    return abs(ppv(ma) - ppv(mb))


def boot(fn, *arrays, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arrays[0]))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        v = fn(*[a[s] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--model-name", default="Stacking")
    p.add_argument("--out", default="validation/fairness_audit.csv")
    return p.parse_args()


def main():
    a = parse_args()
    rd = Path(a.results_dir)
    model_path = rd / "models" / f"model_{a.model_name}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}")

    print(f"Loading {model_path} ...")
    model = load_object(model_path)
    pre = load_object(rd / "preprocessor.joblib")
    le = load_object(rd / "label_encoder.joblib")
    df = pd.read_csv(rd / "test_raw.csv")
    print(f"  test set: {len(df)} rows, columns: {len(df.columns)}")

    X = pre.transform(df[config.FEATURE_COLS])
    y = le.transform(df[config.TARGET])
    y_pred = model.predict(X)
    print(f"  predicting ...")

    positive_classes = [le.transform([c])[0] for c in ["Pass", "Distinction"]
                        if c in le.classes_]
    y_pos = np.isin(y, positive_classes).astype(int)
    y_pred_pos = np.isin(y_pred, positive_classes).astype(int)

    rows = []
    for attr in ["socioeconomic_status", "gender", "parent_education"]:
        if attr not in df.columns:
            print(f"  skip {attr}: not in test_raw.csv")
            continue
        groups = df[attr].value_counts()
        # drop groups with fewer than 5 rows
        valid = groups[groups >= 5].index.tolist()
        if len(valid) < 2:
            print(f"  skip {attr}: fewer than 2 usable groups")
            continue
        ref = valid[0]
        print(f"  auditing {attr}: ref={ref}, comparing to {valid[1:]}")
        for other in valid[1:]:
            ma = (df[attr] == ref).values
            mb = (df[attr] == other).values
            dpm, dplo, dphi = boot(dp, y_pred_pos, ma, mb)
            eom, eolo, eohi = boot(eo, y_pos, y_pred_pos, ma, mb)
            ppm, pplo, pphi = boot(pp, y_pos, y_pred_pos, ma, mb)
            rows.append({
                "attribute": attr, "reference": ref, "comparison": other,
                "dp_mean": dpm, "dp_lo": dplo, "dp_hi": dphi,
                "eo_mean": eom, "eo_lo": eolo, "eo_hi": eohi,
                "pp_mean": ppm, "pp_lo": pplo, "pp_hi": pphi,
            })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(a.out, index=False)
    print(f"\n{df_out.round(3).to_string(index=False)}")
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()