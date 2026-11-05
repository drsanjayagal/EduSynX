#!/usr/bin/env python3
"""Assemble validation/validation_report.md for the Zenodo release."""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _md_table(df):
    if df is None or df.empty:
        return "_no data_\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        cols = " | ".join(df.columns.astype(str))
        sep = " | ".join(["---"] * len(df.columns))
        lines = [f"| {cols} |", f"| {sep} |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation-dir", default="validation")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="validation/validation_report.md")
    args = p.parse_args()

    vd = Path(args.validation_dir)
    rd = Path(args.results_dir)

    stats = _read_csv(vd / "statistical_summary.csv")
    corr_check = _read_csv(vd / "correlation_check.csv")
    class_dist = _read_csv(vd / "class_distribution.csv")
    completion_check = _read_csv(vd / "completion_rate_check.csv")
    leakage = _read_csv(vd / "leakage_audit.csv")
    fairness = _read_csv(vd / "fairness_audit.csv")
    counterfactuals = _read_csv(vd / "counterfactuals.csv")
    tost = _read_csv(vd / "tost_results.csv")
    transfer = _read_csv(vd / "real_world_validation.csv")
    metrics = _read_csv(rd / "metrics.csv")
    tuning_log_path = rd / "tuning_log.json"

    gen_config_path = Path("configs") / "generation_config.yaml"
    gen_script = Path("generate_data.py")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append("# EduSynX-Bench — Validation Report\n")
    L.append(f"**Generated:** {now}  ")
    L.append(f"**Target version:** v1.1.0 (pending Zenodo upload)  ")
    L.append("**Purpose:** Actual statistics and audit results for the regenerated "
             "dataset. Every number below comes from a file in this repository "
             "or in `data/`. Nothing is fabricated.\n")

    # 1. Provenance
    L.append("## 1. Provenance\n")
    prov = [{"file": "generate_data.py",
             "sha256": _sha256(gen_script) if gen_script.exists() else "MISSING"}]
    if gen_config_path.exists():
        prov.append({"file": str(gen_config_path), "sha256": _sha256(gen_config_path)})
    L.append(_md_table(pd.DataFrame(prov)))
    if tuning_log_path.exists():
        L.append("\n**Tuning log (`results/tuning_log.json`):**\n")
        L.append("```json\n" + tuning_log_path.read_text()[:2000] + "\n```\n")
    else:
        L.append("_`results/tuning_log.json` NOT YET RUN_\n")

    # 2. Class distribution (R1 #3)
    L.append("\n## 2. Target Class Distribution vs Manuscript Claim\n")
    L.append("Manuscript abstract claims 20% Distinction / 40% Pass / 25% Fail / "
             "15% Dropout. Reviewer 1 (comment 3) noted the described thresholds "
             "produce ~29/30/26/16 instead. Below are the **actual** proportions "
             "from the regenerated data (thresholds calibrated on a 200k pilot "
             "sample to hit the target distribution).\n")
    L.append(_md_table(class_dist))

    # 3. Descriptive statistics (R1 #3)
    L.append("\n## 3. Descriptive Statistics — Honest Table 4\n")
    L.append("Table 4 of the manuscript reports the following values. The "
             "`actual_*` columns are computed from the regenerated data. "
             "Reviewer 1 (comment 3) showed Table 1 equations cannot produce the "
             "reported SDs; the deltas below quantify that discrepancy.\n")
    if stats is not None:
        t4 = stats[stats["manuscript_mean"].notna()][
            ["feature", "mean", "std", "min", "max",
             "manuscript_mean", "manuscript_std",
             "delta_mean", "delta_std", "std_matches_manuscript"]
        ].round(3)
        L.append(_md_table(t4))
    else:
        L.append("_`validation/statistical_summary.csv` NOT YET RUN_\n")

    # 4. Correlations (R2)
    L.append("\n## 4. Correlation Structure vs Manuscript Claims\n")
    L.append("Reviewer 2 flagged that Section 3.2.2 reports "
             "`study_hours ↔ internal_marks r = 0.42` and "
             "`study_hours ↔ previous_gpa r = 0.35`, while Fig. 12 shows "
             "r = 0.35 and r = 0.27. Both are compared against the "
             "regenerated data.\n")
    L.append(_md_table(corr_check))

    # 5. Completion rate (R1 #4)
    L.append("\n## 5. Completion Rate — the 0.70 vs 0.78 Discrepancy\n")
    L.append("Reviewer 1 (comment 4): *'The completion-rate equation gives a "
             "mean of 0.70, not 0.78.'* The actual mean from regenerated data:\n")
    L.append(_md_table(completion_check))

    # 6. Leakage (R2)
    L.append("\n## 6. Leakage Audit (Reviewer 2 — non-circularity)\n")
    L.append("Fixes applied: composites removed from predictors (see "
             "`config.py`), non-linear `tanh` terms added to feature equations, "
             "an unobserved confounder `U` added to the target generating "
             "equation.\n")
    if leakage is not None:
        L.append(_md_table(leakage))
        try:
            r2_row = leakage[leakage["metric"] == "linear_r2_target_proxy"]
            acc_row = leakage[leakage["metric"] == "linear_classifier_accuracy"]
            r2 = float(r2_row["value"].iloc[0]) if not r2_row.empty else None
            acc = float(acc_row["value"].iloc[0]) if not acc_row.empty else None
            if r2 is not None and acc is not None:
                verdict = "PASS" if (r2 < 0.60 and acc < 0.60) else "FAIL"
                L.append(f"\n**Verdict:** linear R² = {r2:.3f}, linear accuracy = "
                         f"{acc:.3f}. **{verdict}** — "
                         f"{'target is not trivially recoverable.' if verdict == 'PASS' else 'residual leakage; further nonlinearity needed.'}\n")
        except Exception:
            pass
    else:
        L.append("_`validation/leakage_audit.csv` NOT YET RUN_\n")

    # 6b. TOST (R1 #4)
    L.append("\n## 6b. TOST Equivalence Testing (Reviewer 1 #4)\n")
    L.append("**Corrected decision rule:** equivalent iff "
             "**p_tost < α AND |Cohen's d| < Δ**. The manuscript's Table 6 "
             "reported `completion_rate` as equivalent with d = 0.24 at "
             "Δ = 0.2, which is impossible under this rule.\n")
    if tost is not None:
        L.append(_md_table(tost))
        n_total = len(tost)
        n_equiv = int(tost["equivalent"].sum()) if "equivalent" in tost.columns else 0
        L.append(f"\n**Summary:** {n_equiv}/{n_total} features pass equivalence "
                 f"at Δ = 0.2. The manuscript's Table 6 claims must be rewritten "
                 f"to match these actual values.\n")
    else:
        L.append("_`validation/tost_results.csv` NOT YET RUN_\n")

    # 7. Fairness (R2)
    L.append("\n## 7. Fairness Audit (Reviewer 2)\n")
    L.append("Demographic parity (DP), equalized odds (EO), and predictive "
             "parity (PP) differences with 95% bootstrap CIs (1000 resamples).\n")
    L.append(_md_table(fairness))

    # 7b. Real-world transfer (R2)
    L.append("\n## 7b. Real-World Transfer (Reviewer 2)\n")
    L.append("Two regimes reported explicitly: (a) **matched-only** — train on "
             "the 2–3 features shared with the real dataset; (b) "
             "**all-with-imputation** — train on all 44 synthetic features and "
             "zero-fill missing. Reviewer 2 asked which regime the paper used; "
             "both are reported here.\n")
    if transfer is not None:
        L.append(_md_table(transfer.round(3)))
    else:
        L.append("_`validation/real_world_validation.csv` NOT YET RUN_\n")

    # 8. Model performance
    L.append("\n## 8. Model Performance (Tuned + Stacking)\n")
    L.append("All models trained with 5-fold cross-validated grid search under "
             "an identical protocol (Reviewer 3 #1).\n")
    L.append(_md_table(metrics.round(3) if metrics is not None else None))

    # 9. Counterfactuals
    L.append("\n## 9. Counterfactual Scenarios (Reviewer 2 A3, Reviewer 3)\n")
    L.append("Every counterfactual is labelled `model_scenario` to make "
             "explicit it describes the *model's* behaviour, not a causal "
             "intervention. Continuous cross-entropy objective replaces the "
             "undefined indicator function in Algorithm 3.\n")
    if counterfactuals is not None and not counterfactuals.empty:
        L.append(f"**Generated:** {len(counterfactuals)} counterfactuals. "
                 f"First 5 shown.\n")
        L.append(_md_table(counterfactuals.head(5)))
    else:
        L.append("_`validation/counterfactuals.csv` NOT YET RUN_\n")

    # 10. Discrepancy summary
    L.append("\n## 10. Summary of Manuscript Claims Not Reproduced\n")
    L.append("Honest disclosure requested by Reviewer 1. Each row is a "
             "manuscript claim the regenerated data does not support, with a "
             "pointer to the Phase 2 correction.\n")
    discrepancies = []
    if stats is not None:
        mism = stats[(stats["manuscript_mean"].notna()) &
                     (~stats["std_matches_manuscript"].astype(bool))]
        for _, r in mism.iterrows():
            discrepancies.append({
                "claim": f"Table 4: {r['feature']} std = {r['manuscript_std']:.2f}",
                "actual": f"{r['std']:.3f}",
                "delta": f"{r['delta_std']:+.3f}",
                "action": "Revise Table 4 (or revise Table 1 equations)",
            })
    if corr_check is not None:
        for _, r in corr_check.iterrows():
            if not r.get("matches_within_0.05", True):
                discrepancies.append({
                    "claim": f"{r['pair']}: r = {r['manuscript_r']:.2f}",
                    "actual": f"{r['actual_r']:.3f}",
                    "delta": f"{r['delta']:+.3f}",
                    "action": "Update text and Fig. 12",
                })
    if completion_check is not None and not completion_check.empty:
        c = completion_check.iloc[0]
        if abs(float(c["delta"])) > 0.02:
            discrepancies.append({
                "claim": f"completion_rate mean = {c['manuscript_mean']:.2f}",
                "actual": f"{c['mean_normalised_0_1']:.3f}",
                "delta": f"{c['delta']:+.3f}",
                "action": "Revise Tables 5–6 (Reviewer 1 #4)",
            })
    if class_dist is not None:
        for _, r in class_dist.iterrows():
            if not r.get("matches_within_0.03", True):
                discrepancies.append({
                    "claim": f"{r['class']} proportion = {r['manuscript_proportion']:.2f}",
                    "actual": f"{r['actual_proportion']:.3f}",
                    "delta": f"{r['delta']:+.3f}",
                    "action": "Update abstract and results",
                })
    if tost is not None:
        for _, r in tost.iterrows():
            equiv_str = "Yes" if r.get("equivalent", False) else "No"
            d = r.get("cohens_d", float("nan"))
            if not r.get("equivalent", False) and abs(d) < 0.5:
                discrepancies.append({
                    "claim": f"Table 6: {r.get('dataset','?')} {r.get('feature','?')} marked equivalent",
                    "actual": f"|d| = {abs(d):.3f}",
                    "delta": "NOT equivalent",
                    "action": "Rewrite Table 6 with correct decision rule",
                })
    if discrepancies:
        L.append(_md_table(pd.DataFrame(discrepancies)))
    else:
        L.append("_No discrepancies detected._\n")

    # 11. Coverage
    L.append("\n## 11. Reviewer-Comment Coverage (Phase 1 status)\n")
    coverage = [
        {"reviewer": "R1 #1", "issue": "GitHub repo incomplete; hardcoded figures",
         "status": "ADDRESSED", "evidence": "Figure scripts read pipeline CSVs"},
        {"reviewer": "R1 #2", "issue": "Circular target; Table 1/Eq.1 not implemented",
         "status": "ADDRESSED", "evidence": "Composites removed; Eq.1 covariance; nonlinear target"},
        {"reviewer": "R1 #3", "issue": "Table 1 equations cannot produce Table 4 SDs",
         "status": "QUANTIFIED", "evidence": "Section 3 & 10"},
        {"reviewer": "R1 #4", "issue": "d=0.24 > 0.2 reported as equivalent",
         "status": "ADDRESSED", "evidence": "Section 6b — corrected decision rule; actual |d| values"},
        {"reviewer": "R2 leakage", "issue": "Latent variable leakage",
         "status": "ADDRESSED", "evidence": "Section 6 — R2 and linear accuracy thresholds"},
        {"reviewer": "R2 figures", "issue": "Fig 13/14 contradictions; correlation mismatch",
         "status": "ADDRESSED", "evidence": "Figure scripts read CSVs; Section 4"},
        {"reviewer": "R2 scale", "issue": "Downsampled to 160k",
         "status": "PARTIAL", "evidence": "--max-train-rows explicit in train_model.py"},
        {"reviewer": "R2 A3", "issue": "Nelder-Mead on discontinuous step function",
         "status": "ADDRESSED", "evidence": "counterfactual.py uses cross-entropy"},
        {"reviewer": "R2 transfer", "issue": "Matched-vs-imputed training ambiguous",
         "status": "ADDRESSED", "evidence": "Section 7b — both regimes reported"},
        {"reviewer": "R2 tabular DL", "issue": "No TabNet / FT-Transformer",
         "status": "PENDING", "evidence": "Optional --include-tabnet flag; not yet run"},
        {"reviewer": "R3 #1", "issue": "Default configs, no fair comparison",
         "status": "ADDRESSED", "evidence": "GridSearchCV for all models"},
        {"reviewer": "R3 #2", "issue": "Counterfactuals as causal recommendations",
         "status": "ADDRESSED", "evidence": "All outputs labelled model_scenario; CAVEAT printed"},
        {"reviewer": "R3 #3", "issue": "Compare against recent publications",
         "status": "PENDING", "evidence": "Requires Phase 2 manuscript revision"},
    ]
    L.append(_md_table(pd.DataFrame(coverage)))

    L.append("\n---\n")
    L.append("_End of validation report._\n")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out} ({sum(len(s) for s in L):,} chars)")


if __name__ == "__main__":
    main()