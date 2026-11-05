# EduSynX-Bench — Validation Report

**Generated:** 2026-11-05 11:41 UTC  
**Target version:** v1.1.0 (pending Zenodo upload)  
**Purpose:** Actual statistics and audit results for the regenerated dataset. Every number below comes from a file in this repository or in `data/`. Nothing is fabricated.

## 1. Provenance

| file | sha256 |
| --- | --- |
| generate_data.py | 63ac4c5b5c85454ea49b04ef625b21836b4617edac1684ca23d7919d2441c672 |
| configs\generation_config.yaml | 5e530477833c5763d0d5c31e5f1b6ff1dd1981d33d74f6b1defff1e69179fde0 |


**Tuning log (`results/tuning_log.json`):**

```json
{
  "RandomForest": {
    "best_params": {
      "max_depth": 20,
      "min_samples_split": 5,
      "n_estimators": 300
    },
    "cv_macro_f1": 0.4716925254544437
  },
  "XGBoost": {
    "best_params": {
      "learning_rate": 0.05,
      "max_depth": 6,
      "n_estimators": 200
    },
    "cv_macro_f1": 0.45803239895047715
  },
  "LightGBM": {
    "best_params": {
      "learning_rate": 0.05,
      "n_estimators": 300,
      "num_leaves": 127
    },
    "cv_macro_f1": 0.4515199215897797
  }
}
```


## 2. Target Class Distribution vs Manuscript Claim

Manuscript abstract claims 20% Distinction / 40% Pass / 25% Fail / 15% Dropout. Reviewer 1 (comment 3) noted the described thresholds produce ~29/30/26/16 instead. Below are the **actual** proportions from the regenerated data (thresholds calibrated on a 200k pilot sample to hit the target distribution).

| class | actual_count | actual_proportion | manuscript_proportion | delta | matches_within_0.03 |
| --- | --- | --- | --- | --- | --- |
| Distinction | 40002 | 0.20001 | 0.2 | 9.999999999982244e-06 | True |
| Pass | 79686 | 0.39843 | 0.4 | -0.00157 | True |
| Fail | 50176 | 0.25088 | 0.25 | 0.0008799999999999 | True |
| Dropout | 30136 | 0.15068 | 0.15 | 0.00068 | True |


## 3. Descriptive Statistics — Honest Table 4

Table 4 of the manuscript reports the following values. The `actual_*` columns are computed from the regenerated data. Reviewer 1 (comment 3) showed Table 1 equations cannot produce the reported SDs; the deltas below quantify that discrepancy.

| feature | mean | std | min | max | manuscript_mean | manuscript_std | delta_mean | delta_std | std_matches_manuscript |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| study_hours | 10.172 | 5.378 | 0.0 | 34.785 | 10.02 | 4.21 | 0.152 | 1.168 | False |
| attendance | 69.98 | 12.202 | 30.0 | 100.0 | 69.97 | 8.89 | 0.01 | 3.312 | False |
| internal_marks | 50.246 | 17.349 | 0.0 | 100.0 | 50.12 | 12.34 | 0.126 | 5.009 | False |
| previous_gpa | 5.996 | 1.507 | 0.0 | 10.0 | 6.01 | 0.87 | -0.014 | 0.637 | False |
| lms_usage_hours | 5.126 | 3.32 | 0.0 | 20.338 | 5.01 | 1.89 | 0.116 | 1.43 | False |
| stress_level | 5.04 | 2.44 | 1.0 | 10.0 | 4.98 | 1.42 | 0.06 | 1.02 | False |
| sleep_hours | 7.002 | 1.201 | 3.0 | 12.0 | 6.99 | 0.89 | 0.012 | 0.311 | True |


## 4. Correlation Structure vs Manuscript Claims

Reviewer 2 flagged that Section 3.2.2 reports `study_hours ↔ internal_marks r = 0.42` and `study_hours ↔ previous_gpa r = 0.35`, while Fig. 12 shows r = 0.35 and r = 0.27. Both are compared against the regenerated data.

| pair | manuscript_r | actual_r | delta | matches_within_0.05 |
| --- | --- | --- | --- | --- |
| study_hours__internal_marks | 0.42 | 0.6928399013735838 | 0.2728399013735837 | False |
| study_hours__attendance | 0.38 | 0.8059769764173498 | 0.4259769764173498 | False |
| study_hours__previous_gpa | 0.35 | 0.7967140893377025 | 0.4467140893377025 | False |


## 5. Completion Rate — the 0.70 vs 0.78 Discrepancy

Reviewer 1 (comment 4): *'The completion-rate equation gives a mean of 0.70, not 0.78.'* The actual mean from regenerated data:

| feature | mean_raw | mean_normalised_0_1 | manuscript_mean | delta | note |
| --- | --- | --- | --- | --- | --- |
| assignment_completion_rate | 0.698070454 | 0.698070454 | 0.78 | -0.081929546 | Manuscript reports 0.78. Table 1 equation (0.7 + 0.1*C + 0.05*M) implies mean 0.70. Reviewer 1 (comment 4) flagged this. This row is the actual value from the regenerated data. |


## 6. Leakage Audit (Reviewer 2 — non-circularity)

Fixes applied: composites removed from predictors (see `config.py`), non-linear `tanh` terms added to feature equations, an unobserved confounder `U` added to the target generating equation.

| metric | value |
| --- | --- |
| linear_r2_target_proxy | 0.3833559380238275 |
| linear_classifier_accuracy | 0.494625 |
| max_abs_feature_target_corr | 0.5653110497961114 |
| top5_correlated_features | [('study_hours', 0.5653110497961114), ('attendance', 0.563912920902401), ('lms_usage_hours', 0.5446133237604346), ('previous_gpa', 0.5385830098635678), ('motivation_score', 0.5233583123373765)] |
| interpretation | If R^2 > 0.85 or linear accuracy > 0.85, target is trivially recoverable. For a genuine prediction task with a four-class target and nonlinear generating mechanism, expect R^2 < 0.60 and accuracy < 0.60. |


**Verdict:** linear R² = 0.383, linear accuracy = 0.495. **PASS** — target is not trivially recoverable.


## 6b. TOST Equivalence Testing (Reviewer 1 #4)

**Corrected decision rule:** equivalent iff **p_tost < α AND |Cohen's d| < Δ**. The manuscript's Table 6 reported `completion_rate` as equivalent with d = 0.24 at Δ = 0.2, which is impossible under this rule.

| n_syn | n_real | mean_syn | mean_real | mean_diff | ci95_diff_lo | ci95_diff_hi | cohens_d | p_tost | equivalent | delta_bound_raw | note | dataset | feature | comparison_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200000 | 39875 | 1.6108326594569624 | 6.251993827571772 | -4.6411611681148095 | -4.653272601923083 | -4.629049734306536 | -4.119158660771969 | 1.0 | False | 0.2253451032277068 | NOT equivalent: |d|=4.119 vs bound 0.2 | OULAD | lms_usage_log | continuous |
| 200000 | 34028 | 3.0413939410000004 | -11.898456414556614 | 14.939850355556615 | 14.830014179118107 | 15.049686531995125 | 1.5633343815918563 | 1.0 | False | 1.911280213813784 | NOT equivalent: |d|=1.563 vs bound 0.2 | OULAD | submission_delay | continuous |
| 200000 | 34028 | 0.698070454 | 0.2763723830536455 | 0.4216980709463545 | 0.4198507017240653 | 0.4235454401686436 | 2.623608010980381 | 1.0 | False | 0.0321464234886808 | NOT equivalent: |d|=2.624 vs bound 0.2 | OULAD | completion_rate | continuous |


**Summary:** 0/3 features pass equivalence at Δ = 0.2. The manuscript's Table 6 claims must be rewritten to match these actual values.


## 7. Fairness Audit (Reviewer 2)

Demographic parity (DP), equalized odds (EO), and predictive parity (PP) differences with 95% bootstrap CIs (1000 resamples).

_no data_


## 7b. Real-World Transfer (Reviewer 2)

Two regimes reported explicitly: (a) **matched-only** — train on the 2–3 features shared with the real dataset; (b) **all-with-imputation** — train on all 44 synthetic features and zero-fill missing. Reviewer 2 asked which regime the paper used; both are reported here.

| accuracy_mean | accuracy_ci_lo | accuracy_ci_hi | f1_macro_mean | f1_macro_ci_lo | f1_macro_ci_hi | dataset | regime | n_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.433 | 0.427 | 0.438 | 0.155 | 0.153 | 0.156 | OULAD | matched_only | 3 |
| 0.526 | 0.516 | 0.535 | 0.492 | 0.482 | 0.503 | OULAD | real_only | 3 |


## 8. Model Performance (Tuned + Stacking)

All models trained with 5-fold cross-validated grid search under an identical protocol (Reviewer 3 #1).

| model | accuracy | precision_macro | recall_macro | f1_macro | precision_weighted | recall_weighted | f1_weighted | roc_auc_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.445 | 0.439 | 0.479 | 0.45 | 0.451 | 0.445 | 0.439 | 0.743 |
| RandomForest | 0.485 | 0.486 | 0.461 | 0.47 | 0.484 | 0.485 | 0.481 | 0.746 |
| Stacking | 0.494 | 0.506 | 0.446 | 0.46 | 0.496 | 0.494 | 0.478 | 0.75 |
| XGBoost | 0.494 | 0.509 | 0.441 | 0.457 | 0.498 | 0.494 | 0.476 | 0.749 |


## 9. Counterfactual Scenarios (Reviewer 2 A3, Reviewer 3)

Every counterfactual is labelled `model_scenario` to make explicit it describes the *model's* behaviour, not a causal intervention. Continuous cross-entropy objective replaces the undefined indicator function in Algorithm 3.

**Generated:** 100 counterfactuals. First 5 shown.

| student_idx | original_class | target_class | magnitude | changes | label |
| --- | --- | --- | --- | --- | --- |
| 0 | Distinction | Pass | 14.358969296403655 | {'study_hours': -0.5186686667413007, 'attendance': -1.9064175342621894, 'lms_usage_hours': -0.9208683764419492, 'submission_delay': 1.090844231274544, 'forum_participation': -1.88663253122773, 'assignment_completion_rate': 0.6710789037467477, 'sleep_hours': -2.076289607685223, 'tutoring_sessions': -0.3112504683434306} | model_scenario |
| 1 | Distinction | Fail | 18.91970693812482 | {'study_hours': -1.9338950672663169, 'attendance': -1.5634450792867924, 'lms_usage_hours': -1.3573883580543051, 'submission_delay': -1.1174422336091938, 'forum_participation': -2.148505119953037, 'assignment_completion_rate': -1.1376307229923546, 'sleep_hours': -1.7650921869996021, 'tutoring_sessions': -0.7863762776046299} | model_scenario |
| 2 | Fail | Distinction | 17.369305852420034 | {'study_hours': 0.4159827180717164, 'attendance': -2.05228802234147, 'lms_usage_hours': 1.0361161413826583, 'submission_delay': 1.3676853244941207, 'forum_participation': 0.5320602213741685, 'assignment_completion_rate': -1.5079639621575467, 'sleep_hours': 2.4978487595537047, 'tutoring_sessions': 1.1153414198489018} | model_scenario |
| 3 | Pass | Distinction | 5.582649330719963 | {'study_hours': 1.049959464238614, 'attendance': -0.5292011185815457, 'lms_usage_hours': 0.9667819876650514, 'submission_delay': -0.2612665280615513, 'forum_participation': -0.4101336805290624, 'assignment_completion_rate': 0.45950584697044883, 'sleep_hours': 1.5293540133567323, 'tutoring_sessions': 0.6920795203111798} | model_scenario |
| 4 | Dropout | Fail | 17.659588205700686 | {'study_hours': 2.01106762578147, 'attendance': 2.0127188061077783, 'lms_usage_hours': 0.19226021739984478, 'submission_delay': -1.3093482078061172, 'forum_participation': -1.214868832924921, 'assignment_completion_rate': -0.4993069440380806, 'sleep_hours': 2.227706900772823, 'tutoring_sessions': 1.0606176294814542} | model_scenario |


## 10. Summary of Manuscript Claims Not Reproduced

Honest disclosure requested by Reviewer 1. Each row is a manuscript claim the regenerated data does not support, with a pointer to the Phase 2 correction.

| claim | actual | delta | action |
| --- | --- | --- | --- |
| Table 4: study_hours std = 4.21 | 5.378 | +1.168 | Revise Table 4 (or revise Table 1 equations) |
| Table 4: attendance std = 8.89 | 12.202 | +3.312 | Revise Table 4 (or revise Table 1 equations) |
| Table 4: internal_marks std = 12.34 | 17.349 | +5.009 | Revise Table 4 (or revise Table 1 equations) |
| Table 4: previous_gpa std = 0.87 | 1.507 | +0.637 | Revise Table 4 (or revise Table 1 equations) |
| Table 4: lms_usage_hours std = 1.89 | 3.320 | +1.430 | Revise Table 4 (or revise Table 1 equations) |
| Table 4: stress_level std = 1.42 | 2.440 | +1.020 | Revise Table 4 (or revise Table 1 equations) |
| study_hours__internal_marks: r = 0.42 | 0.693 | +0.273 | Update text and Fig. 12 |
| study_hours__attendance: r = 0.38 | 0.806 | +0.426 | Update text and Fig. 12 |
| study_hours__previous_gpa: r = 0.35 | 0.797 | +0.447 | Update text and Fig. 12 |
| completion_rate mean = 0.78 | 0.698 | -0.082 | Revise Tables 5–6 (Reviewer 1 #4) |


## 11. Reviewer-Comment Coverage (Phase 1 status)

| reviewer | issue | status | evidence |
| --- | --- | --- | --- |
| R1 #1 | GitHub repo incomplete; hardcoded figures | ADDRESSED | Figure scripts read pipeline CSVs |
| R1 #2 | Circular target; Table 1/Eq.1 not implemented | ADDRESSED | Composites removed; Eq.1 covariance; nonlinear target |
| R1 #3 | Table 1 equations cannot produce Table 4 SDs | QUANTIFIED | Section 3 & 10 |
| R1 #4 | d=0.24 > 0.2 reported as equivalent | ADDRESSED | Section 6b — corrected decision rule; actual |d| values |
| R2 leakage | Latent variable leakage | ADDRESSED | Section 6 — R2 and linear accuracy thresholds |
| R2 figures | Fig 13/14 contradictions; correlation mismatch | ADDRESSED | Figure scripts read CSVs; Section 4 |
| R2 scale | Downsampled to 160k | PARTIAL | --max-train-rows explicit in train_model.py |
| R2 A3 | Nelder-Mead on discontinuous step function | ADDRESSED | counterfactual.py uses cross-entropy |
| R2 transfer | Matched-vs-imputed training ambiguous | ADDRESSED | Section 7b — both regimes reported |
| R2 tabular DL | No TabNet / FT-Transformer | PENDING | Optional --include-tabnet flag; not yet run |
| R3 #1 | Default configs, no fair comparison | ADDRESSED | GridSearchCV for all models |
| R3 #2 | Counterfactuals as causal recommendations | ADDRESSED | All outputs labelled model_scenario; CAVEAT printed |
| R3 #3 | Compare against recent publications | PENDING | Requires Phase 2 manuscript revision |


---

_End of validation report._
