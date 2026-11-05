# Changelog

All notable changes to EduSynX-Bench are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-05

### Added
- Non-linear `tanh` transformations and unobserved confounder in the target-generating equation (addresses Reviewer 2 latent-leakage concern).
- Empirical threshold calibration on a 200,000-row pilot sample (fixes class-distribution mismatch flagged by Reviewer 1 #3).
- `counterfactual.py` with continuous cross-entropy objective (fixes Reviewer 2's Algorithm-3 undefined-indicator error).
- `fairness_audit.py` with 1,000-resample bootstrap CIs (addresses Reviewer 2 fairness requirement).
- `leakage_audit.py` with linear-reconstruction diagnostic.
- `validation.py` with corrected TOST decision rule.
- `tost_validation.py` and `real_world_validation.py` for UCI and OULAD.
- `tests/` directory: `test_schema.py`, `test_target_generation.py`, `test_reproducibility.py`.
- `docs/data_dictionary.csv` — 45-feature machine-readable dictionary.
- `validation/validation_report.md` — full traceability report.

### Changed
- `config.py`: removed `performance_index`, `engagement_score`, `consistency_score`, `risk_score`, `improvement_trend` from `FEATURE_COLS` — these were the circular predictors.
- `train_model.py`: added `GridSearchCV` for all models under identical 5-fold CV (addresses Reviewer 3 #1).
- `generate_data.py`: implements Eq.(1) covariance matrix and Table 1 equations faithfully.
- `requirements.txt`: pinned versions, added `faker`, `tqdm`, `joblib`, `pyyaml`; `matplotlib` bumped to 3.10.9 for Windows wheel availability.
- All figure scripts now read pipeline outputs; hardcoded arrays removed.

### Fixed
- TOST equivalence: `completion_rate` with `d = 0.24` at `Δ = 0.2` was previously reported as equivalent; corrected to **not equivalent**. The correction extends to all 10 matched features (0/10 pass).
- Model accuracy: 0.494 (corrected) vs 0.724 (previous). The 23-pp drop quantifies the leakage inflation.
- Fairness disparities: 7× smaller in corrected pipeline (max SES DP 0.012 vs 0.087).
- Figure 13/14 content now consistent with text (real-world validation, LIME, counterfactuals, fairness are completed).
- Correlation values corrected to 0.69/0.81/0.80 (actual products of Table 1 equations, vs previous 0.42/0.38/0.35).

### Removed
- `performance_index`, `engagement_score`, `consistency_score`, `risk_score`, `improvement_trend` from predictor set (leakage).
- Hardcoded accuracy arrays in figure scripts.
- Synthetic SHAP fallback in `generate_shap_figure.py`.

## [1.0.0] - 2025-XX-XX

Initial release. Zenodo concept DOI: [10.5281/zenodo.22052661](https://doi.org/10.5281/zenodo.22052661)