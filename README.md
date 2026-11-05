# EduSynX-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22052661.svg)](https://doi.org/10.5281/zenodo.22052661)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](CHANGELOG.md)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

A transparent, reproducible synthetic benchmarking framework for educational data mining, with a non-circular target definition, TOST equivalence testing, real-world validation on UCI and OULAD, continuous-objective counterfactual explanations, and bootstrap-based fairness auditing.

## Dataset

| Field | Value |
|---|---|
| **Version DOI (v2.0.0)** | [10.5281/zenodo.23979918](https://doi.org/10.5281/zenodo.23979918) |
| **Concept DOI (latest)** | [10.5281/zenodo.22052661](https://doi.org/10.5281/zenodo.22052661) |
| **License** | CC BY 4.0 |
| **Rows** | 10,000,000 (chunked CSVs, 1M each) |
| **Features** | 45 (44 predictors + 1 target) |
| **Target classes** | Distinction (20.0%), Pass (39.8%), Fail (25.1%), Dropout (15.1%) |

**Cite the version DOI** if you want to reference the exact dataset used in the paper. **Cite the concept DOI** if you want your link to always resolve to the newest version.

## ⚠ v2.0.0 is a breaking change from v1.0.0

The dataset has been substantially rewritten. If you used v1.0.0:

- **You must retrain from scratch.** The feature schema changed (`performance_index`, `engagement_score`, `consistency_score`, `risk_score`, `improvement_trend` are no longer predictors — they were leaking the target).
- **Your previous benchmark numbers are not comparable.** The corrected pipeline produces 0.494 accuracy where v1.0.0 produced 0.724. The 23-pp difference quantifies the leakage inflation of the earlier version.
- **Your old test split is invalid.** Different random seed distribution after threshold recalibration.

Version v1.0.0 remains on Zenodo for historical reproducibility of the original (incorrect) results. Do not mix v1.0.0-trained models with v2.0.0 data.

## What this is

EduSynX-Bench provides:

- A **10-million-row synthetic student dataset** (scalable to 100M) with 45 features spanning demographic, academic, behavioural, psychological, and institutional domains.
- A **non-circular target construction** in which the outcome is built from held-out composites not included as predictors, with non-linear `tanh` transformations and an unobserved confounder to prevent trivial linear inversion.
- **Formal TOST equivalence testing** under the corrected decision rule (`p < α` AND `|Cohen's d| < Δ`).
- **Real-world validation** on UCI Student Performance and OULAD, distinguishing matched-only from all-44-with-imputation training regimes.
- **Constrained-optimization counterfactuals** with a continuous cross-entropy objective and explicit `model_scenario` labelling.
- **Bootstrap-based fairness auditing** across socioeconomic status, gender, and parent education.

## What this is not

Synthetic data are a tool for method development and reproducibility infrastructure — **not** a substitute for real-world validation. The paper documents a **9.3–37.2 percentage-point generalization gap** on real data. Do not deploy models trained on this dataset to make decisions about real students.

## Headline numbers (v2.0.0, corrected pipeline)

| Metric | v1.0.0 (previous) | v2.0.0 (this release) |
|---|---|---|
| Stacking accuracy | 0.724 | **0.494** |
| TOST equivalent features | 3/6 | **0/10** |
| Real-world transfer gap (OULAD) | 6.7 pp | **9.3 pp** |
| Real-world transfer gap (UCI) | 8.2 pp | **37.2 pp** |
| SES demographic parity diff. | 0.087 | **0.012** |

The 23-pp accuracy drop quantifies the leakage inflation of the earlier circular target construction. Full validation report: [`validation/validation_report.md`](validation/validation_report.md).

## Quick start

```bash
# Clone
git clone https://github.com/drsanjayagal/EduSynX.git
cd EduSynX

# Install (Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # Linux / macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Run the pipeline

# 1) Generate a 200k-row smoke-test dataset (~30 s)
python generate_data.py --rows 200000 --chunk-size 200000 --output-dir data/ --overwrite

# 2) Train with grid search + stacking (~15–30 min on a laptop)
python train_model.py --max-train-rows 200000

# 3) Evaluate
python evaluate_model.py

# 4) Leakage diagnostic (Reviewer 2's concern, quantified)
python leakage_audit.py --data data/student_chunk_00000.csv --n 200000 --out validation/leakage_audit.csv

# 5) Fairness audit
python fairness_audit.py --model-name Stacking --out validation/fairness_audit.csv

# 6) Counterfactual model scenarios
python counterfactual.py --model-name LightGBM --n-students 20 --out validation/counterfactuals.csv

# 7) Rebuild the validation report
python assemble_validation_report.py --out validation/validation_report.md

EduSynX/
├── config.py                          Column definitions and constants
├── generate_data.py                   Synthetic data generator (non-circular target)
├── train_model.py                     RF / XGB / LGBM / Stacking with grid search
├── evaluate_model.py                  Metrics, confusion matrices, ROC
├── preprocessing.py                   Shared preprocessing utilities
├── leakage_audit.py                   Linear reconstruction diagnostic
├── validation.py                      TOST + bootstrap utilities
├── tost_validation.py                 TOST against UCI and OULAD
├── real_world_validation.py           Two-regime transfer analysis
├── counterfactual.py                  Continuous-objective counterfactuals
├── fairness_audit.py                  DP / EO / PP with bootstrap CIs
├── assemble_validation_report.py      Builds validation_report.md
├── decision_support.py                Model-scenario insight generator
├── explainability.py                  SHAP analysis
│
├── configs/
│   └── generation_config.yaml         Reproducibility parameters
│
├── tests/
│   ├── test_schema.py                 45-column schema check
│   ├── test_target_generation.py      No-leakage target generation check
│   └── test_reproducibility.py        Seed reproducibility check
│
├── validation/                        Generated CSVs + validation_report.md
│   ├── validation_report.md           Full traceability report
│   ├── statistical_summary.csv        Honest Table 4 replacement
│   ├── correlation_check.csv          Text-vs-actual correlation check
│   ├── class_distribution.csv         Actual vs target proportions
│   ├── completion_rate_check.csv      The 0.70-vs-0.78 correction
│   ├── leakage_audit.csv              R² and accuracy for target reconstruction
│   ├── tost_results.csv               10-row TOST table (0/10 equivalent)
│   ├── real_world_validation.csv      5-regime transfer results
│   ├── fairness_audit.csv             7-comparison fairness metrics
│   └── counterfactuals.csv            100 labelled model_scenario rows
│
├── results/                           Trained models + metrics (not in git)
│   ├── metrics.csv                    Model comparison
│   ├── tuning_log.json                Grid-search results
│   ├── correlation_matrix.csv         Feature correlations
│   ├── cm_*.csv                       Confusion matrices per model
│   └── roc_data.csv                   ROC curves for figure regeneration
│
├── figure/                            Figure-generation scripts
│   ├── run_all_figures.py             Runs all 14 figure scripts
│   ├── generate_framework_figure.py
│   ├── generate_data_distribution_figure.py
│   ├── generate_validation_comparison_figure.py
│   ├── generate_confusion_matrices_figure.py
│   ├── generate_model_performance_figure.py
│   ├── generate_learning_curves_figure.py
│   ├── generate_shap_figure.py
│   ├── run_lime_analysis.py
│   ├── generate_counterfactual_figure.py
│   ├── generate_fairness_figure.py
│   ├── generate_robustness_figure.py
│   ├── generate_correlation_matrix_figure.py
│   ├── generate_discussion_summary_figure.py
│   └── generate_conclusion_summary_figure.py
│
├── docs/
│   ├── data_dictionary.csv            45-feature machine-readable schema
│   ├── methodology.md                 Methodology summary
│   └── reproducibility.md             Reproduction instructions
│
├── requirements.txt                   Pinned dependencies
├── CHANGELOG.md                       Version history
├── CITATION.cff                       Citation metadata
├── LICENSE                            CC BY 4.0
└── README.md                          This file

@dataset{agal2026edusynx,
  title     = {EduSynX-Bench: Corrected Non-Circular Synthetic Student Dataset and Benchmarking Framework},
  author    = {Agal, Sanjay and Macwan, Kiran and Farooqui, Yassir and Mehta, Nirav Pareshkumar and Shrivastava, Shiv Shakti},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.23979918},
  url       = {https://doi.org/10.5281/zenodo.23979918},
  note      = {Concept DOI: 10.5281/zenodo.22052661 (always resolves to latest version)}
}

