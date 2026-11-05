# EduSynX-Bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22052661.svg)](https://doi.org/10.5281/zenodo.22052661)
[![Version](https://img.shields.io/badge/version-1.1.0-blue)](CHANGELOG.md)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

A transparent, reproducible synthetic benchmarking framework for educational data mining with a non-circular target definition, TOST equivalence testing, real-world validation on UCI and OULAD, counterfactual explanations, and fairness auditing.

## What this is

EduSynX-Bench provides:

- A **10-million-row synthetic student dataset** (scalable to 100M) with 45 features spanning demographic, academic, behavioural, psychological, and institutional domains.
- A **non-circular target construction** in which the outcome is built from held-out composites not included as predictors, with non-linear `tanh` transformations and an unobserved confounder to prevent trivial linear inversion.
- **Formal TOST equivalence testing** under the corrected decision rule (`p < α` AND `|Cohen's d| < Δ`).
- **Real-world validation** on UCI Student Performance and OULAD, distinguishing matched-only from all-44-with-imputation training regimes.
- **Constrained-optimization counterfactuals** with a continuous cross-entropy objective and explicit "model scenario" labelling.
- **Bootstrap-based fairness auditing** across socioeconomic status, gender, and parent education.

## What this is not

Synthetic data are a tool for method development and reproducibility infrastructure — **not** a substitute for real-world validation. The paper documents a 9.3–37.2 percentage-point generalization gap on real data. Do not deploy models trained on this dataset to make decisions about real students.

## Headline numbers (v1.1.0, corrected pipeline)

| Metric | v1.0.0 (previous) | v1.1.0 (this release) |
|---|---|---|
| Stacking accuracy | 0.724 | **0.494** |
| TOST equivalent features | 3/6 | **0/10** |
| Real-world transfer gap | 6.7–8.2 pp | **9.3–37.2 pp** |
| SES demographic parity diff. | 0.087 | **0.012** |

The 23-pp accuracy drop quantifies the leakage inflation of the earlier circular target construction.

## Quick start

```bash
# Clone
git clone https://github.com/drsanjayagal/EduSynX.git
cd EduSynX

# Install (Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # Linux/Mac
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Generate 200k-row smoke-test dataset (~30 s)
python generate_data.py --rows 200000 --chunk-size 200000 --output-dir data/ --overwrite

# Train with hyperparameter tuning + stacking (~15–30 min)
python train_model.py --max-train-rows 200000

# Evaluate
python evaluate_model.py

# Leakage diagnostic
python leakage_audit.py --data data/student_chunk_00000.csv --n 200000 --out validation/leakage_audit.csv

# Fairness audit
python fairness_audit.py --model-name Stacking --out validation/fairness_audit.csv

# Counterfactual model scenarios
python counterfactual.py --model-name LightGBM --n-students 20 --out validation/counterfactuals.csv

# Rebuild the validation report
python assemble_validation_report.py --out validation/validation_report.md

python generate_data.py --rows 10000000 --chunk-size 1000000 --output-dir data/ --overwrite

EduSynX/
├── config.py                          Column definitions and constants
├── generate_data.py                   Synthetic data generator (non-circular target)
├── train_model.py                     RF / XGB / LGBM / Stacking with grid search
├── evaluate_model.py                  Metrics, confusion matrices, ROC
├── leakage_audit.py                   Linear reconstruction diagnostic
├── validation.py                      TOST + bootstrap utilities
├── tost_validation.py                 TOST against UCI and OULAD
├── real_world_validation.py           Two-regime transfer analysis
├── counterfactual.py                  Continuous-objective counterfactuals
├── fairness_audit.py                  DP / EO / PP with bootstrap CIs
├── assemble_validation_report.py      Builds validation_report.md
├── preprocessing.py                   Shared preprocessing utilities
├── configs/
│   └── generation_config.yaml         Reproducibility parameters
├── tests/
│   ├── test_schema.py
│   ├── test_target_generation.py
│   └── test_reproducibility.py
├── validation/                        Generated CSVs + validation_report.md
├── results/                           Trained models + metrics (not in git)
├── docs/
│   ├── data_dictionary.csv
│   ├── methodology.md
│   └── reproducibility.md
├── figure/                            Figure-generation scripts
├── requirements.txt
├── CHANGELOG.md
├── CITATION.cff
└── LICENSE

pytest tests/ -v

@dataset{agal2026edusynx,
  title     = {EduSynX-Bench: Corrected Non-Circular Synthetic Student Dataset and Benchmarking Framework},
  author    = {Agal, Sanjay and Macwan, Kiran and Farooqui, Yassir and Mehta, Nirav Pareshkumar and Shrivastava, Shiv Shakti},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.1.0},
  doi       = {10.5281/zenodo.22052661},
  url       = {https://doi.org/10.5281/zenodo.22052661}
}