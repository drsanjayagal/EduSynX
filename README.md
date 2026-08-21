# EduSynX

A Machine Learning Framework with a Public Synthetic Student Dataset for Explainable Decision Support and Academic Success Prediction.

## Overview

EduSynX generates a very large synthetic student dataset (10M–100M+ rows) with realistic correlations and class imbalance, and provides a complete machine learning pipeline:

- Chunk-based CSV data generation
- Preprocessing with scaling and categorical encoding
- Model training (RandomForest, XGBoost, LightGBM)
- Evaluation with metrics and plots
- SHAP-based explainability
- Actionable decision support insights

## Dataset Description

The dataset is generated in chunks and saved as multiple CSV files in the `data/` directory. Each row represents one student with 45 features (44 predictors + 1 target).

### Feature Categories

| Category | Features |
|----------|----------|
| Demographics | age, gender, region, socioeconomic_status, parent_education |
| Academic | study_hours, attendance, internal_marks, previous_gpa, assignments_score, credit_load, backlog_count, semester, online_learning_hours, library_usage_hours |
| Behavioral | LMS_usage_time, login_frequency, submission_delay, forum_participation, video_watch_time, assignment_completion_rate, extra_curricular_hours |
| Psychological | stress_level, anxiety_score, motivation_score, self_efficacy_score, peer_support_score, family_support_score, sleep_hours, physical_activity_hours |
| Institutional | course_difficulty, faculty_rating, class_size, lab_availability_hours, internet_access, scholarship, campus_distance, tutoring_sessions, part_time_job_hours |
| Derived | performance_index, engagement_score, consistency_score, risk_score, improvement_trend |
| Target | final_result (Pass, Fail, Dropout, Distinction) |

Feature correlations are modelled explicitly (e.g., `study_hours` → `assignments_score` → `performance_index`), with noise injection to simulate real-world variability.

## Installation

python generate_data.py --rows 10000000 --chunk-size 1000000 --output-dir data --output-mode multiple

python generate_data.py --rows 100000000 --chunk-size 1000000 --output-dir data

python train_model.py --data-dir data --max-train-rows 200000 --test-size 0.2 --output-dir results

python evaluate_model.py --results-dir results

python explainability.py --results-dir results --model-name XGBoost --sample-size 500

python decision_support.py --results-dir results --model-name XGBoost --num-students 5

EduSynX/
├── generate_data.py
├── preprocessing.py
├── train_model.py
├── evaluate_model.py
├── explainability.py
├── decision_support.py
├── config.py
├── requirements.txt
└── README.md





```bash
pip install -r requirements.txt
