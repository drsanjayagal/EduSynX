````md
# EduSynX

**A Machine Learning Framework with a Public Synthetic Student Dataset for Explainable Decision Support and Academic Success Prediction**

---

## 🚀 Overview

**EduSynX** is a scalable machine learning framework designed to generate large-scale synthetic student datasets (ranging from **10 million to 100+ million records**) with realistic feature correlations and class imbalance.

It provides a complete, end-to-end pipeline for:

- 📊 Large-scale CSV dataset generation (chunk-based)
- ⚙️ Data preprocessing (scaling + categorical encoding)
- 🤖 Model training (Random Forest, XGBoost, LightGBM)
- 📈 Performance evaluation (metrics + visualization)
- 🔍 Explainability using SHAP
- 🧠 Actionable decision-support insights

---

## 📂 Dataset Description

The dataset is generated in **chunks** and stored as multiple CSV files within the `data/` directory.

- Each row represents **one student**
- Total features: **45 (44 predictors + 1 target)**

---

## 🧩 Feature Categories

| Category        | Features |
|----------------|----------|
| **Demographics** | age, gender, region, socioeconomic_status, parent_education |
| **Academic**     | study_hours, attendance, internal_marks, previous_gpa, assignments_score, credit_load, backlog_count, semester, online_learning_hours, library_usage_hours |
| **Behavioral**   | LMS_usage_time, login_frequency, submission_delay, forum_participation, video_watch_time, assignment_completion_rate, extra_curricular_hours |
| **Psychological**| stress_level, anxiety_score, motivation_score, self_efficacy_score, peer_support_score, family_support_score, sleep_hours, physical_activity_hours |
| **Institutional**| course_difficulty, faculty_rating, class_size, lab_availability_hours, internet_access, scholarship, campus_distance, tutoring_sessions, part_time_job_hours |
| **Derived**      | performance_index, engagement_score, consistency_score, risk_score, improvement_trend |
| **Target**       | final_result *(Pass, Fail, Dropout, Distinction)* |

---

## 🔗 Feature Engineering Logic

Feature relationships are explicitly modeled to mimic real-world dependencies:

- `study_hours → assignments_score → performance_index`
- `attendance → internal_marks → final_result`

Noise injection is applied to simulate real-world variability and uncertainty.

---

## ⚙️ Installation & Usage

### 1. Generate Dataset

```bash
python generate_data.py --rows 10000000 --chunk-size 1000000 --output-dir data --output-mode multiple

python generate_data.py --rows 100000000 --chunk-size 1000000 --output-dir data
````

### 2. Train Models

```bash
python train_model.py --data-dir data --max-train-rows 200000 --test-size 0.2 --output-dir results
```

### 3. Evaluate Models

```bash
python evaluate_model.py --results-dir results
```

### 4. Explainability (SHAP)

```bash
python explainability.py --results-dir results --model-name XGBoost --sample-size 500
```

### 5. Decision Support

```bash
python decision_support.py --results-dir results --model-name XGBoost --num-students 5
```

---

## 📁 Project Structure

```bash
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
```

---

## 📦 Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Key Highlights

* Scalable to **100M+ records**
* Fully **synthetic (privacy-safe) dataset**
* Built-in **Explainable AI (XAI)**
* Designed for **academic decision support systems**
* Suitable for **research, benchmarking, and deployment**

---

## 📌 Use Cases

* Academic success prediction
* Student risk identification
* Institutional policy simulation
* AI research benchmarking
* Explainable AI in education systems

---

## 🧠 Summary

**EduSynX** bridges the gap between **data availability** and **intelligent decision-making** in education through scalable synthetic data and explainable machine learning.

```
```
