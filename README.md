```md
# EduSynX

**A Machine Learning Framework with a Public Synthetic Student Dataset for Explainable Decision Support and Academic Success Prediction**

---

## Author  
**Dr. Sanjay Agal**  
Professor & Head, Artificial Intelligence & Data Science  
Parul University, India  

---

## DOI  
https://doi.org/10.5281/zenodo.22052661  

---

## Abstract

EduSynX is a research-grade framework for generating large-scale synthetic student datasets and building explainable machine learning models for academic outcome prediction. The system integrates data generation, modeling, evaluation, and interpretability into a unified pipeline, enabling reproducible and privacy-preserving research in educational data science.

---

## Key Contributions

- A scalable synthetic dataset generator supporting 10M–100M+ records  
- Explicit modeling of academic feature dependencies  
- Integrated machine learning pipeline (RF, XGBoost, LightGBM)  
- Explainable AI using SHAP for transparent decision-making  
- DOI-backed dataset for reproducibility and citation  

---

## System Overview

```

Synthetic Data Generation
↓
Preprocessing
↓
Model Training
↓
Evaluation
↓
Explainability
↓
Decision Support

````

---

## Dataset Description

- **Type:** Fully Synthetic (No real student data)  
- **Format:** CSV (chunk-based storage)  
- **Scale:** Configurable (10M to 100M+ records)  
- **Features:** 45 total (44 predictors + 1 target)  

### Feature Groups

| Category        | Description |
|----------------|------------|
| Demographics   | Socio-economic and personal attributes |
| Academic       | Study patterns, GPA, attendance, assessments |
| Behavioral     | LMS usage and engagement metrics |
| Psychological  | Motivation, stress, and support indicators |
| Institutional  | Learning environment and resources |
| Derived        | Computed indices (performance, risk, engagement) |
| Target         | Final outcome (Pass, Fail, Dropout, Distinction) |

---

## Feature Modeling

The dataset incorporates structured relationships to reflect realistic academic behavior:

- `study_hours → assignments_score → performance_index → final_result`  
- `attendance → internal_marks → final_result`  

Controlled noise is introduced to simulate real-world variability.

---

## Installation

```bash
pip install -r requirements.txt
````

---

## Usage

### Generate Dataset

```bash
python generate_data.py --rows 10000000 --chunk-size 1000000 --output-dir data --output-mode multiple
```

### Train Model

```bash
python train_model.py --data-dir data --max-train-rows 200000 --test-size 0.2 --output-dir results
```

### Evaluate Model

```bash
python evaluate_model.py --results-dir results
```

### Explainability

```bash
python explainability.py --results-dir results --model-name XGBoost --sample-size 500
```

### Decision Support

```bash
python decision_support.py --results-dir results --model-name XGBoost --num-students 5
```

---

## Project Structure

```
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

## Applications

* Academic success prediction
* Student risk and dropout analysis
* Institutional decision support systems
* Benchmarking for educational AI models
* Explainable AI research

---

## Reproducibility

The dataset and framework are publicly available via DOI:

https://doi.org/10.5281/zenodo.22052661

This ensures long-term accessibility, citation, and reproducibility of results.

---

## Citation

```
Agal, S. (2026). EduSynX: Synthetic Student Dataset for Explainable Academic Prediction.
Zenodo. https://doi.org/10.5281/zenodo.22052661
```

---

## Conclusion

EduSynX provides a unified and scalable framework for combining synthetic data generation with explainable machine learning, enabling transparent and reproducible research in academic analytics and intelligent educational systems.

```
```
