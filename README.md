# Machine Learning Lab Projects (AI-3202)

**Dawood University of Engineering & Technology, Karachi**
Department of Artificial Intelligence · BS Artificial Intelligence
Course: Machine Learning (Practical), AI-3202 · Instructor: Engr. Hamza Farooqui
Author: Noman Ali (24F-AI-202)

---

## Projects in this repository

| # | Project | Lab | Work type | Folder |
|---|---------|-----|-----------|--------|
| 1 | Health Risk Classification | Lab 07: Open Ended Lab | Individual | `open-ended-lab-health-risk/` |
| 2 | AI-Powered Student Performance Prediction and Early Warning System | Lab 14: Complex Computing Activity (CCA) | Group of 4 | `cca-student-performance-early-warning/` |

## Repository structure

```
.
├── README.md
├── open-ended-lab-health-risk/
│   ├── health_risk_classification.ipynb      # [FILL IN: your notebook name]
│   └── data/                                 # [FILL IN: dataset file(s)]
└── cca-student-performance-early-warning/
    ├── app.py                  # Flask web application and API endpoints
    ├── train_model.py          # Training, evaluation and model saving
    ├── database.py             # SQLite tables, prediction and recommendation logic
    ├── requirements.txt
    ├── README.md               # (optional) copy of Project 2 section below
    ├── dataset/student_data.csv
    ├── models/                 # saved .pkl models and model_metadata.json
    ├── templates/              # HTML pages (Bootstrap 5)
    ├── static/                 # CSS and JavaScript (Chart.js)
    └── student_performance.db  # SQLite database, pre-filled with students
```

> Folder names must be lowercase (`models`, `dataset`, `templates`, `static`). Flask and the Python scripts look for these exact names, and Linux and macOS are case-sensitive.

---

# Project 1: Health Risk Classification

**Lab 07: Open Ended Lab**

## Overview

The open-ended lab asks students to apply the concepts from Labs 1 to 6 in one small, working machine learning project. This project classifies **health risk** using supervised machine learning in Python.

- Goal: [FILL IN: what is predicted, for example the risk classes or labels]
- Dataset: [FILL IN: dataset name, source link, number of rows and columns, target column]
- Preprocessing: [FILL IN: missing values, encoding, scaling, train/test split ratio]
- Models used: [FILL IN: at least two models, for example Logistic Regression, Decision Tree, Random Forest, SVM, KNN]
- How the models are combined or compared: [FILL IN]

## Lab concepts used

| Lab | Concept | Used for |
|-----|---------|----------|
| 01 | Data preprocessing | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] |

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

Confusion matrices and plots are inside the notebook.

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook open-ended-lab-health-risk/health_risk_classification.ipynb
```

Run all cells from top to bottom. Add any other library you imported to the `pip install` line.

---

# Project 2: AI-Powered Student Performance Prediction and Early Warning System

**Lab 14: Complex Computing Activity (CCA)**

## Group members

| Name | Roll number |
|------|-------------|
| Noman Ali | 24F-AI-202 |
| Muhammad Ibrahim | 24F-AI-201 |
| Ali Muhammad | 24F-AI-203 |
| Muhammad Gulab | 24F-AI-204 |

## Overview

Students who are failing are often identified only after final grades are published, when it is too late to help. This project is a Flask web application that predicts which students are at risk of failing, groups students into three behaviour personas, and gives an academic advisor a recommended action for every flagged student.

It combines four labs from the practical manual (the manual requires at least three):

| Lab | Concept | Where it is used |
|-----|---------|------------------|
| 01 | Data preprocessing | Encoding yes/no columns, building the target, `StandardScaler`, stratified 80/20 split |
| 03 | Logistic Regression | Risk probability model and comparison baseline |
| 05 | Random Forest | Main classifier and feature importance |
| 10 | K-Means and the Elbow Method | Three student personas |

## Features

- **Dashboard** (`/dashboard`): student counts, at-risk and on-track totals, average grades and absences, and Chart.js charts.
- **Students** (`/students`): add, view, edit and delete records, search, filter by risk, sort, re-score all students, export CSV.
- **Prediction** (`/prediction`): enter a student's details and get a risk probability from Random Forest, Logistic Regression or a 50/50 average, plus a K-Means persona and recommended actions. The result can be saved to the database.
- **Early Warning** (`/early-warning`): queue of flagged students where advisors set intervention status (Pending, In Progress, Resolved), counselor and notes.
- **Reports** (`/reports`): model comparison table, confusion matrices, feature importance, elbow curve and persona profiles.
- **About** (`/about`): lab mapping, technology stack and ethical notes.

## How it works

```
student_data.csv
      |
      v
Preprocessing (Lab 01)   encode 6 yes/no columns, build target (G3 < 10), drop G3 from inputs,
      |                  StandardScaler, stratified 80/20 split
      v
Models (train_model.py)  Logistic Regression (Lab 03) | Random Forest (Lab 05) | K-Means k = 3 (Lab 10)
      |
      v   saved as .pkl files in models/
Risk engine (database.py)  probability -> risk level -> persona -> recommended actions
      |
      v
SQLite (students, predictions, interventions)  <->  Flask app (app.py)  ->  web pages
```

## Dataset

UCI Student Performance dataset, Math course (Cortez and Silva, 2008): 395 students and 33 attributes. 17 attributes are used as input features:

`age, sex, studytime, failures, absences, G1, G2, schoolsup, famsup, paid, higher, internet, Medu, Fedu, freetime, goout, health`

- **Target:** At Risk = 1 if the final grade `G3` is below 10 (pass mark is 10 out of 20), otherwise 0.
- **Class balance:** 265 not at risk (67.1%) and 130 at risk (32.9%).
- **Data leakage:** `G3` is used only to build the target and is never part of the input features.
- **Missing values:** none found.

## Models and settings

| Model | Lab | Settings |
|-------|-----|----------|
| Logistic Regression | 03 | `C=1.0`, `max_iter=1000`, `random_state=42` |
| Random Forest | 05 | `n_estimators=100`, `max_depth=6`, `min_samples_split=4`, `random_state=42` |
| K-Means | 10 | `k=3`, `n_init=10`, `random_state=42`, on `studytime, absences, failures, G1, G2` (scaled) |

Train/test split: 316 training and 79 test students (80/20, stratified, `random_state=42`).

## Results

Evaluated on the 79 held-out students (53 not at risk, 26 at risk). "At Risk" is the positive class.

| Metric | Logistic Regression | Random Forest |
|--------|--------------------:|--------------:|
| Accuracy | 92.41% | 91.14% |
| Precision | 88.46% | 85.19% |
| Recall | 88.46% | 88.46% |
| F1-score | 88.46% | 86.79% |
| ROC-AUC | 97.82% | 96.37% |

| Confusion matrix | TN | FP | FN | TP |
|------------------|---:|---:|---:|---:|
| Logistic Regression | 50 | 3 | 3 | 23 |
| Random Forest | 49 | 4 | 3 | 23 |

Both models find 23 of the 26 at-risk students. The difference between them is one or two students out of 79, so they perform about the same on this split.

**Most important features (Random Forest):** G2 (48.0%), G1 (25.7%), past failures (4.2%), absences (4.1%), going out (2.7%). The two period grades together carry about 74% of the importance.

**K-Means personas (k = 3):**

| Persona | Students | Avg G1 | Avg G2 | Avg absences | Avg failures |
|---------|---------:|-------:|-------:|-------------:|-------------:|
| High Achiever and Dedicated | 154 (39.0%) | 14.31 | 14.21 | 4.29 | 0.08 |
| Moderate Performer / Watchlist | 199 (50.4%) | 9.11 | 8.92 | 6.99 | 0.15 |
| Academic Risk / High Needs | 42 (10.6%) | 6.98 | 6.38 | 4.83 | 2.12 |

## Risk levels

| Risk probability | Level |
|------------------|-------|
| 0.70 or higher | High |
| 0.40 to 0.69 | Moderate |
| Below 0.40 | Low |

A student is flagged **At Risk** when the probability is 0.50 or higher. Recommendations depend on the student's data, for example attendance monitoring for more than 8 absences, tutorial sessions when G1 or G2 is below 10, and a study plan when study time is low.

## Technology stack

- Python 3.10 or newer, Flask 3
- scikit-learn, pandas, NumPy, joblib
- SQLite 3
- Bootstrap 5, Chart.js

## Installation and running

**1. Get the code**

```bash
git clone <your-repository-url>
cd cca-student-performance-early-warning
```

**2. Create an environment** (pip or conda)

```bash
# pip
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# or conda
conda create -n early-warning python=3.11 -y
conda activate early-warning
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. (Optional) Retrain the models**

```bash
python train_model.py
```

This cleans the data, trains Logistic Regression, Random Forest and K-Means, and rewrites the files in `models/`.

**5. Start the web app**

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser.

Hardware: a normal laptop is enough. No GPU is needed.

## Limitations

- The dataset is small (395 students, 79 in the test set) and the results come from a single split, so the scores can change with a different split.
- The models rely mostly on G1 and G2, so the warning comes after the first two exams, not before the course starts.
- The data comes from one school and one subject. It should be retrained on local data before real use.
- Inputs such as sex, age and parental education should be reviewed for fairness before any real deployment.

## Ethical use

Predictions are advisory. They are meant to direct support, tutoring and mentoring to students, not to punish them. A human advisor makes every final decision.

## Acknowledgements

Dataset: P. Cortez and A. Silva, "Using Data Mining to Predict Secondary School Student Performance," Proceedings of the 5th FUture BUsiness TEChnology Conference (FUBUTEC), 2008. UCI Machine Learning Repository.

Developed for academic evaluation under the Dawood University of Engineering & Technology syllabus for Machine Learning (Practical), AI-3202.
