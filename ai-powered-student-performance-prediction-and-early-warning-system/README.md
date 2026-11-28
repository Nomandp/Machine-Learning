# AI-Powered Student Performance Prediction and Early Warning System

**Machine Learning (Practical) AI-3202 · Lab 14: Complex Computing Activity (CCA)**  
*Dawood University of Engineering & Technology (DUET), Karachi*  
*Department of Artificial Intelligence — Batch 2024*  
*Instructor: Engr. Hamza Farooqui*

---

## 👥 Group Members

| Name | Roll Number |
|------|-------------|
| Noman Ali | 24F-AI-202 |
| Muhammad Ibrahim | 24F-AI-201 |
| Ali Muhammad | 24F-AI-203 |
| Muhammad Gulab | 24F-AI-204 |

---

## 📌 Project Overview

Academic failure and dropouts in tertiary education are often diagnosed too late in the semester to mount effective interventions. The **AI-Powered Student Performance Prediction and Early Warning System** is an end-to-end, production-ready Machine Learning web application designed to proactively predict student academic risk, discover latent behavioral learning personas, and automatically trigger targeted remedial interventions.

The project rigorously adheres to the requirements of **Lab 14 (Complex Computing Activity)** by synthesizing four foundational machine learning laboratories:
- **Lab 01**: Data Preprocessing, Cleaning, Encoding, and Feature Scaling (StandardScaler).
- **Lab 03**: Logistic Regression (Probabilistic Binary Classification via Sigmoid Activation).
- **Lab 05**: Random Forest Classifier (Ensemble Bagging with Feature Importance Extraction).
- **Lab 10**: K-Means Clustering (Unsupervised Academic Pattern Segmentation & Elbow Method).
- **Strict Data Leakage Avoidance**: The final examination grade (`G3`) is isolated exclusively as the prediction target and strictly eliminated from the feature input matrix ($X$).

---

## 🌟 Key Application Features

1. **Executive Dashboard (`/dashboard`)**:
   - Live KPI counters: Total Students, Flagged At-Risk, On-Track, Pending Interventions, Cohort Average Exam Scores ($G_1, G_2$), and Absences.
   - Interactive Chart.js visualizations:
     - Supervised Risk Status Distribution (Doughnut Chart)
     - Grade Progression Trajectory ($G_1$ vs. $G_2$ Bar Chart)
     - Unsupervised K-Means Behavioral Personas (Polar Area Chart)
     - Attendance vs. Predicted Risk Correlation (Dual-Axis Chart)
   - Recent High-Priority Early Warning alerts with one-click intervention review.

2. **Student Directory & Record Management (`/students`)**:
   - Full CRUD support: Add, View, Edit, and Delete student records.
   - Real-time search across student names and student IDs.
   - Multi-criteria filtering by Risk Status (*All*, *At Risk*, *On Track*) and dynamic sorting.
   - One-click batch re-evaluation of the entire student database using trained models.
   - CSV export functionality (`/students/export-csv`).

3. **Live AI Risk Prediction Engine (`/prediction`)**:
   - Interactive parameter input for academic grades ($G_1, G_2$), study hours, absences, past failures, and socio-demographic indicators.
   - Model selection: **Random Forest (Recommended)**, **Logistic Regression**, or **Consensus Ensemble (50/50 Average)**.
   - Instant visual risk output: Animated progress meter, severity level (*High*, *Moderate*, *Low*), and dual-model comparative probabilities.
   - **K-Means Persona Assignment**: Assigns the student to one of three behavioral clusters (*High Achiever*, *Moderate Watchlist*, *Academic Risk*).
   - **Personalized Recommendations**: Automatically generates tailored guidance based on detected deficits (e.g., attendance monitoring, structured study schedule, remedial tutoring).
   - Direct option to save the prediction to the SQLite database.

4. **Early Warning & Intervention Center (`/early-warning`)**:
   - High and moderate risk student queues with automated risk severity badges.
   - Dedicated intervention tracking: Faculty advisors can update intervention status (*Pending*, *In Progress*, *Resolved*), assign counselors, and record progress notes.

5. **Model Evaluation & Analytics (`/reports`)**:
   - Side-by-side benchmark table comparing **Logistic Regression** and **Random Forest** across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Interactive $2 \times 2$ Confusion Matrices displaying True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
   - Random Forest Feature Importance chart identifying the primary predictors of academic risk.
   - K-Means Elbow Method curve ($k=1$ to $8$) demonstrating why $k=3$ is the optimal number of clusters.
   - Cluster centroid table detailing average grades, study hours, absences, and failure rates per cluster.

6. **Educational & Syllabus Guide (`/about`)**:
   - Complete architectural breakdown, Lab mapping documentation, technology stack inventory, and ethical AI in education principles.

---

## 🛠️ Technology Stack

- **Backend Web Framework**: Python 3.13, Flask 3.1
- **Machine Learning**: Scikit-Learn 1.9, NumPy 2.3, Pandas 3.1, SciPy 1.16, Joblib
- **Database**: SQLite 3 (Thread-safe, relational schema)
- **Frontend UI**: HTML5, CSS3, Bootstrap 5.3, FontAwesome 6
- **Data Visualizations**: Chart.js 4.4
- **Dataset**: UCI Machine Learning Repository — *Student Performance Dataset* (Cortez & Silva, 2008)

---

## 📂 Project Structure

```
AI POWERED STUDENT PERFORMANCE PREDICTION AND EARLY WARNING SYSTEM/
│
├── app.py                      # Core Flask web application and REST API endpoints
├── train_model.py              # ML training, evaluation, and serialization pipeline
├── database.py                 # SQLite database models, seeding, and inference helpers
├── requirements.txt            # Python dependencies
├── README.md                   # Installation, setup, and usage documentation
├── PROJECT_REPORT.md           # Formal 5-page Lab 14 CCA academic project report
│
├── dataset/
│   └── student_data.csv        # Authentic UCI Student Performance dataset (Math)
│
├── models/
│   ├── random_forest.pkl       # Serialized Random Forest classifier (Lab 05)
│   ├── logistic_regression.pkl # Serialized Logistic Regression classifier (Lab 03)
│   ├── kmeans.pkl              # Serialized K-Means clustering model (Lab 10)
│   ├── scaler.pkl              # Feature scaling transformer (StandardScaler - Lab 01)
│   ├── cluster_scaler.pkl      # Clustering feature scaling transformer
│   └── model_metadata.json     # Saved metrics, confusion matrices, and cluster profiles
│
├── templates/
│   ├── base.html               # Master layout with responsive navbar and footer
│   ├── index.html              # Landing page and overview
│   ├── dashboard.html          # Performance KPI charts and analytics dashboard
│   ├── students.html           # Student management table with search and filters
│   ├── student_form.html       # Student addition/editing form
│   ├── prediction.html         # Real-time AI prediction and recommendations form
│   ├── early_warning.html      # Risk alerts and counselor intervention management
│   ├── reports.html            # Model evaluation, confusion matrices, and Elbow curve
│   └── about.html              # Lab mappings, course details, and ethical considerations
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom styles, badges, and modern typography
│   └── js/
│       ├── main.js             # UI interactions, alerts, and batch predictions
│       └── charts.js           # Dynamic Chart.js dashboard and report charts
│
└── student_performance.db      # SQLite relational database pre-seeded with students
```

---

## 🚀 Installation & Running Guide

### 1. Prerequisites
Ensure **Python 3.10+** (or Python 3.13) is installed on your operating system.

### 2. Install Required Dependencies
Open PowerShell or Terminal in the project root directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-Train the Machine Learning Models
To re-run the full training pipeline (Labs 01, 03, 05, and 10), execute:
```bash
python train_model.py
```
This script will:
- Clean and normalize the dataset.
- Train and evaluate Logistic Regression and Random Forest models.
- Compute the K-Means Elbow Curve and cluster profiles.
- Save trained artifacts into the `models/` directory.

### 4. Run the Web Application
Launch the Flask development server:
```bash
python app.py
```

### 5. Access the System
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📊 Summary of Model Performance

Evaluated on an independent $20\%$ stratified test split ($N = 79$ students):

| Metric | Logistic Regression (Lab 03) | Random Forest (Lab 05) | Selected Role |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **92.41%** | **91.14%** | Both achieve $>90\%$ generalization |
| **Precision** | **88.46%** | **85.19%** | Low false alarm rate |
| **Recall** | **88.46%** | **88.46%** | Captures $88.5\%$ of at-risk students |
| **F1-Score** | **88.46%** | **86.79%** | Harmonized balance |
| **ROC-AUC** | **97.82%** | **96.37%** | Near-perfect class separability |

---

## ⚖️ Academic Integrity & Ethical Use

This project was developed strictly for academic and educational evaluation in accordance with the Dawood University of Engineering & Technology (DUET) syllabus guidelines. Predictions are advisory tools meant to guide supportive academic mentorship and tutoring, not punitive evaluation.
