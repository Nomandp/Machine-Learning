"""
Model Training & Evaluation Pipeline
Course: Machine Learning (AI-3202) - Lab 14: Complex Computing Activity
Dawood University of Engineering & Technology (DUET), Batch 2024

Integrated Lab Concepts:
- Lab 01: Data Preprocessing, Categorical Encoding, and Feature Scaling (StandardScaler)
- Lab 03: Logistic Regression for Binary Classification (At Risk vs. Not At Risk)
- Lab 05: Random Forest Classifier (Ensemble Bagging and Feature Importance)
- Lab 10: K-Means Clustering (Unsupervised Academic Pattern Segmentation and Elbow Method)
- Avoidance of Data Leakage: Final grade (G3) is strictly excluded from features.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Absolute paths to ensure scripts work from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'student_performance.db')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Feature definitions for consistent modeling and inference
CORE_FEATURES = [
    'age', 'sex', 'studytime', 'failures', 'absences', 'G1', 'G2',
    'schoolsup', 'famsup', 'paid', 'higher', 'internet',
    'Medu', 'Fedu', 'freetime', 'goout', 'health'
]

CLUSTER_FEATURES = ['studytime', 'absences', 'failures', 'G1', 'G2']

BINARY_MAPS = {
    'sex': {'F': 0, 'M': 1, 0: 0, 1: 1},
    'schoolsup': {'no': 0, 'yes': 1, 0: 0, 1: 1},
    'famsup': {'no': 0, 'yes': 1, 0: 0, 1: 1},
    'paid': {'no': 0, 'yes': 1, 0: 0, 1: 1},
    'higher': {'no': 0, 'yes': 1, 0: 0, 1: 1},
    'internet': {'no': 0, 'yes': 1, 0: 0, 1: 1}
}


def preprocess_data(csv_path=None):
    """Lab 01: Load and preprocess raw dataset, handling encodings and preventing data leakage."""
    print("=" * 60)
    print("LAB 01: DATA PREPROCESSING & CLEANING")
    print("=" * 60)

    if csv_path is None:
        csv_path = DATASET_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded raw dataset with shape: {df.shape}")

    # Check missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total missing values found: {total_missing}")

    # Create binary target: At Risk (1) if G3 < 10, else Not At Risk (0)
    # 10 is the official passing threshold out of 20 in Portuguese secondary education
    y = (df['G3'] < 10).astype(int)
    print("Target distribution (At Risk = 1, Not At Risk = 0):")
    print(y.value_counts(normalize=True).round(3).to_dict())

    # Preprocess categorical features
    df_clean = df.copy()
    for col, mapping in BINARY_MAPS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping).fillna(0).astype(int)

    # STRICT DATA LEAKAGE PREVENTION: Drop G3 from features
    X = df_clean[CORE_FEATURES].copy()
    print(f"Features selected ({len(CORE_FEATURES)}): {CORE_FEATURES}")
    print("Crucial: Target G3 is completely excluded from input feature space.")

    return df, X, y


def train_models():
    """Train Logistic Regression, Random Forest, and K-Means models."""
    os.makedirs('models', exist_ok=True)
    df_raw, X, y = preprocess_data()

    # Train / Test split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Feature Scaling (Lab 01)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # LAB 03: LOGISTIC REGRESSION CLASSIFIER
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LAB 03: LOGISTIC REGRESSION (SUPERVISED CLASSIFICATION)")
    print("=" * 60)
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train_scaled, y_train)

    lr_train_preds = lr_model.predict(X_train_scaled)
    lr_test_preds = lr_model.predict(X_test_scaled)
    lr_test_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_acc = float(accuracy_score(y_test, lr_test_preds))
    lr_prec = float(precision_score(y_test, lr_test_preds, zero_division=0))
    lr_rec = float(recall_score(y_test, lr_test_preds, zero_division=0))
    lr_f1 = float(f1_score(y_test, lr_test_preds, zero_division=0))
    lr_auc = float(roc_auc_score(y_test, lr_test_probs))
    lr_cm = confusion_matrix(y_test, lr_test_preds).tolist()

    print(f"LR Accuracy : {lr_acc:.4f}")
    print(f"LR Precision: {lr_prec:.4f}")
    print(f"LR Recall   : {lr_rec:.4f}")
    print(f"LR F1-Score : {lr_f1:.4f}")
    print(f"LR ROC-AUC  : {lr_auc:.4f}")
    print("LR Confusion Matrix:")
    print(lr_cm)

    # -------------------------------------------------------------
    # LAB 05: RANDOM FOREST CLASSIFIER
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LAB 05: RANDOM FOREST (MAIN ENSEMBLE CLASSIFIER)")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        random_state=42
    )
    # Random Forest works on unscaled or scaled; training on scaled keeps pipeline unified
    rf_model.fit(X_train_scaled, y_train)

    rf_train_preds = rf_model.predict(X_train_scaled)
    rf_test_preds = rf_model.predict(X_test_scaled)
    rf_test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

    rf_acc = float(accuracy_score(y_test, rf_test_preds))
    rf_prec = float(precision_score(y_test, rf_test_preds, zero_division=0))
    rf_rec = float(recall_score(y_test, rf_test_preds, zero_division=0))
    rf_f1 = float(f1_score(y_test, rf_test_preds, zero_division=0))
    rf_auc = float(roc_auc_score(y_test, rf_test_probs))
    rf_cm = confusion_matrix(y_test, rf_test_preds).tolist()

    print(f"RF Accuracy : {rf_acc:.4f}")
    print(f"RF Precision: {rf_prec:.4f}")
    print(f"RF Recall   : {rf_rec:.4f}")
    print(f"RF F1-Score : {rf_f1:.4f}")
    print(f"RF ROC-AUC  : {rf_auc:.4f}")
    print("RF Confusion Matrix:")
    print(rf_cm)

    # Feature Importance analysis (Random Forest)
    importances = rf_model.feature_importances_
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(CORE_FEATURES, importances), key=lambda x: x[1], reverse=True)
    ]
    print("\nTop 5 Most Important Features in Predicting Risk:")
    for item in feature_importance_list[:5]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")

    # -------------------------------------------------------------
    # LAB 10: K-MEANS CLUSTERING (UNSUPERVISED BEHAVIORAL PROFILES)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LAB 10: K-MEANS CLUSTERING (STUDENT PATTERN SEGMENTATION)")
    print("=" * 60)

    X_cluster_raw = df_raw[CLUSTER_FEATURES].copy()
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster_raw)

    # Elbow Method evaluation (k=1 to 8)
    elbow_inertias = []
    k_range = list(range(1, 9))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster_scaled)
        elbow_inertias.append(round(float(km.inertia_), 2))

    print(f"Elbow inertias (k=1..8): {elbow_inertias}")

    # Optimal K = 3
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_cluster_scaled)
    df_raw['cluster'] = cluster_labels

    # Compute descriptive statistics for each cluster
    cluster_stats = {}
    cluster_summary = df_raw.groupby('cluster')[CLUSTER_FEATURES + ['G3']].mean()

    # Assign meaningful academic personas based on G3 and failure rates
    for cid in range(3):
        row = cluster_summary.loc[cid]
        g3_avg = float(row['G3'])
        failures_avg = float(row['failures'])
        absences_avg = float(row['absences'])
        study_avg = float(row['studytime'])

        if g3_avg >= 13.0:
            name = "High Achiever & Dedicated"
            desc = "High exam scores, low past failures, disciplined study schedule."
            color = "success"
        elif failures_avg > 1.0 or g3_avg < 8.0:
            name = "Academic Risk / High Needs"
            desc = "Multiple past failures, lower exam grades, urgent academic intervention needed."
            color = "danger"
        else:
            name = "Moderate Performer / Watchlist"
            desc = "Average grades, moderate absences, vulnerable to falling behind without guidance."
            color = "warning"

        count = int((df_raw['cluster'] == cid).sum())
        percentage = round((count / len(df_raw)) * 100, 1)

        cluster_stats[str(cid)] = {
            "name": name,
            "description": desc,
            "badge_color": color,
            "count": count,
            "percentage": percentage,
            "avg_g1": round(float(row['G1']), 2),
            "avg_g2": round(float(row['G2']), 2),
            "avg_g3": round(float(row['G3']), 2),
            "avg_studytime": round(study_avg, 2),
            "avg_absences": round(absences_avg, 2),
            "avg_failures": round(failures_avg, 2)
        }

    print("\nCluster Profiles Identified:")
    for cid, stats in cluster_stats.items():
        print(f"  Cluster {cid} ({stats['name']}): {stats['count']} students ({stats['percentage']}%) - Avg G3: {stats['avg_g3']}")

    # -------------------------------------------------------------
    # SERIALIZATION OF MODELS AND METRICS METADATA
    # -------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\nSaving trained models and preprocessors to {MODELS_DIR}...")
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(cluster_scaler, os.path.join(MODELS_DIR, 'cluster_scaler.pkl'))

    metadata = {
        "dataset_name": "UCI Student Performance Dataset (Math)",
        "total_samples": len(df_raw),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": CORE_FEATURES,
        "cluster_features": CLUSTER_FEATURES,
        "logistic_regression": {
            "accuracy": lr_acc,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1_score": lr_f1,
            "roc_auc": lr_auc,
            "confusion_matrix": lr_cm,
            "report": classification_report(y_test, lr_test_preds, output_dict=True)
        },
        "random_forest": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1_score": rf_f1,
            "roc_auc": rf_auc,
            "confusion_matrix": rf_cm,
            "report": classification_report(y_test, rf_test_preds, output_dict=True),
            "feature_importance": feature_importance_list
        },
        "kmeans": {
            "k": 3,
            "elbow_k": k_range,
            "elbow_inertias": elbow_inertias,
            "clusters": cluster_stats
        }
    }

    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("All models and metadata successfully generated and saved to models/.")
    print("=" * 60)
    return metadata


if __name__ == '__main__':
    train_models()
