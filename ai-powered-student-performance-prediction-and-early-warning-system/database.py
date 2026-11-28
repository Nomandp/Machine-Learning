"""
SQLite Database Layer & Model Inference Connectors
Course: Machine Learning (AI-3202) - Complex Computing Activity (Lab 14)
Dawood University of Engineering & Technology (DUET)
"""

import sqlite3
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'student_performance.db')

def get_db_connection():
    """Create a thread-safe connection to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables for students, predictions, and early warning interventions."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Students
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            studytime INTEGER NOT NULL,
            failures INTEGER NOT NULL,
            absences INTEGER NOT NULL,
            g1 REAL NOT NULL,
            g2 REAL NOT NULL,
            schoolsup TEXT NOT NULL DEFAULT 'no',
            famsup TEXT NOT NULL DEFAULT 'no',
            paid TEXT NOT NULL DEFAULT 'no',
            higher TEXT NOT NULL DEFAULT 'yes',
            internet TEXT NOT NULL DEFAULT 'yes',
            medu INTEGER NOT NULL DEFAULT 2,
            fedu INTEGER NOT NULL DEFAULT 2,
            freetime INTEGER NOT NULL DEFAULT 3,
            goout INTEGER NOT NULL DEFAULT 3,
            health INTEGER NOT NULL DEFAULT 3,
            risk_status TEXT DEFAULT 'Pending',
            risk_probability REAL DEFAULT 0.0,
            risk_level TEXT DEFAULT 'Low',
            cluster_id INTEGER DEFAULT -1,
            cluster_label TEXT DEFAULT 'Unassigned',
            recommendations TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table 2: Predictions History
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            model_used TEXT NOT NULL,
            risk_status TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            cluster_id INTEGER,
            cluster_label TEXT,
            recommendations TEXT,
            inputs_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table 3: Early Warning Interventions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            intervention_type TEXT NOT NULL,
            action_plan TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            assigned_counselor TEXT DEFAULT 'Academic Support Cell',
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def generate_recommendations(features_dict, risk_prob, cluster_info):
    """
    Generate customized, actionable early warning recommendations 
    based on student behavioral features, past grades, and absences.
    """
    recs = []
    g1 = float(features_dict.get('g1', features_dict.get('G1', 0)))
    g2 = float(features_dict.get('g2', features_dict.get('G2', 0)))
    absences = int(features_dict.get('absences', 0))
    studytime = int(features_dict.get('studytime', 2))
    failures = int(features_dict.get('failures', 0))
    schoolsup = str(features_dict.get('schoolsup', 'no')).lower()
    paid = str(features_dict.get('paid', 'no')).lower()

    if risk_prob >= 0.5:
        # At Risk interventions
        if absences > 8:
            recs.append(f"Critical Attendance Alert: Student has {absences} recorded absences. Implement mandatory weekly attendance tracking and notify academic coordinator.")
        if g2 < 10 or g1 < 10:
            recs.append(f"Targeted Remedial Support: Prior period scores (G1: {g1}, G2: {g2}) are below the passing benchmark (10/20). Enroll in faculty-led tutorial sessions.")
        if studytime <= 2:
            recs.append("Study Schedule Restructuring: Self-reported study time is less than 5 hours per week. Schedule an academic advisor session to draft a weekly structured study plan.")
        if failures > 0:
            recs.append(f"Past Course Deficit Review: Student has {failures} historical course failure(s). Prioritize core prerequisite topic reinforcement.")
        if schoolsup == 'no':
            recs.append("Enroll in University Remedial Tutoring (Extra School Educational Support).")
        if not recs:
            recs.append("Schedule 1-on-1 counseling with department mentor to identify conceptual hurdles.")
    else:
        # On Track guidance
        recs.append("Maintain Current Study Discipline: Academic momentum is strong. Encourage participation in peer mentoring and advanced problem sets.")
        if studytime < 2:
            recs.append("Encourage gradual increase in self-study time to reinforce concepts before final examinations.")
        if absences > 5:
            recs.append(f"Attendance Watch: Absences ({absences}) should be contained below 5 to prevent borderline grade erosion.")

    return " | ".join(recs)


def predict_student(data_dict, model_choice="random_forest"):
    """
    Run preprocessing, scaling, supervised risk prediction, and unsupervised K-Means clustering.
    Avoids data leakage by ensuring only available historical/demographic features are evaluated.
    """
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    cluster_scaler = joblib.load(os.path.join(MODELS_DIR, 'cluster_scaler.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans.pkl'))

    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Prepare feature vector for supervised models
    # Features: age, sex, studytime, failures, absences, G1, G2, schoolsup, famsup, paid, higher, internet, Medu, Fedu, freetime, goout, health
    sex_val = 1 if str(data_dict.get('sex', 'M')).upper() == 'M' else 0
    schoolsup_val = 1 if str(data_dict.get('schoolsup', 'no')).lower() in ['yes', '1', 'true'] else 0
    famsup_val = 1 if str(data_dict.get('famsup', 'no')).lower() in ['yes', '1', 'true'] else 0
    paid_val = 1 if str(data_dict.get('paid', 'no')).lower() in ['yes', '1', 'true'] else 0
    higher_val = 1 if str(data_dict.get('higher', 'yes')).lower() in ['yes', '1', 'true'] else 0
    internet_val = 1 if str(data_dict.get('internet', 'yes')).lower() in ['yes', '1', 'true'] else 0

    feature_values = [
        int(data_dict.get('age', 17)),
        sex_val,
        int(data_dict.get('studytime', 2)),
        int(data_dict.get('failures', 0)),
        int(data_dict.get('absences', 4)),
        float(data_dict.get('g1', data_dict.get('G1', 11))),
        float(data_dict.get('g2', data_dict.get('G2', 11))),
        schoolsup_val,
        famsup_val,
        paid_val,
        higher_val,
        internet_val,
        int(data_dict.get('medu', data_dict.get('Medu', 2))),
        int(data_dict.get('fedu', data_dict.get('Fedu', 2))),
        int(data_dict.get('freetime', 3)),
        int(data_dict.get('goout', 3)),
        int(data_dict.get('health', 3))
    ]

    X_input = pd.DataFrame([feature_values], columns=metadata['features'])
    X_input_scaled = scaler.transform(X_input)

    # Supervised Predictions
    rf_prob = float(rf_model.predict_proba(X_input_scaled)[0][1])
    lr_prob = float(lr_model.predict_proba(X_input_scaled)[0][1])

    if model_choice == "logistic_regression":
        primary_prob = lr_prob
        selected_model = "Logistic Regression"
    elif model_choice == "ensemble":
        primary_prob = (rf_prob + lr_prob) / 2.0
        selected_model = "Ensemble (RF + LR)"
    else:
        primary_prob = rf_prob
        selected_model = "Random Forest"

    is_at_risk = 1 if primary_prob >= 0.50 else 0
    risk_status = "At Risk" if is_at_risk == 1 else "Not At Risk"

    if primary_prob >= 0.70:
        risk_level = "High"
    elif primary_prob >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Unsupervised K-Means Clustering on ['studytime', 'absences', 'failures', 'G1', 'G2']
    cluster_features = [
        int(data_dict.get('studytime', 2)),
        int(data_dict.get('absences', 4)),
        int(data_dict.get('failures', 0)),
        float(data_dict.get('g1', data_dict.get('G1', 11))),
        float(data_dict.get('g2', data_dict.get('G2', 11)))
    ]
    X_cluster = pd.DataFrame([cluster_features], columns=metadata['cluster_features'])
    X_cluster_scaled = cluster_scaler.transform(X_cluster)
    cluster_id = int(kmeans_model.predict(X_cluster_scaled)[0])

    cluster_info = metadata['kmeans']['clusters'].get(str(cluster_id), {
        "name": f"Cluster {cluster_id}",
        "description": "Behavioral pattern cohort",
        "badge_color": "secondary"
    })

    # Recommendations
    recs = generate_recommendations(data_dict, primary_prob, cluster_info)

    return {
        "model_used": selected_model,
        "is_at_risk": is_at_risk,
        "risk_status": risk_status,
        "risk_probability": round(primary_prob, 4),
        "risk_percentage": round(primary_prob * 100, 1),
        "risk_level": risk_level,
        "rf_probability": round(rf_prob * 100, 1),
        "lr_probability": round(lr_prob * 100, 1),
        "cluster_id": cluster_id,
        "cluster_label": cluster_info['name'],
        "cluster_desc": cluster_info['description'],
        "cluster_badge": cluster_info['badge_color'],
        "recommendations": recs
    }


def seed_initial_data():
    """Populate database with authentic sample students and predictions if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM students')
    count = cursor.fetchone()[0]

    if count > 0:
        conn.close()
        return

    print("Seeding initial student database with diverse academic profiles...")
    # Curated sample of realistic students from DUET 2024 Batch
    sample_students = [
        {"student_id": "DUET-2024-AI-001", "name": "Bilal Ahmed", "age": 18, "sex": "M", "studytime": 4, "failures": 0, "absences": 2, "g1": 16.0, "g2": 17.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 4, "fedu": 4, "freetime": 2, "goout": 2, "health": 5},
        {"student_id": "DUET-2024-AI-002", "name": "Ayesha Khan", "age": 17, "sex": "F", "studytime": 3, "failures": 0, "absences": 4, "g1": 14.0, "g2": 15.0, "schoolsup": "no", "famsup": "yes", "paid": "yes", "higher": "yes", "internet": "yes", "medu": 3, "fedu": 4, "freetime": 3, "goout": 3, "health": 4},
        {"student_id": "DUET-2024-AI-003", "name": "Hamza Tariq", "age": 19, "sex": "M", "studytime": 1, "failures": 2, "absences": 18, "g1": 6.0, "g2": 5.0, "schoolsup": "yes", "famsup": "no", "paid": "no", "higher": "no", "internet": "yes", "medu": 1, "fedu": 1, "freetime": 4, "goout": 5, "health": 2},
        {"student_id": "DUET-2024-AI-004", "name": "Fatima Zahra", "age": 18, "sex": "F", "studytime": 3, "failures": 0, "absences": 6, "g1": 13.0, "g2": 14.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 4, "fedu": 3, "freetime": 3, "goout": 2, "health": 5},
        {"student_id": "DUET-2024-AI-005", "name": "Usman Ali", "age": 19, "sex": "M", "studytime": 2, "failures": 1, "absences": 12, "g1": 8.0, "g2": 7.0, "schoolsup": "no", "famsup": "no", "paid": "no", "higher": "yes", "internet": "no", "medu": 2, "fedu": 2, "freetime": 4, "goout": 4, "health": 3},
        {"student_id": "DUET-2024-AI-006", "name": "Zainab Noor", "age": 17, "sex": "F", "studytime": 4, "failures": 0, "absences": 1, "g1": 17.0, "g2": 18.0, "schoolsup": "no", "famsup": "yes", "paid": "yes", "higher": "yes", "internet": "yes", "medu": 4, "fedu": 4, "freetime": 2, "goout": 1, "health": 4},
        {"student_id": "DUET-2024-AI-007", "name": "Daniyal Farooq", "age": 18, "sex": "M", "studytime": 2, "failures": 0, "absences": 7, "g1": 10.0, "g2": 11.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 3, "fedu": 2, "freetime": 3, "goout": 3, "health": 4},
        {"student_id": "DUET-2024-AI-008", "name": "Maryam Siddiqui", "age": 18, "sex": "F", "studytime": 2, "failures": 0, "absences": 5, "g1": 11.0, "g2": 10.0, "schoolsup": "no", "famsup": "no", "paid": "no", "higher": "yes", "internet": "yes", "medu": 2, "fedu": 3, "freetime": 3, "goout": 3, "health": 3},
        {"student_id": "DUET-2024-AI-009", "name": "Shahzaib Raza", "age": 19, "sex": "M", "studytime": 1, "failures": 3, "absences": 22, "g1": 5.0, "g2": 4.0, "schoolsup": "yes", "famsup": "no", "paid": "no", "higher": "no", "internet": "yes", "medu": 1, "fedu": 2, "freetime": 5, "goout": 5, "health": 2},
        {"student_id": "DUET-2024-AI-010", "name": "Hira Batool", "age": 18, "sex": "F", "studytime": 3, "failures": 0, "absences": 3, "g1": 15.0, "g2": 16.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 3, "fedu": 3, "freetime": 2, "goout": 2, "health": 5},
        {"student_id": "DUET-2024-AI-011", "name": "Mohammad Saad", "age": 18, "sex": "M", "studytime": 1, "failures": 1, "absences": 14, "g1": 7.0, "g2": 6.0, "schoolsup": "no", "famsup": "no", "paid": "no", "higher": "yes", "internet": "yes", "medu": 2, "fedu": 1, "freetime": 4, "goout": 4, "health": 3},
        {"student_id": "DUET-2024-AI-012", "name": "Laiba Rehman", "age": 17, "sex": "F", "studytime": 4, "failures": 0, "absences": 0, "g1": 18.0, "g2": 19.0, "schoolsup": "no", "famsup": "yes", "paid": "yes", "higher": "yes", "internet": "yes", "medu": 4, "fedu": 4, "freetime": 2, "goout": 1, "health": 5},
        {"student_id": "DUET-2024-AI-013", "name": "Omer Khalid", "age": 19, "sex": "M", "studytime": 2, "failures": 0, "absences": 8, "g1": 10.0, "g2": 9.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 2, "fedu": 2, "freetime": 3, "goout": 4, "health": 4},
        {"student_id": "DUET-2024-AI-014", "name": "Anum Javed", "age": 18, "sex": "F", "studytime": 2, "failures": 1, "absences": 10, "g1": 8.0, "g2": 8.0, "schoolsup": "yes", "famsup": "no", "paid": "no", "higher": "yes", "internet": "yes", "medu": 2, "fedu": 3, "freetime": 3, "goout": 3, "health": 3},
        {"student_id": "DUET-2024-AI-015", "name": "Hassan Mehmood", "age": 18, "sex": "M", "studytime": 3, "failures": 0, "absences": 3, "g1": 13.0, "g2": 14.0, "schoolsup": "no", "famsup": "yes", "paid": "no", "higher": "yes", "internet": "yes", "medu": 3, "fedu": 4, "freetime": 3, "goout": 2, "health": 4}
    ]

    for st in sample_students:
        pred = predict_student(st, model_choice="random_forest")
        cursor.execute('''
            INSERT INTO students (
                student_id, name, age, sex, studytime, failures, absences,
                g1, g2, schoolsup, famsup, paid, higher, internet,
                medu, fedu, freetime, goout, health,
                risk_status, risk_probability, risk_level,
                cluster_id, cluster_label, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            st['student_id'], st['name'], st['age'], st['sex'],
            st['studytime'], st['failures'], st['absences'],
            st['g1'], st['g2'], st['schoolsup'], st['famsup'],
            st['paid'], st['higher'], st['internet'],
            st['medu'], st['fedu'], st['freetime'], st['goout'], st['health'],
            pred['risk_status'], pred['risk_probability'], pred['risk_level'],
            pred['cluster_id'], pred['cluster_label'], pred['recommendations']
        ))

        # Log prediction
        cursor.execute('''
            INSERT INTO predictions (
                student_id, student_name, model_used, risk_status,
                risk_probability, risk_level, cluster_id, cluster_label,
                recommendations, inputs_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            st['student_id'], st['name'], pred['model_used'],
            pred['risk_status'], pred['risk_probability'], pred['risk_level'],
            pred['cluster_id'], pred['cluster_label'], pred['recommendations'],
            json.dumps(st)
        ))

        # If student is At Risk, create an actionable early warning intervention record
        if pred['risk_status'] == 'At Risk':
            priority_type = "Tutoring & Remedial" if st['failures'] > 0 else "Attendance Counseling"
            cursor.execute('''
                INSERT INTO interventions (
                    student_id, student_name, risk_level, intervention_type,
                    action_plan, status, assigned_counselor, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                st['student_id'], st['name'], pred['risk_level'], priority_type,
                pred['recommendations'], "In Progress" if pred['risk_level'] == 'High' else "Pending",
                "Prof. Hamza Farooqui (Advisor)", "Auto-flagged by Early Warning ML Classifier."
            ))

    conn.commit()
    conn.close()
    print("Database seeding completed successfully.")


def get_dashboard_metrics():
    """Retrieve summarized statistics and aggregations for dashboard charts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    total_students = cursor.execute('SELECT COUNT(*) FROM students').fetchone()[0]
    at_risk_count = cursor.execute("SELECT COUNT(*) FROM students WHERE risk_status = 'At Risk'").fetchone()[0]
    not_at_risk_count = cursor.execute("SELECT COUNT(*) FROM students WHERE risk_status = 'Not At Risk'").fetchone()[0]
    pending_interventions = cursor.execute("SELECT COUNT(*) FROM interventions WHERE status != 'Completed'").fetchone()[0]
    total_predictions = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # Averages
    avg_grades = cursor.execute('SELECT AVG(g1), AVG(g2), AVG(absences), AVG(studytime) FROM students').fetchone()
    avg_g1 = round(avg_grades[0] or 0, 1)
    avg_g2 = round(avg_grades[1] or 0, 1)
    avg_absences = round(avg_grades[2] or 0, 1)
    avg_studytime = round(avg_grades[3] or 0, 1)

    # Risk level breakdown
    high_risk_count = cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'High'").fetchone()[0]
    moderate_risk_count = cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Moderate'").fetchone()[0]
    low_risk_count = cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Low'").fetchone()[0]

    # Cluster distribution
    cluster_counts = cursor.execute('''
        SELECT cluster_label, COUNT(*) as count 
        FROM students 
        WHERE cluster_label != 'Unassigned' 
        GROUP BY cluster_label
    ''').fetchall()
    cluster_dist = {row['cluster_label']: row['count'] for row in cluster_counts}

    # Student list for charts
    students_data = cursor.execute('''
        SELECT student_id, name, g1, g2, absences, studytime, risk_status, risk_probability 
        FROM students 
        ORDER BY id DESC LIMIT 50
    ''').fetchall()
    student_chart_list = [dict(row) for row in students_data]

    conn.close()

    return {
        "total_students": total_students,
        "at_risk_count": at_risk_count,
        "not_at_risk_count": not_at_risk_count,
        "pending_interventions": pending_interventions,
        "total_predictions": total_predictions,
        "avg_g1": avg_g1,
        "avg_g2": avg_g2,
        "avg_absences": avg_absences,
        "avg_studytime": avg_studytime,
        "high_risk_count": high_risk_count,
        "moderate_risk_count": moderate_risk_count,
        "low_risk_count": low_risk_count,
        "cluster_dist": cluster_dist,
        "students": student_chart_list
    }


def get_all_students(search_query=None, risk_filter=None, sort_by="id", order="DESC"):
    """Query student records with optional search and risk filters."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM students WHERE 1=1"
    params = []

    if search_query:
        query += " AND (name LIKE ? OR student_id LIKE ?)"
        q = f"%{search_query.strip()}%"
        params.extend([q, q])

    if risk_filter and risk_filter != 'All':
        query += " AND risk_status = ?"
        params.append(risk_filter)

    valid_sorts = ["id", "name", "student_id", "risk_probability", "g1", "g2", "absences", "studytime"]
    sort_column = sort_by if sort_by in valid_sorts else "id"
    sort_direction = "ASC" if order.upper() == "ASC" else "DESC"

    query += f" ORDER BY {sort_column} {sort_direction}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    students = [dict(row) for row in rows]
    conn.close()
    return students


def get_student_by_id(pk_id):
    """Fetch single student by internal primary key or student_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id = ? OR student_id = ?", (pk_id, pk_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def add_student_record(data_dict, model_choice="random_forest"):
    """Add a new student, automatically compute ML prediction, and create intervention if At Risk."""
    pred = predict_student(data_dict, model_choice=model_choice)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO students (
            student_id, name, age, sex, studytime, failures, absences,
            g1, g2, schoolsup, famsup, paid, higher, internet,
            medu, fedu, freetime, goout, health,
            risk_status, risk_probability, risk_level,
            cluster_id, cluster_label, recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_dict['student_id'].strip(),
        data_dict['name'].strip(),
        int(data_dict['age']),
        data_dict['sex'].upper(),
        int(data_dict['studytime']),
        int(data_dict['failures']),
        int(data_dict['absences']),
        float(data_dict['g1']),
        float(data_dict['g2']),
        data_dict.get('schoolsup', 'no'),
        data_dict.get('famsup', 'no'),
        data_dict.get('paid', 'no'),
        data_dict.get('higher', 'yes'),
        data_dict.get('internet', 'yes'),
        int(data_dict.get('medu', 2)),
        int(data_dict.get('fedu', 2)),
        int(data_dict.get('freetime', 3)),
        int(data_dict.get('goout', 3)),
        int(data_dict.get('health', 3)),
        pred['risk_status'],
        pred['risk_probability'],
        pred['risk_level'],
        pred['cluster_id'],
        pred['cluster_label'],
        pred['recommendations']
    ))
    new_id = cursor.lastrowid

    # Log prediction
    cursor.execute('''
        INSERT INTO predictions (
            student_id, student_name, model_used, risk_status,
            risk_probability, risk_level, cluster_id, cluster_label,
            recommendations, inputs_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_dict['student_id'].strip(),
        data_dict['name'].strip(),
        pred['model_used'],
        pred['risk_status'],
        pred['risk_probability'],
        pred['risk_level'],
        pred['cluster_id'],
        pred['cluster_label'],
        pred['recommendations'],
        json.dumps(data_dict)
    ))

    # Auto-create intervention if At Risk
    if pred['risk_status'] == 'At Risk':
        priority_type = "Tutoring & Remedial" if int(data_dict['failures']) > 0 else "Attendance Counseling"
        cursor.execute('''
            INSERT INTO interventions (
                student_id, student_name, risk_level, intervention_type,
                action_plan, status, assigned_counselor, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_dict['student_id'].strip(),
            data_dict['name'].strip(),
            pred['risk_level'],
            priority_type,
            pred['recommendations'],
            "Pending",
            "Academic Advisor",
            "Generated on student enrollment."
        ))

    conn.commit()
    conn.close()
    return new_id, pred


def update_student_record(pk_id, data_dict, model_choice="random_forest"):
    """Update existing student, re-evaluate ML risk score, and update interventions."""
    pred = predict_student(data_dict, model_choice=model_choice)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE students SET
            student_id = ?, name = ?, age = ?, sex = ?, studytime = ?, failures = ?, absences = ?,
            g1 = ?, g2 = ?, schoolsup = ?, famsup = ?, paid = ?, higher = ?, internet = ?,
            medu = ?, fedu = ?, freetime = ?, goout = ?, health = ?,
            risk_status = ?, risk_probability = ?, risk_level = ?,
            cluster_id = ?, cluster_label = ?, recommendations = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (
        data_dict['student_id'].strip(),
        data_dict['name'].strip(),
        int(data_dict['age']),
        data_dict['sex'].upper(),
        int(data_dict['studytime']),
        int(data_dict['failures']),
        int(data_dict['absences']),
        float(data_dict['g1']),
        float(data_dict['g2']),
        data_dict.get('schoolsup', 'no'),
        data_dict.get('famsup', 'no'),
        data_dict.get('paid', 'no'),
        data_dict.get('higher', 'yes'),
        data_dict.get('internet', 'yes'),
        int(data_dict.get('medu', 2)),
        int(data_dict.get('fedu', 2)),
        int(data_dict.get('freetime', 3)),
        int(data_dict.get('goout', 3)),
        int(data_dict.get('health', 3)),
        pred['risk_status'],
        pred['risk_probability'],
        pred['risk_level'],
        pred['cluster_id'],
        pred['cluster_label'],
        pred['recommendations'],
        pk_id
    ))

    # Log new prediction entry
    cursor.execute('''
        INSERT INTO predictions (
            student_id, student_name, model_used, risk_status,
            risk_probability, risk_level, cluster_id, cluster_label,
            recommendations, inputs_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_dict['student_id'].strip(),
        data_dict['name'].strip(),
        pred['model_used'],
        pred['risk_status'],
        pred['risk_probability'],
        pred['risk_level'],
        pred['cluster_id'],
        pred['cluster_label'],
        pred['recommendations'],
        json.dumps(data_dict)
    ))

    conn.commit()
    conn.close()
    return pred


def delete_student_record(pk_id):
    """Delete a student and associated interventions from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    student = get_student_by_id(pk_id)
    if student:
        cursor.execute("DELETE FROM interventions WHERE student_id = ?", (student['student_id'],))
        cursor.execute("DELETE FROM predictions WHERE student_id = ?", (student['student_id'],))
        cursor.execute("DELETE FROM students WHERE id = ?", (pk_id,))
        conn.commit()
    conn.close()
    return bool(student)


def get_early_warning_students(status_filter=None):
    """Retrieve students identified as At Risk along with their intervention plans."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = '''
        SELECT s.*, 
               i.id as intervention_id,
               i.intervention_type,
               i.status as intervention_status,
               i.assigned_counselor,
               i.action_plan,
               i.notes as intervention_notes
        FROM students s
        LEFT JOIN interventions i ON s.student_id = i.student_id
        WHERE s.risk_status = 'At Risk'
    '''
    params = []
    if status_filter and status_filter != 'All':
        query += " AND i.status = ?"
        params.append(status_filter)

    query += " ORDER BY s.risk_probability DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_intervention_status(intervention_id, new_status, counselor=None, notes=None):
    """Update status and progress notes on an early warning intervention."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE interventions SET
            status = ?,
            assigned_counselor = COALESCE(?, assigned_counselor),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (new_status, counselor, notes, intervention_id))
    conn.commit()
    conn.close()


def create_intervention_record(student_id, intervention_type, action_plan, counselor="Advisor", notes=""):
    """Create a manual intervention task for an at-risk student."""
    conn = get_db_connection()
    cursor = conn.cursor()
    student = get_student_by_id(student_id)
    student_name = student['name'] if student else "Student"
    risk_level = student['risk_level'] if student else "Moderate"

    cursor.execute('''
        INSERT INTO interventions (
            student_id, student_name, risk_level, intervention_type,
            action_plan, status, assigned_counselor, notes
        ) VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?)
    ''', (student_id, student_name, risk_level, intervention_type, action_plan, counselor, notes))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_prediction_history(limit=50):
    """Retrieve audit history of model predictions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def batch_repredict_all():
    """Batch re-evaluate risk scores for all students using the latest trained models."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    students = [dict(row) for row in cursor.fetchall()]

    updated_count = 0
    for st in students:
        pred = predict_student(st, model_choice="random_forest")
        cursor.execute('''
            UPDATE students SET
                risk_status = ?,
                risk_probability = ?,
                risk_level = ?,
                cluster_id = ?,
                cluster_label = ?,
                recommendations = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            pred['risk_status'],
            pred['risk_probability'],
            pred['risk_level'],
            pred['cluster_id'],
            pred['cluster_label'],
            pred['recommendations'],
            st['id']
        ))
        updated_count += 1

    conn.commit()
    conn.close()
    return updated_count
