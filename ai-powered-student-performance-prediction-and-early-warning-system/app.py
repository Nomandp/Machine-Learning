"""
Flask Backend Application
AI-Powered Student Performance Prediction and Early Warning System


"""

import os
import io
import csv
import json
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response
)

# Absolute paths to ensure scripts work from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'student_performance.db')

import database

app = Flask(__name__)
app.secret_key = 'duet_machine_learning_2024_secret_key'

# Ensure database tables exist and are seeded
database.init_db()
database.seed_initial_data()


# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route('/')
def index():
    """Home Page: Project summary, highlights, and quick shortcuts."""
    stats = database.get_dashboard_metrics()
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    """Dashboard Page: Key performance indicators, metrics, and Chart.js visualizations."""
    stats = database.get_dashboard_metrics()
    recent_warnings = database.get_early_warning_students()[:5]
    return render_template('dashboard.html', stats=stats, recent_warnings=recent_warnings)


@app.route('/students')
def students():
    """Student Management: View, search, and filter student records."""
    search_query = request.args.get('search', '').strip()
    risk_filter = request.args.get('risk', 'All')
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'DESC')

    student_list = database.get_all_students(
        search_query=search_query,
        risk_filter=risk_filter,
        sort_by=sort_by,
        order=order
    )
    return render_template(
        'students.html',
        students=student_list,
        search_query=search_query,
        risk_filter=risk_filter,
        sort_by=sort_by,
        order=order
    )


@app.route('/students/add', methods=['GET', 'POST'])
def add_student():
    """Add a new student record and automatically predict risk."""
    if request.method == 'POST':
        try:
            form_data = {
                'student_id': request.form.get('student_id', '').strip(),
                'name': request.form.get('name', '').strip(),
                'age': int(request.form.get('age', 18)),
                'sex': request.form.get('sex', 'M'),
                'studytime': int(request.form.get('studytime', 2)),
                'failures': int(request.form.get('failures', 0)),
                'absences': int(request.form.get('absences', 0)),
                'g1': float(request.form.get('g1', 10.0)),
                'g2': float(request.form.get('g2', 10.0)),
                'schoolsup': request.form.get('schoolsup', 'no'),
                'famsup': request.form.get('famsup', 'no'),
                'paid': request.form.get('paid', 'no'),
                'higher': request.form.get('higher', 'yes'),
                'internet': request.form.get('internet', 'yes'),
                'medu': int(request.form.get('medu', 2)),
                'fedu': int(request.form.get('fedu', 2)),
                'freetime': int(request.form.get('freetime', 3)),
                'goout': int(request.form.get('goout', 3)),
                'health': int(request.form.get('health', 3))
            }

            model_choice = request.form.get('model_choice', 'random_forest')
            new_id, pred = database.add_student_record(form_data, model_choice=model_choice)
            flash(f"Student '{form_data['name']}' added successfully! AI predicted status: {pred['risk_status']} ({pred['risk_percentage']}%)", "success")
            return redirect(url_for('students'))
        except Exception as e:
            flash(f"Error adding student: {str(e)}", "danger")

    return render_template('student_form.html', student=None, action_title="Add New Student")


@app.route('/students/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    """Edit existing student record and re-evaluate AI risk prediction."""
    student = database.get_student_by_id(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for('students'))

    if request.method == 'POST':
        try:
            form_data = {
                'student_id': request.form.get('student_id', '').strip(),
                'name': request.form.get('name', '').strip(),
                'age': int(request.form.get('age', 18)),
                'sex': request.form.get('sex', 'M'),
                'studytime': int(request.form.get('studytime', 2)),
                'failures': int(request.form.get('failures', 0)),
                'absences': int(request.form.get('absences', 0)),
                'g1': float(request.form.get('g1', 10.0)),
                'g2': float(request.form.get('g2', 10.0)),
                'schoolsup': request.form.get('schoolsup', 'no'),
                'famsup': request.form.get('famsup', 'no'),
                'paid': request.form.get('paid', 'no'),
                'higher': request.form.get('higher', 'yes'),
                'internet': request.form.get('internet', 'yes'),
                'medu': int(request.form.get('medu', 2)),
                'fedu': int(request.form.get('fedu', 2)),
                'freetime': int(request.form.get('freetime', 3)),
                'goout': int(request.form.get('goout', 3)),
                'health': int(request.form.get('health', 3))
            }

            model_choice = request.form.get('model_choice', 'random_forest')
            pred = database.update_student_record(student_id, form_data, model_choice=model_choice)
            flash(f"Student '{form_data['name']}' updated! Current Status: {pred['risk_status']} ({pred['risk_percentage']}%)", "success")
            return redirect(url_for('students'))
        except Exception as e:
            flash(f"Error updating student: {str(e)}", "danger")

    return render_template('student_form.html', student=student, action_title=f"Edit Student: {student['name']}")


@app.route('/students/delete/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    """Delete student record."""
    success = database.delete_student_record(student_id)
    if success:
        flash("Student record deleted successfully.", "success")
    else:
        flash("Failed to delete student record.", "danger")
    return redirect(url_for('students'))


@app.route('/students/export-csv')
def export_students_csv():
    """Export student database records as CSV file."""
    students_data = database.get_all_students()
    if not students_data:
        flash("No student records to export.", "warning")
        return redirect(url_for('students'))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(students_data[0].keys()))
    writer.writeheader()
    for row in students_data:
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=student_performance_records.csv"}
    )


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """
    AI Prediction Page:
    Allows user to input academic & behavioral variables, choose the ML algorithm,
    and inspect real-time risk classification, probability, K-Means clustering persona,
    and personalized academic recommendations.
    """
    prediction_result = None
    input_data = None

    if request.method == 'POST':
        try:
            input_data = {
                'student_id': request.form.get('student_id', 'DUET-TEMP-PRED').strip(),
                'name': request.form.get('name', 'Anonymous Student').strip(),
                'age': int(request.form.get('age', 18)),
                'sex': request.form.get('sex', 'M'),
                'studytime': int(request.form.get('studytime', 2)),
                'failures': int(request.form.get('failures', 0)),
                'absences': int(request.form.get('absences', 4)),
                'g1': float(request.form.get('g1', 11.0)),
                'g2': float(request.form.get('g2', 11.0)),
                'schoolsup': request.form.get('schoolsup', 'no'),
                'famsup': request.form.get('famsup', 'no'),
                'paid': request.form.get('paid', 'no'),
                'higher': request.form.get('higher', 'yes'),
                'internet': request.form.get('internet', 'yes'),
                'medu': int(request.form.get('medu', 2)),
                'fedu': int(request.form.get('fedu', 2)),
                'freetime': int(request.form.get('freetime', 3)),
                'goout': int(request.form.get('goout', 3)),
                'health': int(request.form.get('health', 3))
            }
            model_choice = request.form.get('model_choice', 'random_forest')
            save_to_db = request.form.get('save_to_db') == 'yes'

            prediction_result = database.predict_student(input_data, model_choice=model_choice)

            if save_to_db and input_data['student_id'] != 'DUET-TEMP-PRED':
                # Check if exists or insert
                existing = database.get_student_by_id(input_data['student_id'])
                if existing:
                    database.update_student_record(existing['id'], input_data, model_choice=model_choice)
                    flash(f"Prediction complete and updated student {input_data['student_id']}.", "success")
                else:
                    database.add_student_record(input_data, model_choice=model_choice)
                    flash(f"Prediction complete and saved new student {input_data['student_id']}.", "success")
            else:
                # Log audit record into predictions history
                conn = database.get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (
                        student_id, student_name, model_used, risk_status,
                        risk_probability, risk_level, cluster_id, cluster_label,
                        recommendations, inputs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    input_data['student_id'],
                    input_data['name'],
                    prediction_result['model_used'],
                    prediction_result['risk_status'],
                    prediction_result['risk_probability'],
                    prediction_result['risk_level'],
                    prediction_result['cluster_id'],
                    prediction_result['cluster_label'],
                    prediction_result['recommendations'],
                    json.dumps(input_data)
                ))
                conn.commit()
                conn.close()

        except Exception as e:
            flash(f"Error during AI inference: {str(e)}", "danger")

    recent_predictions = database.get_prediction_history(limit=5)
    return render_template(
        'prediction.html',
        result=prediction_result,
        input_data=input_data,
        recent_predictions=recent_predictions
    )


@app.route('/early-warning')
def early_warning():
    """Early Warning System: Highlights high and moderate risk students needing intervention."""
    status_filter = request.args.get('status', 'All')
    students_at_risk = database.get_early_warning_students(status_filter=status_filter)
    return render_template(
        'early_warning.html',
        students=students_at_risk,
        status_filter=status_filter
    )


@app.route('/interventions/update', methods=['POST'])
def update_intervention():
    """Update status, counselor note, or progress on an intervention task."""
    try:
        intervention_id = int(request.form.get('intervention_id'))
        new_status = request.form.get('status', 'In Progress')
        counselor = request.form.get('assigned_counselor', 'Academic Advisor')
        notes = request.form.get('notes', '')

        database.update_intervention_status(
            intervention_id=intervention_id,
            new_status=new_status,
            counselor=counselor,
            notes=notes
        )
        flash("Intervention status updated successfully.", "success")
    except Exception as e:
        flash(f"Error updating intervention: {str(e)}", "danger")

    return redirect(url_for('early_warning'))


@app.route('/reports')
def reports():
    """Download Reports Page: Student data charts and CSV export options."""
    stats = database.get_dashboard_metrics()
    return render_template('reports.html', stats=stats)


@app.route('/reports/export-atrisk-csv')
def export_atrisk_csv():
    """Export only At-Risk students as CSV."""
    students_data = database.get_all_students(risk_filter='At Risk')
    if not students_data:
        flash("No at-risk student records to export.", "warning")
        return redirect(url_for('reports'))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(students_data[0].keys()))
    writer.writeheader()
    for row in students_data:
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=at_risk_students.csv"}
    )


@app.route('/reports/export-predictions-csv')
def export_predictions_csv():
    """Export full prediction history as CSV."""
    predictions = database.get_prediction_history(limit=1000)
    if not predictions:
        flash("No prediction history to export.", "warning")
        return redirect(url_for('reports'))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(predictions[0].keys()))
    writer.writeheader()
    for row in predictions:
        writer.writerow(row)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"}
    )


@app.route('/about')
def about():
    """About Page: Project overview, lab mapping (Labs 01, 03, 05, 10, 14), technologies, and team info."""
    return render_template('about.html')


# -------------------------------------------------------------
# REST API Endpoints (For Asynchronous Charts and Integrations)
# -------------------------------------------------------------

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Returns dynamic stats for asynchronous Chart.js updates."""
    stats = database.get_dashboard_metrics()
    return jsonify(stats)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API to evaluate student risk and return probability and recommendations."""
    try:
        data = request.get_json(force=True)
        model_choice = data.get('model_choice', 'random_forest')
        result = database.predict_student(data, model_choice=model_choice)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/batch-repredict', methods=['POST'])
def api_batch_repredict():
    """Batch re-predict for all registered students in the database."""
    try:
        updated = database.batch_repredict_all()
        return jsonify({"success": True, "updated_count": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # Start the Flask web application
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 AI Student Performance System running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
