"""
AI Email Classifier - Flask Web Application
A student-friendly mini-project for intelligent email categorization using NLP & Naive Bayes.
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

from src.data_loader import get_dataset_summary, load_and_validate_data
from src.train_model import get_or_train_model, train_and_evaluate
from src.predict import classify_email

# Initialize Flask application
app = Flask(__name__)
# Secret key used for session cookies (for prediction history storage)
app.secret_key = os.environ.get("SECRET_KEY", "student-ai-email-classifier-secret-key-2026")

# Global reference for model, vectorizer, and evaluation metrics
# Loaded once at startup so inference is instantaneous
classifier = None
vectorizer = None
model_metrics = None


def initialize_app():
    """Initializes and preloads the ML model, vectorizer, and evaluation metrics."""
    global classifier, vectorizer, model_metrics
    try:
        classifier, vectorizer, model_metrics = get_or_train_model()
        print(">> [INFO] ML Classifier and TF-IDF Vectorizer initialized successfully.")
        print(f">> [INFO] Held-out Test Set Accuracy: {model_metrics.get('test_accuracy')}%")
    except Exception as e:
        print(f">> [ERROR] Failed to initialize model: {e}")
        raise e


# Run initialization
initialize_app()


@app.route("/", methods=["GET"])
def index():
    """
    Home Dashboard Route.
    Displays project overview, stats, interactive classification form, and session history.
    """
    history = session.get("prediction_history", [])
    return render_template(
        "index.html",
        active_page="home",
        metrics=model_metrics,
        history=history
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Email Classification Route.
    Accepts email text via JSON or standard form submission,
    runs NLP preprocessing and TF-IDF Naive Bayes prediction,
    and returns JSON result or redirects.
    """
    try:
        # Support both JSON payload and standard form data
        if request.is_json:
            data = request.get_json() or {}
            email_text = data.get("email_text", "")
        else:
            email_text = request.form.get("email_text", "")

        # Run classification
        result = classify_email(email_text, classifier=classifier, vectorizer=vectorizer)

        # Store prediction in session history (capped at 15 items)
        if "prediction_history" not in session:
            session["prediction_history"] = []

        history_item = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "text": (email_text[:90] + "...") if len(email_text) > 90 else email_text,
            "category": result["predicted_category"],
            "confidence": result["confidence"],
            "badge_class": result["metadata"].get("bg_class", "bg-primary"),
            "icon": result["metadata"].get("icon", "bi-envelope")
        }

        # Keep history list limited to the 15 most recent entries
        history = [history_item] + session["prediction_history"][:14]
        session["prediction_history"] = history
        session.modified = True

        return jsonify(result), 200

    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred during classification: {str(exc)}"
        }), 500


@app.route("/dataset", methods=["GET"])
def dataset_view():
    """
    Dataset Overview Route.
    Displays dataset summary statistics, category distributions, and sample rows.
    """
    try:
        summary = get_dataset_summary()
        # Retrieve first 25 rows for exploration table
        df = load_and_validate_data()
        sample_rows = df.head(25).to_dict(orient="records")

        return render_template(
            "dataset.html",
            active_page="dataset",
            summary=summary,
            samples=sample_rows,
            metrics=model_metrics
        )
    except Exception as exc:
        return render_template(
            "dataset.html",
            active_page="dataset",
            summary={"total_emails": 0, "category_distribution": {}},
            samples=[],
            metrics=model_metrics,
            error=str(exc)
        )


@app.route("/model-performance", methods=["GET"])
def performance_view():
    """
    Model Performance Route.
    Displays empirical test accuracy, precision, recall, F1-scores, and confusion matrix.
    """
    return render_template(
        "performance.html",
        active_page="performance",
        metrics=model_metrics
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clears the session prediction history."""
    session["prediction_history"] = []
    session.modified = True
    return jsonify({"success": True, "message": "History cleared successfully."}), 200


if __name__ == "__main__":
    # Run development server suitable for student local machine
    print(">> Starting AI Email Classifier on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
