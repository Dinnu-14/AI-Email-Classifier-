"""
Unit and integration tests for the Flask web application routes.
"""

import pytest
from app import app
from src.data_loader import EXPECTED_CATEGORIES


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify home dashboard loads with status 200 and expected branding."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "AI Email Classifier" in html
    assert "Intelligent Email Classification" in html
    assert "MultinomialNB" in html


def test_dataset_route(client):
    """Verify dataset overview page loads with status 200 and category breakdown."""
    response = client.get("/dataset")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Predefined Dataset Overview" in html
    for cat in EXPECTED_CATEGORIES:
        assert cat in html


def test_performance_route(client):
    """Verify model performance page loads with status 200 and metrics."""
    response = client.get("/model-performance")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Model Evaluation &amp; Metrics" in html or "Model Evaluation & Metrics" in html
    assert "Confusion Matrix" in html
    assert "Classification Report" in html


def test_predict_endpoint_valid_json(client):
    """Verify /predict endpoint returns proper JSON structure for valid email."""
    payload = {
        "email_text": "Please submit your timesheet for this pay period before 6 PM today."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["success"] is True
    assert data["predicted_category"] in EXPECTED_CATEGORIES
    assert data["predicted_category"] == "Work"
    assert "confidence" in data
    assert "all_probabilities" in data
    assert "explanation" in data


def test_predict_endpoint_empty_input(client):
    """Verify /predict endpoint returns 400 Bad Request for empty string."""
    response = client.post("/predict", json={"email_text": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_predict_endpoint_symbols_only(client):
    """Verify /predict endpoint returns 400 Bad Request for symbols-only input."""
    response = client.post("/predict", json={"email_text": "!@#$%^&*()"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_clear_history_endpoint(client):
    """Verify /clear-history endpoint clears session history."""
    response = client.post("/clear-history")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
