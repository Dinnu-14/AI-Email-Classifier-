"""
Integration and unit tests for dataset loading, model training, and prediction.
"""

import os
import pytest
from src.data_loader import load_and_validate_data, EXPECTED_CATEGORIES
from src.train_model import train_and_evaluate
from src.predict import classify_email


def test_dataset_loads_and_contains_required_categories():
    """Verify dataset loads successfully and all 5 predefined categories are present."""
    df = load_and_validate_data()
    assert len(df) >= 100, f"Expected at least 100 samples, got {len(df)}"

    unique_categories = set(df["category"].unique())
    assert unique_categories == set(EXPECTED_CATEGORIES), (
        f"Categories mismatch: {unique_categories} vs {EXPECTED_CATEGORIES}"
    )

    # Check each category has at least 20 samples
    counts = df["category"].value_counts()
    for cat in EXPECTED_CATEGORIES:
        assert counts[cat] >= 20, f"Category '{cat}' has fewer than 20 samples ({counts[cat]})"


def test_missing_dataset_raises_file_not_found():
    """Verify loading from non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("non_existent_folder/missing_file.csv")


def test_model_training_and_evaluation():
    """Verify model trains cleanly and produces legitimate, non-trivial evaluation metrics."""
    metrics = train_and_evaluate(save_artifacts=False)

    assert metrics["is_trained"] is True
    assert metrics["total_samples"] >= 100
    assert metrics["train_samples"] > metrics["test_samples"]
    assert 0.0 < metrics["test_accuracy"] <= 100.0

    # Ensure confusion matrix dimensions match categories count
    cm = metrics["confusion_matrix"]
    assert len(cm) == len(EXPECTED_CATEGORIES)
    for row in cm:
        assert len(row) == len(EXPECTED_CATEGORIES)

    # Ensure classification report contains each class
    report = metrics["classification_report"]
    for cat in EXPECTED_CATEGORIES:
        assert cat in report
        assert "precision" in report[cat]
        assert "recall" in report[cat]
        assert "f1-score" in report[cat]


def test_prediction_valid_work_email():
    """Verify predicting a work email returns one of the 5 categories and probability data."""
    text = "Team standup meeting tomorrow at 9 AM in the main conference room to discuss sprint backlog."
    result = classify_email(text)

    assert result["success"] is True
    assert result["predicted_category"] in EXPECTED_CATEGORIES
    assert result["predicted_category"] == "Work"
    assert 0.0 <= result["confidence"] <= 100.0

    # All probabilities should sum close to 100%
    probs = result["all_probabilities"]
    assert set(probs.keys()) == set(EXPECTED_CATEGORIES)
    total_prob = sum(probs.values())
    assert 99.0 <= total_prob <= 101.0  # Accounting for rounding


def test_prediction_valid_spam_email():
    """Verify predicting a spam email returns one of the 5 categories."""
    text = "Congratulations! You have won $1,000,000 in the sweepstakes lottery! Claim your prize now."
    result = classify_email(text)

    assert result["success"] is True
    assert result["predicted_category"] in EXPECTED_CATEGORIES
    assert result["predicted_category"] == "Spam"


def test_prediction_empty_input_raises_value_error():
    """Verify empty or symbol-only input raises ValueError."""
    with pytest.raises(ValueError):
        classify_email("")

    with pytest.raises(ValueError):
        classify_email("     ")

    with pytest.raises(ValueError):
        classify_email("!@#$%^")
