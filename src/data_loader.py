"""
Data Loader Module for AI Email Classifier.

Responsible for safely loading, validating, and inspecting the email dataset.
"""

import os
from typing import Dict, List, Tuple
import pandas as pd

from src.preprocessing import clean_text, is_valid_input

# Predefined valid categories for this project
EXPECTED_CATEGORIES = ["Work", "Personal", "Promotions", "Spam", "Important"]
REQUIRED_COLUMNS = ["email_text", "category"]

# Default dataset path relative to project root
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "emails.csv"
)


def load_and_validate_data(filepath: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Loads and validates the emails CSV dataset.

    Performs the following checks:
    1. Ensures the CSV file exists at the given path.
    2. Validates that required columns ('email_text', 'category') are present.
    3. Drops null or blank rows.
    4. Validates that every category belongs to the expected predefined categories.
    5. Applies text cleaning to the email_text column.
    6. Ensures there is sufficient training data for each category.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Validated and cleaned DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing, data is empty, or categories are invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: '{filepath}'. Please ensure 'dataset/emails.csv' exists."
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file at '{filepath}': {str(exc)}") from exc

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. Expected: {REQUIRED_COLUMNS}"
        )

    # Drop rows with null values in required columns
    initial_count = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Clean text and filter out empty / invalid texts
    df["email_text"] = df["email_text"].astype(str).apply(clean_text)
    df["category"] = df["category"].astype(str).str.strip()

    # Filter out empty or non-alphanumeric text rows
    valid_mask = df["email_text"].apply(is_valid_input)
    df = df[valid_mask].copy()

    # Validate categories against expected set
    invalid_categories = set(df["category"].unique()) - set(EXPECTED_CATEGORIES)
    if invalid_categories:
        raise ValueError(
            f"Dataset contains unexpected categories: {list(invalid_categories)}. "
            f"Allowed categories are: {EXPECTED_CATEGORIES}"
        )

    # Check if dataset has rows
    if len(df) == 0:
        raise ValueError("Dataset is empty after removing invalid or blank rows.")

    # Check if each expected category has at least a few samples
    counts = df["category"].value_counts()
    for cat in EXPECTED_CATEGORIES:
        if cat not in counts or counts[cat] < 2:
            raise ValueError(
                f"Category '{cat}' has insufficient samples ({counts.get(cat, 0)}) for stratified splitting."
            )

    return df


def get_dataset_summary(filepath: str = DEFAULT_DATASET_PATH) -> Dict:
    """
    Returns a dictionary summarizing dataset statistics for dashboard display.
    """
    df = load_and_validate_data(filepath)
    category_counts = df["category"].value_counts().to_dict()

    # Sample rows formatted for display
    sample_rows = df.head(10).to_dict(orient="records")

    return {
        "total_emails": int(len(df)),
        "total_categories": len(EXPECTED_CATEGORIES),
        "categories": EXPECTED_CATEGORIES,
        "category_distribution": category_counts,
        "samples": sample_rows,
        "dataset_path": filepath,
    }
