"""
Text Preprocessing Module for AI Email Classifier.

This module provides clean, explainable NLP text preprocessing functions
suitable for a student mini project.
"""

import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """
    Cleans and normalizes email text for NLP feature extraction.

    Steps performed:
    1. Check for None, empty, or non-string inputs safely.
    2. Convert all text to lowercase to prevent casing discrepancies (e.g., 'URGENT' vs 'urgent').
    3. Normalize whitespace (remove multiple spaces, tabs, and newlines).
    4. Strip leading and trailing whitespace.

    Parameters:
        text (str): Raw email subject or body text.

    Returns:
        str: Cleaned, normalized lowercase string. Returns an empty string if input is invalid.
    """
    if text is None:
        return ""

    # Ensure input is a string
    if not isinstance(text, str):
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace newlines, carriage returns, and tabs with single spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    return text.strip()


def is_valid_input(text: Optional[str]) -> bool:
    """
    Validates whether the provided text contains meaningful non-whitespace content.

    Parameters:
        text (str): Input text to validate.

    Returns:
        bool: True if text contains at least one alphanumeric character, False otherwise.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return False
    # Check if there is at least one letter or digit
    return bool(re.search(r"[a-zA-Z0-9]", cleaned))
