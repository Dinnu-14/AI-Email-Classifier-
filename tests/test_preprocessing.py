"""
Unit tests for text preprocessing functions.
"""

import pytest
from src.preprocessing import clean_text, is_valid_input


def test_clean_text_lowercasing():
    """Verify text is converted to lowercase."""
    raw = "URGENT: Review Q3 Project Roadmap NOW!"
    expected = "urgent: review q3 project roadmap now!"
    assert clean_text(raw) == expected


def test_clean_text_whitespace_normalization():
    """Verify excessive spaces, tabs, and newlines are collapsed to a single space."""
    raw = "  Hello   \n\n\t team,   let's meet   today.  "
    expected = "hello team, let's meet today."
    assert clean_text(raw) == expected


def test_clean_text_empty_and_none():
    """Verify None and empty strings safely return empty strings."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_clean_text_non_string():
    """Verify non-string inputs are converted to strings safely."""
    assert clean_text(12345) == "12345"


def test_is_valid_input():
    """Verify input validation correctly filters empty or symbol-only inputs."""
    # Valid inputs
    assert is_valid_input("Meeting at 10 AM") is True
    assert is_valid_input("50% discount sale") is True
    assert is_valid_input("a") is True

    # Invalid inputs
    assert is_valid_input("") is False
    assert is_valid_input("    ") is False
    assert is_valid_input(None) is False
    assert is_valid_input("!@#$%^&*()_+{}[]:;<>?,./~`") is False
