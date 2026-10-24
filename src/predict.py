"""
Prediction Module for AI Email Classifier.

Takes raw user input, applies preprocessing, vectorization, and predicts
the email category along with class confidence estimates using MultinomialNB.
"""

from typing import Dict, Any, Optional
from src.preprocessing import clean_text, is_valid_input
from src.train_model import get_or_train_model

# Mapping categories to UI colors and icons for rich presentation
CATEGORY_METADATA = {
    "Work": {
        "color": "primary",
        "bg_class": "bg-primary-subtle text-primary border-primary",
        "icon": "bi-briefcase-fill",
        "description": "Work-related communications, meetings, tasks, and project updates."
    },
    "Personal": {
        "color": "info",
        "bg_class": "bg-info-subtle text-info-emphasis border-info",
        "icon": "bi-people-fill",
        "description": "Messages from friends, family, greetings, and personal conversations."
    },
    "Promotions": {
        "color": "warning",
        "bg_class": "bg-warning-subtle text-warning-emphasis border-warning",
        "icon": "bi-tag-fill",
        "description": "Sales, discount offers, marketing promotions, and commercial deals."
    },
    "Spam": {
        "color": "danger",
        "bg_class": "bg-danger-subtle text-danger border-danger",
        "icon": "bi-shield-slash-fill",
        "description": "Suspicious offers, fake lottery prizes, and unsolicited phishing messages."
    },
    "Important": {
        "color": "success",
        "bg_class": "bg-success-subtle text-success border-success",
        "icon": "bi-exclamation-triangle-fill",
        "description": "Critical alerts, account security notices, exams, and urgent deadlines."
    }
}


def classify_email(
    raw_text: Optional[str],
    classifier=None,
    vectorizer=None
) -> Dict[str, Any]:
    """
    Classifies a raw email text string into one of the 5 predefined categories.

    Parameters:
        raw_text (str): The raw email text provided by the user.
        classifier: Trained MultinomialNB instance (optional, will load if None).
        vectorizer: Fitted TfidfVectorizer instance (optional, will load if None).

    Returns:
        dict: Prediction results including predicted category, confidence,
              probability distribution across all categories, and model explanation.

    Raises:
        ValueError: If input is empty or contains no alphanumeric text.
    """
    if not is_valid_input(raw_text):
        raise ValueError(
            "Please provide meaningful email text. The input cannot be empty or contain only symbols/spaces."
        )

    # 1. Preprocess
    cleaned = clean_text(raw_text)

    # 2. Ensure model and vectorizer are loaded
    if classifier is None or vectorizer is None:
        classifier, vectorizer, _ = get_or_train_model()

    # 3. Transform using fitted TF-IDF
    features = vectorizer.transform([cleaned])

    # 4. Predict category
    predicted_category = str(classifier.predict(features)[0])

    # 5. Extract probabilities
    probs = classifier.predict_proba(features)[0]
    classes = list(classifier.classes_)

    # Map each class to its estimated percentage
    prob_distribution = {
        cls_name: round(float(prob) * 100, 2)
        for cls_name, prob in zip(classes, probs)
    }

    # Sort distribution descending by probability
    sorted_probs = dict(
        sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True)
    )

    confidence = prob_distribution.get(predicted_category, 0.0)
    meta = CATEGORY_METADATA.get(predicted_category, {
        "color": "secondary",
        "bg_class": "bg-secondary text-white",
        "icon": "bi-envelope-fill",
        "description": "Classified email"
    })

    return {
        "success": True,
        "input_preview": (raw_text[:120] + "...") if len(raw_text) > 120 else raw_text,
        "cleaned_text": cleaned,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "all_probabilities": sorted_probs,
        "metadata": meta,
        "explanation": (
            f"The model predicted '{predicted_category}' with an estimated confidence of {confidence}%. "
            "Note: Probability values are statistical estimates derived from training feature weights, "
            "not absolute guarantees of real-world correctness."
        )
    }
