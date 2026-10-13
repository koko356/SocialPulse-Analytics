"""
Utility functions for text preprocessing and sentiment classification.
Extracts and preserves the exact logic from 01_dataset_inspection.ipynb
to ensure single-source-of-truth across Spark, Pandas, and Streamlit.
"""

import re
import string
from typing import Optional, Dict, Any

# Ensure stopwords are available with safe fallback
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        try:
            STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            STOP_WORDS = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                "your", "yours", "yourself", "yourselves", "he", "him", "his",
                "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                "they", "them", "their", "theirs", "themselves", "what", "which",
                "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and",
                "but", "if", "or", "because", "as", "until", "while", "of", "at",
                "by", "for", "with", "about", "against", "between", "into",
                "through", "during", "before", "after", "above", "below", "to",
                "from", "up", "down", "in", "out", "on", "off", "over", "under",
                "again", "further", "then", "once", "here", "there", "when",
                "where", "why", "how", "all", "any", "both", "each", "few",
                "more", "most", "other", "some", "such", "no", "nor", "not",
                "only", "own", "same", "so", "than", "too", "very", "s", "t",
                "can", "will", "just", "don", "should", "now"
            }
except Exception:
    STOP_WORDS = set()

_ANALYZER = None


def get_vader_analyzer():
    """Returns singleton SentimentIntensityAnalyzer instance."""
    global _ANALYZER
    if _ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def clean_text(text: Any) -> str:
    r"""
    Cleans raw text according to the exact project specification in
    01_dataset_inspection.ipynb (Cell 12):
    1. Cast to str
    2. Remove URLs (http\S+|www\S+|https\S+)
    3. Remove mentions (@\w+)
    4. Remove hashtags (#\w+)
    5. Remove special characters and numbers ([^A-Za-z\s])
    6. Convert to lowercase
    7. Remove English stopwords
    8. Return ' '.join(words)
    """
    if text is None:
        return ""
    text_str = str(text)

    # Remove URLs
    text_str = re.sub(r"http\S+|www\S+|https\S+", "", text_str)

    # Remove mentions
    text_str = re.sub(r"@\w+", "", text_str)

    # Remove hashtags
    text_str = re.sub(r"#\w+", "", text_str)

    # Remove special characters and numbers
    text_str = re.sub(r"[^A-Za-z\s]", "", text_str)

    # Convert to lowercase
    text_str = text_str.lower()

    # Remove stopwords
    words = text_str.split()
    if STOP_WORDS:
        words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)


def classify_sentiment(score: float) -> str:
    """
    Classifies VADER compound score according to notebook Cell 22:
    - score >= 0.05 -> 'Positive'
    - score <= -0.05 -> 'Negative'
    - otherwise -> 'Neutral'
    """
    if score is None:
        return "Neutral"
    try:
        val = float(score)
    except (ValueError, TypeError):
        return "Neutral"

    if val >= 0.05:
        return "Positive"
    elif val <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment_single(cleaned_text: str) -> Dict[str, Any]:
    """
    Runs VADER on a single cleaned text and returns scores and label.
    """
    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(cleaned_text)
    compound = scores.get("compound", 0.0)
    label = classify_sentiment(compound)
    return {
        "vader_scores": scores,
        "compound_score": compound,
        "sentiment_label": label,
    }
