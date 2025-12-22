"""
Text preprocessing utilities for NLP sentiment analysis.
Author: Alexy Louis
"""
import re
from typing import List

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts."""
    return [clean_text(t) for t in texts]
