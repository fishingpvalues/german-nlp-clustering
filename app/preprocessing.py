"""
Preprocessing utilities for German text: lowercasing, stopword removal, lemmatization (spaCy), and cleaning.
Optimized for production: lazy loading, caching, and SOTA German NLP.
"""

import re
from functools import lru_cache
from typing import Set

__all__ = ["preprocess_german_text"]


@lru_cache(maxsize=1)
def get_german_stopwords() -> Set[str]:
    """Return a set of German stopwords (NLTK + custom). Lazy-loads and caches."""
    try:
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("german"))
    except LookupError:
        import nltk

        nltk.download("stopwords")
        from nltk.corpus import stopwords

        stopwords_set = set(stopwords.words("german"))
    custom = {
        "bitte",
        "danke",
        "kunde",
        "kunden",
        "hosting",
        "server",
        "website",
        "webseiten",
    }
    return stopwords_set | custom


@lru_cache(maxsize=1)
def get_spacy_nlp():
    """Return spaCy German model, or None if unavailable. Lazy-loads and caches."""
    try:
        import spacy

        return spacy.load("de_core_news_sm")
    except Exception:
        return None


def preprocess_german_text(text: str) -> str:
    """
    Preprocess German text: lowercase, remove HTML/numbers/punctuation, remove stopwords, lemmatize (if spaCy available).
    Args:
        text (str): Input German text.
    Returns:
        str: Preprocessed text.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zäöüß\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    stopwords_set = get_german_stopwords()
    tokens = [w for w in text.split() if w not in stopwords_set]
    nlp = get_spacy_nlp()
    if nlp and tokens:
        doc = nlp(" ".join(tokens))
        tokens = [t.lemma_ for t in doc if t.lemma_ not in stopwords_set]
    return " ".join(tokens)
