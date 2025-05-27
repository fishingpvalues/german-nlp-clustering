"""
Keyword extraction utilities: TF-IDF and YAKE for German text clusters.
Production-optimized: batch, robust, and SOTA.
"""

from typing import List

import numpy as np
import yake
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["get_top_tfidf_keywords_for_cluster", "get_yake_keywords_for_cluster"]


def get_top_tfidf_keywords_for_cluster(
    texts_in_cluster: List[str], top_n: int = 5
) -> List[str]:
    """
    Extract top-N TF-IDF keywords from a list of texts.
    Args:
        texts_in_cluster (List[str]): List of texts in a cluster.
        top_n (int): Number of keywords to return.
    Returns:
        List[str]: Top-N keywords by TF-IDF.
    """
    if not texts_in_cluster or len(texts_in_cluster) < 1:
        return []
    vectorizer = TfidfVectorizer(stop_words=None, max_features=1000, ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(texts_in_cluster)
        feature_names = vectorizer.get_feature_names_out()
        summed_tfidf = tfidf_matrix.sum(axis=0)
        sorted_indices = np.argsort(summed_tfidf).A1[::-1]
        top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        return top_keywords
    except ValueError:
        return []


def get_yake_keywords_for_cluster(
    texts_in_cluster: List[str], top_n: int = 5, language: str = "de"
) -> List[str]:
    """
    Extract top-N YAKE keywords from a list of texts.
    Args:
        texts_in_cluster (List[str]): List of texts in a cluster.
        top_n (int): Number of keywords to return.
        language (str): Language code for YAKE.
    Returns:
        List[str]: Top-N keywords by YAKE.
    """
    if not texts_in_cluster:
        return []
    full_text_of_cluster = " ".join(texts_in_cluster)
    if not full_text_of_cluster.strip():
        return []
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language, n=2, dedupLim=0.9, top=top_n, features=None
    )
    try:
        keywords = custom_kw_extractor.extract_keywords(full_text_of_cluster)
        return [kw for kw, score in keywords]
    except Exception:
        return []
