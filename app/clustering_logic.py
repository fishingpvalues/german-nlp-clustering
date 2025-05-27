"""
Clustering logic for German text: Top2Vec and UMAP+HDBSCAN.
Production-optimized: batch operations, minimal preprocessing, and SOTA clustering.
"""

from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .keyword_extraction import (
    get_top_tfidf_keywords_for_cluster,
    get_yake_keywords_for_cluster,
)
from .model_loader import (
    embedding_model_instance,
    hdbscan_clusterer,
    top2vec_model_instance,
    umap_reducer,
)
from .preprocessing import preprocess_german_text

__all__ = ["get_clusters_top2vec", "get_clusters_hdbscan"]


def get_clusters_top2vec(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Cluster German texts using Top2Vec. Returns topic and keywords for each text.
    Args:
        texts (List[str]): List of raw German texts.
    Returns:
        List[Dict[str, Any]]: Cluster results with topic and keywords.
    """
    if top2vec_model_instance is None:
        raise RuntimeError("Top2Vec model is not loaded.")
    processed_texts = [preprocess_german_text(text) for text in texts if text.strip()]
    if not processed_texts:
        return []
    topic_vectors = top2vec_model_instance.topic_vectors
    doc_embeddings = top2vec_model_instance._embed(processed_texts)
    similarity_matrix = cosine_similarity(doc_embeddings, topic_vectors)
    doc_topic_indices = similarity_matrix.argmax(axis=1)
    topic_words_list, _, _ = top2vec_model_instance.get_topics()
    results = []
    for i, text in enumerate(texts):
        topic_index = doc_topic_indices[i]
        keywords = (
            topic_words_list[topic_index][:5]
            if 0 <= topic_index < len(topic_words_list)
            else ["N/A"]
        )
        results.append(
            {
                "original_text": text,
                "processed_text": processed_texts[i],
                "cluster_method": "top2vec",
                "topic_number": int(topic_index),
                "keywords": keywords,
            }
        )
    return results


def get_clusters_hdbscan(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Cluster German texts using UMAP+HDBSCAN. Returns cluster and keywords for each text.
    Args:
        texts (List[str]): List of raw German texts.
    Returns:
        List[Dict[str, Any]]: Cluster results with cluster label and keywords.
    """
    if (
        embedding_model_instance is None
        or umap_reducer is None
        or hdbscan_clusterer is None
    ):
        raise RuntimeError("HDBSCAN/UMAP pipeline not loaded.")
    processed_texts = [preprocess_german_text(text) for text in texts if text.strip()]
    if not processed_texts:
        return []
    embeddings = embedding_model_instance.encode(
        processed_texts, convert_to_tensor=False
    )
    reduced_embeddings = umap_reducer.transform(embeddings)
    cluster_labels = hdbscan_clusterer.predict(reduced_embeddings)
    df = pd.DataFrame(
        {
            "original_text": texts,
            "processed_text": processed_texts,
            "cluster_label": cluster_labels,
        }
    )
    results = []
    for cluster_id in sorted(df["cluster_label"].unique()):
        if cluster_id == -1:
            continue
        cluster_texts = df[df["cluster_label"] == cluster_id]["original_text"].tolist()
        tfidf_keywords = get_top_tfidf_keywords_for_cluster(cluster_texts, top_n=5)
        yake_keywords = get_yake_keywords_for_cluster(cluster_texts, top_n=5)
        for text in cluster_texts:
            results.append(
                {
                    "original_text": text,
                    "processed_text": preprocess_german_text(text),
                    "cluster_method": "hdbscan",
                    "cluster_label": int(cluster_id),
                    "tfidf_keywords": tfidf_keywords,
                    "yake_keywords": yake_keywords,
                }
            )
    return results
