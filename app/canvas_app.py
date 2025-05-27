"""
Canvas UI for German text clustering SaaS. Production-ready: error handling, docstrings, and SOTA UX.
"""

import canvas as cv
import requests

API_URL = "http://localhost:8000/cluster-suggestions"

__all__ = ["main"]


def cluster_texts(texts, method):
    """
    Call clustering API and return results. Handles errors gracefully.
    Args:
        texts (List[str]): List of input texts.
        method (str): Clustering method ("top2vec" or "hdbscan").
    Returns:
        List[dict]: Clustering results or error message.
    """
    try:
        response = requests.post(
            API_URL, json={"suggestions": texts, "method": method}, timeout=30
        )
        if response.status_code == 200:
            return response.json()["results"]
        else:
            return [{"original_text": "Error", "keywords": [response.text]}]
    except Exception as e:
        return [{"original_text": "Error", "keywords": [str(e)]}]


def main():
    """
    Canvas UI main function for German text clustering.
    """
    with cv.page():
        cv.text(
            "German Customer Suggestion Clustering (SOTA)", size="xl", weight="bold"
        )
        input_text = cv.text_area(
            "Paste German customer suggestions (one per line):", height=200
        )
        method = cv.select(
            "Clustering Method", options=["top2vec", "hdbscan"], value="top2vec"
        )
        if cv.button("Cluster"):
            texts = [line.strip() for line in input_text.split("\n") if line.strip()]
            results = cluster_texts(texts, method)
            for res in results:
                cv.text(f"**Original:** {res['original_text']}")
                if method == "top2vec":
                    cv.text(
                        f"**Cluster Keywords:** {', '.join(res.get('keywords', []))}"
                    )
                else:
                    cv.text(
                        f"**TF-IDF Keywords:** {', '.join(res.get('tfidf_keywords', []))}"
                    )
                    cv.text(
                        f"**YAKE! Keywords:** {', '.join(res.get('yake_keywords', []))}"
                    )
                cv.divider()


if __name__ == "__main__":
    cv.run(main)
