"""
Model loading utilities: Top2Vec, SentenceTransformer, UMAP, HDBSCAN.
Production-ready: thread/process safety, device selection, hot-reload, and logging.
"""

import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

import joblib
import torch
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec

__all__ = [
    "get_torch_device",
    "load_models",
    "get_model_version",
    "top2vec_model_instance",
    "embedding_model_instance",
    "umap_reducer",
    "hdbscan_clusterer",
]

MODEL_PATH = os.getenv(
    "MODEL_PATH", "/app/models/german_customer_suggestions_top2vec.model"
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
UMAP_PATH = os.getenv("UMAP_PATH", "/app/models/umap_reducer.joblib")
HDBSCAN_PATH = os.getenv("HDBSCAN_PATH", "/app/models/hdbscan_clusterer.joblib")
MODEL_VERSION_PATH = os.getenv("MODEL_VERSION_PATH", "/app/models/version.txt")

logger = logging.getLogger("service-logger")

# Model instances (global, but protected by lock)
top2vec_model_instance = None
embedding_model_instance = None
umap_reducer = None
hdbscan_clusterer = None
model_version = None
_model_lock = threading.Lock()


def get_torch_device() -> str:
    """Return the best available torch device (cuda, mps, cpu), or forced via env."""
    forced = os.getenv("TORCH_DEVICE")
    if forced:
        logger.info({"event": "device", "msg": f"Using forced device: {forced}"})
        return forced
    if torch.cuda.is_available():
        logger.info({"event": "device", "msg": "Using CUDA"})
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info({"event": "device", "msg": "Using MPS"})
        return "mps"
    else:
        logger.info({"event": "device", "msg": "Using CPU"})
        return "cpu"


def load_models() -> None:
    """Load all models (Top2Vec, embedding, UMAP, HDBSCAN) with thread/process safety."""
    global top2vec_model_instance, embedding_model_instance, umap_reducer, hdbscan_clusterer, model_version
    with _model_lock:
        device = get_torch_device()
        if top2vec_model_instance is None and os.path.exists(MODEL_PATH):
            top2vec_model_instance = Top2Vec.load(
                MODEL_PATH, embedding_model_override=EMBEDDING_MODEL_NAME
            )
        if embedding_model_instance is None:
            embedding_model_instance = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device=device
            )
        if umap_reducer is None and os.path.exists(UMAP_PATH):
            umap_reducer = joblib.load(UMAP_PATH)
        if hdbscan_clusterer is None and os.path.exists(HDBSCAN_PATH):
            hdbscan_clusterer = joblib.load(HDBSCAN_PATH)
        if os.path.exists(MODEL_VERSION_PATH):
            with open(MODEL_VERSION_PATH) as f:
                model_version = f.read().strip()
        else:
            model_version = "unknown"


def get_model_version() -> str:
    """Return the current model version string."""
    if os.path.exists(MODEL_VERSION_PATH):
        with open(MODEL_VERSION_PATH) as f:
            return f.read().strip()
    return "unknown"
