"""
FastAPI app for German text clustering SaaS. Production-ready: metrics, JWT auth, CORS, logging, and SOTA clustering.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from starlette.responses import Response
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from .clustering_logic import get_clusters_hdbscan, get_clusters_top2vec
from .model_loader import get_model_version, load_models

MODEL_VERSION_PATH = os.getenv("MODEL_VERSION_PATH", "/app/models/version.txt")

# --- Logging setup ---
logger = logging.getLogger("service-logger")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# --- Prometheus metrics ---
REQUEST_COUNT = Counter("request_count", "Total API Requests", ["endpoint"])

# --- JWT Auth setup ---
from jose import JWTError, jwt

SECRET_KEY = os.getenv("API_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
security = HTTPBearer()


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


app = FastAPI(
    title="German Text Clustering SaaS",
    description="SOTA, production-grade clustering for German customer suggestions.",
    version="1.0.0",
    openapi_tags=[
        {"name": "clustering", "description": "Text clustering endpoints."},
        {"name": "health", "description": "Health and metrics endpoints."},
        {
            "name": "admin",
            "description": "Admin/model reload endpoints (JWT protected).",
        },
    ],
)

# --- CORS for UI integration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextSuggestionRequest(BaseModel):
    suggestions: List[str]
    method: Optional[str] = Query("top2vec", enum=["top2vec", "hdbscan"])


class ClusterResponse(BaseModel):
    results: List[Dict[str, Any]]


@app.on_event("startup")
async def startup_event() -> None:
    """Load models at startup."""
    load_models()
    logger.info({"event": "startup", "msg": "Models loaded."})


@app.post("/cluster-suggestions", response_model=ClusterResponse, tags=["clustering"])
async def cluster_suggestions_endpoint(
    request: TextSuggestionRequest,
) -> ClusterResponse:
    """Cluster German customer suggestions using Top2Vec or HDBSCAN."""
    REQUEST_COUNT.labels(endpoint="/cluster-suggestions").inc()
    logger.info(
        {
            "event": "request",
            "endpoint": "/cluster-suggestions",
            "method": request.method,
        }
    )
    if not request.suggestions:
        logger.warning({"event": "bad_request", "reason": "No suggestions provided."})
        raise HTTPException(status_code=400, detail="No suggestions provided.")
    try:
        if request.method == "top2vec":
            cluster_results = get_clusters_top2vec(request.suggestions)
        else:
            cluster_results = get_clusters_hdbscan(request.suggestions)
        return ClusterResponse(results=cluster_results)
    except Exception as e:
        logger.error({"event": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    version = get_model_version()
    return {"status": "healthy", "model_version": version}


@app.get("/metrics", tags=["health"])
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload-models", tags=["admin"])
async def reload_models(user=Depends(verify_jwt)) -> Dict[str, str]:
    """Reload all models (JWT protected)."""
    load_models()
    logger.info({"event": "reload", "msg": "Models reloaded via API."})
    return {"status": "reloaded"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(
        {"event": "http_request", "method": request.method, "url": str(request.url)}
    )
    try:
        response = await call_next(request)
        logger.info(
            {
                "event": "http_response",
                "status_code": response.status_code,
                "url": str(request.url),
            }
        )
        return response
    except Exception as e:
        logger.error(
            {"event": "unhandled_exception", "error": str(e), "url": str(request.url)}
        )
        raise


@app.exception_handler(HTTP_422_UNPROCESSABLE_ENTITY)
async def validation_exception_handler(request: Request, exc):
    logger.warning(
        {"event": "validation_error", "error": str(exc), "url": str(request.url)}
    )
    return Response(
        content='{"detail": "Validation error. Check your request payload."}',
        status_code=422,
        media_type="application/json",
    )
