"""
api/main.py
===========
FastAPI application for Student Performance Prediction.

Endpoints:
  GET  /health       → health check
  GET  /model-info   → model metadata
  POST /predict      → regression prediction
  GET  /metrics      → Prometheus metrics

Prometheus metrics tracked:
  - prediction_requests_total   (counter)
  - prediction_latency_seconds  (histogram)
  - api_errors_total            (counter)

The model is loaded ONCE at startup (lifespan event), not per request.
"""

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import logging

# ── Path setup (supports running from repo root) ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_score, get_metadata, load_artifacts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
API_ERRORS = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint"],
)
HEALTH_REQUESTS = Counter(
    "health_requests_total",
    "Total number of health check requests",
)


# ---------------------------------------------------------------------------
# Lifespan — load model ONCE at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model and preprocessor into memory."""
    logger.info("Loading model artifacts at startup ...")
    try:
        load_artifacts()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Could not load model: {e}")
        logger.warning("API will start but /predict will fail until model is trained.")
    yield
    # Shutdown — nothing to clean up
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "MLOps project for IT4V43. "
        "Predicts a student's final exam score using regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    study_hours: float = Field(..., ge=0, le=24, description="Daily study hours")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Attendance %")
    previous_exam_score: float = Field(..., ge=0, le=100, description="Previous exam score")
    assignment_completion_percentage: float = Field(..., ge=0, le=100, description="Assignment completion %")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average nightly sleep hours")
    extracurricular_hours: float = Field(..., ge=0, le=24, description="Daily extracurricular hours")

    model_config = {"json_schema_extra": {
        "example": {
            "study_hours": 5,
            "attendance_percentage": 85,
            "previous_exam_score": 72,
            "assignment_completion_percentage": 90,
            "sleep_hours": 7,
            "extracurricular_hours": 2,
        }
    }}


class PredictResponse(BaseModel):
    predicted_score: float
    performance: str


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Simple health check."""
    HEALTH_REQUESTS.inc()
    return {"status": "healthy"}


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Return model metadata: name, version, type, training metrics."""
    try:
        meta = get_metadata()
        if not meta:
            raise HTTPException(status_code=503, detail="Model metadata not available.")
        return {
            "model_name": meta.get("model_name", "unknown"),
            "model_version": meta.get("model_version", "unknown"),
            "model_type": "regression",
            "features": meta.get("features", []),
            "target": meta.get("target", "final_exam_score"),
            "training_metrics": meta.get("metrics", {}),
            "quality_gate": meta.get("quality_gate", {}),
            "mlflow_run_id": meta.get("mlflow_run_id", ""),
            "random_seed": meta.get("random_seed", 42),
        }
    except Exception as exc:
        API_ERRORS.labels(endpoint="/model-info").inc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict a student's final exam score.

    Returns predicted_score (0–100) and performance category.
    """
    start = time.time()
    try:
        features = request.model_dump()
        result = predict_score(features)
        PREDICTION_REQUESTS.inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        logger.info(f"Prediction: score={result['predicted_score']}, perf={result['performance']}")
        return result
    except FileNotFoundError as exc:
        API_ERRORS.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")
    except Exception as exc:
        API_ERRORS.labels(endpoint="/predict").inc()
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entry point (for development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
