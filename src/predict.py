"""
predict.py
==========
Inference module for student performance prediction.

Architecture:
  - Loads the preprocessor (fitted Scikit-learn Pipeline)
  - Loads the best model (Joblib artifact)
  - Accepts a dict or DataFrame of features
  - Returns predicted score (clipped to [0, 100]) and performance category

Performance categories:
   0–39  → POOR
  40–59  → AVERAGE
  60–74  → GOOD
  75–89  → VERY GOOD
  90–100 → EXCELLENT

Usage:
    from src.predict import predict_score

    result = predict_score({
        "study_hours": 5,
        "attendance_percentage": 85,
        "previous_exam_score": 72,
        "assignment_completion_percentage": 90,
        "sleep_hours": 7,
        "extracurricular_hours": 2,
    })
    # → {"predicted_score": 78.4, "performance": "VERY GOOD"}
"""

import json
import os
import sys
from typing import Dict, Union

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import FEATURES, load_preprocessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "best_model.joblib")
METADATA_PATH = os.path.join("models", "model_metadata.json")


# ---------------------------------------------------------------------------
# Performance category
# ---------------------------------------------------------------------------
PERFORMANCE_BANDS = [
    (90, 100, "EXCELLENT"),
    (75, 89.99, "VERY GOOD"),
    (60, 74.99, "GOOD"),
    (40, 59.99, "AVERAGE"),
    (0, 39.99, "POOR"),
]


def categorise(score: float) -> str:
    """Map a numeric score to its performance category string."""
    for lo, hi, label in PERFORMANCE_BANDS:
        if lo <= score <= hi:
            return label
    return "POOR"  # fallback for edge cases


# ---------------------------------------------------------------------------
# Model singleton (load once)
# ---------------------------------------------------------------------------
_model = None
_preprocessor = None
_metadata = None


def _load_artifacts():
    """Load model and preprocessor into module-level singletons."""
    global _model, _preprocessor, _metadata

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train.py first."
            )
        _model = joblib.load(MODEL_PATH)

    if _preprocessor is None:
        _preprocessor = load_preprocessor()

    if _metadata is None and os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)

    return _model, _preprocessor, _metadata


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------
def predict_score(features: Union[Dict, pd.DataFrame]) -> Dict:
    """
    Predict the final exam score for a student.

    Parameters
    ----------
    features : dict or single-row DataFrame with the six feature columns

    Returns
    -------
    dict:
        predicted_score : float (0–100, 1 decimal place)
        performance     : str  (POOR / AVERAGE / GOOD / VERY GOOD / EXCELLENT)
    """
    model, preprocessor, metadata = _load_artifacts()

    # Normalise input to DataFrame
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError(f"features must be dict or DataFrame, got {type(features)}")

    # Validate required features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURES].astype(float)

    # Preprocess (uses the same pipeline fitted during training — no leakage)
    X_t = preprocessor.transform(X)

    # Predict and clip
    raw = float(model.predict(X_t)[0])
    score = round(float(np.clip(raw, 0, 100)), 1)

    return {
        "predicted_score": score,
        "performance": categorise(score),
    }


def get_metadata() -> Dict:
    """Return model metadata (name, version, metrics, etc.)."""
    _, _, metadata = _load_artifacts()
    return metadata or {}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "study_hours": 5.0,
        "attendance_percentage": 85.0,
        "previous_exam_score": 72.0,
        "assignment_completion_percentage": 90.0,
        "sleep_hours": 7.0,
        "extracurricular_hours": 2.0,
    }
    result = predict_score(sample)
    print(f"\nSample input:  {sample}")
    print(f"Prediction:    {result}")
