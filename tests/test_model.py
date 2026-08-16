"""
tests/test_model.py
====================
Tests for the trained model and evaluation module.

Covers:
  - Model loads correctly
  - Prediction returns numeric value in [0, 100]
  - Invalid input raises an error
  - Evaluation metrics are correct
  - Quality gate logic
"""

import json
import os
import sys

import joblib
import numpy as np
import pytest

from src.evaluate import compute_metrics, quality_gate, MIN_R2, MAX_MAE
from src.predict import categorise, predict_score

# ---------------------------------------------------------------------------
# Quality gate tests (pure logic — no model file needed)
# ---------------------------------------------------------------------------
class TestQualityGate:
    def test_passes_when_metrics_are_good(self):
        metrics = {"r2": 0.85, "mae": 4.0, "rmse": 5.0}
        assert quality_gate(metrics, raise_on_fail=False) is True

    def test_fails_when_r2_low(self):
        metrics = {"r2": 0.50, "mae": 4.0, "rmse": 5.0}
        assert quality_gate(metrics, raise_on_fail=False) is False

    def test_fails_when_mae_high(self):
        metrics = {"r2": 0.85, "mae": 12.0, "rmse": 15.0}
        assert quality_gate(metrics, raise_on_fail=False) is False

    def test_raises_on_failure_when_requested(self):
        metrics = {"r2": 0.50, "mae": 15.0, "rmse": 18.0}
        with pytest.raises(ValueError, match="quality gate FAILED"):
            quality_gate(metrics, raise_on_fail=True)

    def test_min_thresholds_are_documented(self):
        assert MIN_R2 == 0.75
        assert MAX_MAE == 8.0


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([50.0, 60.0, 70.0, 80.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_metrics_are_non_negative(self):
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([48.0, 63.0, 68.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_r2_below_1_for_imperfect_predictions(self):
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_pred = np.array([52.0, 58.0, 73.0, 77.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["r2"] < 1.0

    def test_keys_present(self):
        y = np.array([50.0, 60.0])
        metrics = compute_metrics(y, y)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


class TestCategorise:
    def test_excellent(self):
        assert categorise(95.0) == "EXCELLENT"
        assert categorise(90.0) == "EXCELLENT"

    def test_very_good(self):
        assert categorise(80.0) == "VERY GOOD"
        assert categorise(75.0) == "VERY GOOD"

    def test_good(self):
        assert categorise(65.0) == "GOOD"
        assert categorise(60.0) == "GOOD"

    def test_average(self):
        assert categorise(50.0) == "AVERAGE"
        assert categorise(40.0) == "AVERAGE"

    def test_poor(self):
        assert categorise(20.0) == "POOR"
        assert categorise(0.0) == "POOR"

    def test_boundary_100(self):
        assert categorise(100.0) == "EXCELLENT"

    def test_boundary_0(self):
        assert categorise(0.0) == "POOR"


# ---------------------------------------------------------------------------
# Model artifact tests (skip if model not yet trained)
# ---------------------------------------------------------------------------
MODEL_EXISTS = os.path.exists(os.path.join("models", "best_model.joblib"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet — run train.py first")
class TestModelArtifact:
    def test_model_loads(self):
        model = joblib.load(os.path.join("models", "best_model.joblib"))
        assert model is not None

    def test_prediction_is_numeric(self, sample_features):
        result = predict_score(sample_features)
        assert isinstance(result["predicted_score"], (int, float))

    def test_prediction_in_valid_range(self, sample_features):
        result = predict_score(sample_features)
        assert 0 <= result["predicted_score"] <= 100

    def test_prediction_returns_performance(self, sample_features):
        result = predict_score(sample_features)
        assert result["performance"] in {"POOR", "AVERAGE", "GOOD", "VERY GOOD", "EXCELLENT"}

    def test_invalid_input_raises_error(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            predict_score({"study_hours": 5.0})  # missing 5 features

    def test_multiple_predictions_consistent(self, sample_features):
        """Same input should give same output (deterministic model)."""
        r1 = predict_score(sample_features)
        r2 = predict_score(sample_features)
        assert r1["predicted_score"] == r2["predicted_score"]

    def test_metadata_file_exists(self):
        assert os.path.exists(os.path.join("models", "model_metadata.json"))

    def test_metadata_has_required_keys(self):
        with open(os.path.join("models", "model_metadata.json")) as f:
            meta = json.load(f)
        for key in ("model_name", "model_version", "metrics", "features"):
            assert key in meta

    def test_model_metrics_pass_quality_gate(self):
        with open(os.path.join("models", "model_metadata.json")) as f:
            meta = json.load(f)
        metrics = meta["metrics"]
        assert quality_gate(metrics, raise_on_fail=False), (
            f"Saved model fails quality gate: R2={metrics['r2']}, MAE={metrics['mae']}"
        )
