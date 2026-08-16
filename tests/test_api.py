"""
tests/test_api.py
==================
Tests for FastAPI endpoints (api/main.py).

Uses httpx.AsyncClient (ASGI test client) — does NOT require a running server.

Covers:
  - GET /health
  - GET /model-info
  - POST /predict (valid input)
  - POST /predict (invalid input → validation error)
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Use synchronous TestClient for simplicity
client = TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestModelInfo:
    """These tests may return 503 if the model hasn't been trained."""

    def test_model_info_endpoint_reachable(self):
        response = client.get("/model-info")
        assert response.status_code in (200, 503)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model_metadata.json")),
        reason="Model not trained yet",
    )
    def test_model_info_returns_expected_keys(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        for key in ("model_name", "model_version", "model_type", "training_metrics"):
            assert key in data

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "model_metadata.json")),
        reason="Model not trained yet",
    )
    def test_model_type_is_regression(self):
        response = client.get("/model-info")
        assert response.json()["model_type"] == "regression"


class TestPredict:
    VALID_PAYLOAD = {
        "study_hours": 5.0,
        "attendance_percentage": 85.0,
        "previous_exam_score": 72.0,
        "assignment_completion_percentage": 90.0,
        "sleep_hours": 7.0,
        "extracurricular_hours": 2.0,
    }

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "best_model.joblib")),
        reason="Model not trained yet",
    )
    def test_predict_valid_input_returns_200(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "best_model.joblib")),
        reason="Model not trained yet",
    )
    def test_predict_score_in_range(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert 0 <= data["predicted_score"] <= 100

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "best_model.joblib")),
        reason="Model not trained yet",
    )
    def test_predict_performance_is_valid_category(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert data["performance"] in {"POOR", "AVERAGE", "GOOD", "VERY GOOD", "EXCELLENT"}

    def test_predict_missing_field_returns_422(self):
        """Missing required fields should return HTTP 422 (validation error)."""
        partial = {"study_hours": 5.0}
        response = client.post("/predict", json=partial)
        assert response.status_code == 422

    def test_predict_out_of_range_attendance_rejected(self):
        """Attendance > 100 should return HTTP 422."""
        bad_payload = {**self.VALID_PAYLOAD, "attendance_percentage": 150.0}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_negative_study_hours_rejected(self):
        """Negative study hours should return HTTP 422."""
        bad_payload = {**self.VALID_PAYLOAD, "study_hours": -1.0}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestMetrics:
    def test_metrics_endpoint_reachable(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_prometheus_text(self):
        response = client.get("/metrics")
        assert "health_requests_total" in response.text or "prediction" in response.text
