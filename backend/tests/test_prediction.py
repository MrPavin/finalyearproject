"""
tests/test_prediction.py
=========================
Tests for the prediction endpoints.

Prediction logic is not implemented yet; these tests verify that the
API correctly handles stub responses and validation errors.
"""

import pytest
from fastapi import status


class TestPredictSingle:
    """Tests for POST /api/v1/predict/."""

    def test_empty_text_returns_422(self, client):
        """Blank text should fail Pydantic validation."""
        response = client.post("/api/v1/predict/", json={"text": "   "})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_text_returns_422(self, client):
        """Omitting required 'text' field should fail validation."""
        response = client.post("/api/v1/predict/", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_text_too_long_returns_422(self, client):
        """Text exceeding 5000 chars should fail validation."""
        response = client.post("/api/v1/predict/", json={"text": "a" * 5001})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_valid_request_returns_501_or_503(self, client):
        """
        A well-formed request should return 501 (not implemented) while
        the model is stubbed, or 503 if the model is not loaded.
        """
        response = client.post("/api/v1/predict/", json={"text": "Hello world"})
        assert response.status_code in (
            status.HTTP_501_NOT_IMPLEMENTED,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    def test_invalid_threshold_returns_422(self, client):
        """Threshold outside [0, 1] should fail validation."""
        response = client.post(
            "/api/v1/predict/",
            json={"text": "Test text", "threshold": 1.5},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_valid_language_accepted(self, client):
        """Valid language codes should pass validation (even if model stubs)."""
        for lang in ("en", "kn", "roman_kn", "hi", "auto"):
            response = client.post(
                "/api/v1/predict/",
                json={"text": "Test", "language": lang},
            )
            assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY, (
                f"Language '{lang}' was unexpectedly rejected."
            )


class TestPredictBatch:
    """Tests for POST /api/v1/predict/batch."""

    def test_empty_list_returns_422(self, client):
        response = client.post("/api/v1/predict/batch", json={"texts": []})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_valid_batch_returns_501_or_503(self, client):
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["Hello", "World"]},
        )
        assert response.status_code in (
            status.HTTP_501_NOT_IMPLEMENTED,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    def test_batch_too_large_returns_422(self, client):
        """More than 32 items should fail validation."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": ["text"] * 33},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestModelInfo:
    """Tests for GET /api/v1/predict/model/info."""

    def test_model_info_returns_200(self, client):
        response = client.get("/api/v1/predict/model/info")
        assert response.status_code == status.HTTP_200_OK

    def test_model_info_schema(self, client):
        data = client.get("/api/v1/predict/model/info").json()
        assert "model_name" in data
        assert "is_loaded" in data
        assert "labels" in data
        assert "supported_languages" in data
