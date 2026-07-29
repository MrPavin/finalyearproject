"""
tests/test_health.py
=====================
Tests for the /health endpoint and root endpoint.
"""

from fastapi import status


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_health_status_is_string(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_model_loaded_is_bool(self, client):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

    def test_root_contains_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data
        assert data["docs"] == "/docs"
