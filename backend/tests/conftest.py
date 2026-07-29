"""
tests/conftest.py
=================
Shared pytest fixtures for the test suite.
"""

import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture(scope="session")
def app():
    """Create a fresh FastAPI application for the test session."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """
    Synchronous TestClient for the application.

    Uses FastAPI's built-in test client (backed by httpx) so that async
    routes are exercised without running a real ASGI server.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
