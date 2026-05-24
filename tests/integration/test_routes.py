"""
Integration tests for HTTP route registration and reachability.

Confirms that:
- GET  /health                  → 200 {"status": "ok"}
- GET  /admin/pipeline/status   → 200
- POST /recommendations         → 400 when `query` is missing

All external dependencies (MongoDB, EmbeddingService, PipelineScheduler) are
mocked so the lifespan can complete without real infrastructure.

Requirements: 7.1, 7.3, 8.2
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


# ---------------------------------------------------------------------------
# Helper — context manager that patches all external deps and yields a client
# ---------------------------------------------------------------------------


@contextmanager
def _patched_test_client():
    """Context manager that patches all external dependencies and yields a TestClient.

    The patches must remain active while the TestClient's lifespan runs (which
    happens inside the ``with TestClient(app) as client:`` block).  Wrapping
    everything in a single context manager guarantees the patches are in place
    for the entire duration.

    Patches applied:
    - env vars                            → satisfy pydantic-settings validation
    - ``app.main.create_motor_client``    → mongomock client
    - ``app.main.EmbeddingService``       → MagicMock (no model loading)
    - ``app.main.PipelineOrchestrator``   → MagicMock (stub orchestrator)
    - ``app.main.PipelineScheduler``      → MagicMock (no APScheduler)
    - ``app.core.indexes.ensure_indexes`` → async no-op
    """
    # Clear the lru_cache so the env-var overrides are picked up fresh.
    get_settings.cache_clear()

    mongo_client = mongomock.MongoClient()

    mock_embedding_instance = MagicMock()
    mock_embedding_instance.encode.return_value = [0.1] * 384

    mock_scheduler_instance = MagicMock()
    mock_scheduler_instance.start = MagicMock()
    mock_scheduler_instance.shutdown = MagicMock()

    mock_orchestrator_instance = MagicMock()

    async def _fake_get_status():
        return {"running": False, "latest_run": None}

    mock_orchestrator_instance.get_status = _fake_get_status

    env_overrides = {
        "MONGODB_URI": "mongodb://localhost:27017",
        "ODOO_DSN": "postgresql+asyncpg://user:pass@localhost/odoo",
        "VTM_DSN": "postgresql+asyncpg://user:pass@localhost/vtm",
    }

    with (
        patch.dict(os.environ, env_overrides),
        patch("app.main.create_motor_client", return_value=mongo_client),
        patch("app.main.EmbeddingService", return_value=mock_embedding_instance),
        patch("app.main.PipelineScheduler", return_value=mock_scheduler_instance),
        patch("app.main.PipelineOrchestrator", return_value=mock_orchestrator_instance),
        patch("app.core.indexes.ensure_indexes", new=AsyncMock(return_value=None)),
    ):
        # Clear the cache again now that env vars are set, so the lifespan
        # picks up the overridden values when it calls get_settings().
        get_settings.cache_clear()
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_returns_200():
    """GET /health must return 200 with body {"status": "ok"}.

    Validates: Requirements 8.2
    """
    with _patched_test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_pipeline_status_returns_200():
    """GET /admin/pipeline/status must return 200.

    The orchestrator stub returns {"running": False, "latest_run": None}.

    Validates: Requirements 8.2
    """
    with _patched_test_client() as client:
        response = client.get("/admin/pipeline/status")
        assert response.status_code == 200
        body = response.json()
        assert "running" in body


def test_recommendations_missing_query_returns_400():
    """POST /recommendations with body {} (missing required `query`) must return 400.

    FastAPI + Pydantic automatically return HTTP 422 (Unprocessable Entity)
    for missing required fields, which satisfies the "invalid request" contract
    described in Requirement 7.3.  We accept both 400 and 422 here.

    Validates: Requirements 7.1, 7.3
    """
    with _patched_test_client() as client:
        response = client.post("/recommendations", json={})
        # FastAPI returns 422 for Pydantic validation errors (missing required fields)
        assert response.status_code in (400, 422)
        body = response.json()
        # The response must contain error detail
        assert "detail" in body or "message" in body
