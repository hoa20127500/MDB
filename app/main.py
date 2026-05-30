"""
FastAPI application factory for the Product Recommendation Engine.

Startup sequence (via lifespan context manager):
1. ``configure_logging()`` — structured JSON logging.
2. Motor client + database — shared across all requests via ``app.state``.
3. ``EmbeddingService`` — loads the sentence-transformers model once.
4. ``BehaviorTracker`` — wraps the ``user_behaviors`` collection.
5. ``RecommendationEngine`` — orchestrates context building + vector search.
6. ``PipelineOrchestrator`` — coordinates ETL runs.
7. ``PipelineScheduler`` — triggers the pipeline on a configurable interval.

Shutdown sequence (reverse order):
- Scheduler is stopped.
- Motor client is closed.

A global exception handler catches any unhandled exception, logs the full
stack trace with the current request ID, and returns HTTP 500 with
``{"error_code": "INTERNAL_ERROR"}`` (Requirement 7.4).

A request-ID middleware sets ``request_id_var`` from the ``X-Request-ID``
header (or generates a UUID v4 if the header is absent) so that every log
record emitted during a request carries the same ``request_id`` field.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.admin import admin_router
from app.api.behavior import BehaviorTracker
from app.api.recommendation import RecommendationEngine
from app.api.routes import router
from app.core.config import get_settings
from app.core.db import create_motor_client, get_database
from app.core.embedding import EmbeddingService
from app.core.indexes import ensure_indexes
from app.core.logging import configure_logging, get_logger, request_id_var
from app.pipeline.adapters.odoo import OdooAdapter
from app.pipeline.adapters.vtm import VTMAdapter
from app.pipeline.loader import Loader
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.scheduler import PipelineScheduler

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    All services are initialised during the startup phase and stored on
    ``app.state`` so that route handlers and dependency-injection functions
    can access them without importing module-level singletons.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("Starting Product Recommendation Engine")

    settings = get_settings()

    # Motor client and database
    client = create_motor_client(settings.MONGODB_URI)
    db = get_database(client)
    app.state.db = db
    app.state.motor_client = client

    # MongoDB indexes
    await ensure_indexes(db)

    # Embedding service (loads model — may take a few seconds on first run)
    embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    app.state.embedding_service = embedding_service

    # Behavior tracker
    behavior_tracker = BehaviorTracker(db)
    app.state.behavior_tracker = behavior_tracker

    # Recommendation engine
    recommendation_engine = RecommendationEngine(db, embedding_service, behavior_tracker)
    app.state.recommendation_engine = recommendation_engine

    # Pipeline orchestrator — wired with real adapters and loader
    odoo_adapter = OdooAdapter(dsn=settings.ODOO_DSN, limit=settings.EXTRACT_LIMIT)
    vtm_adapter = VTMAdapter(dsn=settings.VTM_DSN, limit=settings.EXTRACT_LIMIT)
    loader = Loader(db=db, embedding_service=embedding_service)
    orchestrator = PipelineOrchestrator(
        db=db,
        loader=loader,
        odoo_adapter=odoo_adapter,
        vtm_adapter=vtm_adapter,
        settings=settings,
    )
    app.state.orchestrator = orchestrator

    # Pipeline scheduler
    scheduler = PipelineScheduler(orchestrator)
    scheduler.start(interval_hours=settings.SCHEDULER_INTERVAL_HOURS)
    app.state.scheduler = scheduler

    logger.info("All services initialised — application ready")

    yield  # ── Application is running ──────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down Product Recommendation Engine")

    scheduler.shutdown()
    client.close()

    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance with lifespan
        hooks, middleware, and exception handlers registered.
    """
    application = FastAPI(
        title="Product Recommendation Engine",
        description=(
            "Personalized product recommendations powered by MongoDB Atlas "
            "Vector Search and sentence-transformers embeddings."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    application.include_router(router)
    application.include_router(admin_router)

    # ── Request-ID middleware ─────────────────────────────────────────────────

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request ID to every request for log correlation.

        Reads the ``X-Request-ID`` header when present; otherwise generates a
        UUID v4.  The value is stored in ``request_id_var`` so that all log
        records emitted during the request carry the same ``request_id`` field.
        The same ID is echoed back in the ``X-Request-ID`` response header.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    # ── Global exception handler ──────────────────────────────────────────────

    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch any unhandled exception and return HTTP 500.

        Logs the full stack trace together with the current request ID so that
        errors can be correlated with the originating request in log aggregators
        (Requirement 7.4).
        """
        logger.error(
            "Unhandled exception",
            exc_info=True,
            extra={"request_id": request_id_var.get()},
        )
        return JSONResponse(
            status_code=500,
            content={"error_code": "INTERNAL_ERROR"},
        )

    return application


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn / ASGI servers)
# ---------------------------------------------------------------------------

app = create_app()
