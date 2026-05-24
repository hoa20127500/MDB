"""
Admin API routes for pipeline management and health checks.

Exposes:
- ``POST /admin/pipeline/trigger``: Manually trigger a pipeline run.
- ``GET  /admin/pipeline/status``:  Get the current pipeline run status.
- ``GET  /health``:                 Health check endpoint.

Requirements 8.2, 8.4.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

admin_router = APIRouter()


@admin_router.post("/admin/pipeline/trigger")
async def trigger_pipeline(
    request: Request,
    mode: Literal["full", "incremental"] = "incremental",
) -> JSONResponse:
    """Manually trigger a pipeline run.

    Accepts an optional ``mode`` query parameter (``"full"`` or
    ``"incremental"``; defaults to ``"incremental"``).

    Returns:
        - **HTTP 200** with the result dict when the run is accepted.
        - **HTTP 409** with a status message when a run is already in progress
          (Requirement 8.4).
    """
    orchestrator = request.app.state.orchestrator
    result: dict = await orchestrator.run(mode=mode)

    if result.get("status") == "already_running":
        logger.warning(
            "Pipeline trigger rejected — run already in progress",
            extra={"mode": mode},
        )
        return JSONResponse(status_code=409, content=result)

    logger.info("Pipeline run triggered via admin API", extra={"mode": mode})
    return JSONResponse(status_code=200, content=result)


@admin_router.get("/admin/pipeline/status")
async def pipeline_status(request: Request) -> JSONResponse:
    """Return the current pipeline run status.

    Calls :meth:`~app.pipeline.orchestrator.PipelineOrchestrator.get_status`
    and returns the resulting dict as JSON.
    """
    orchestrator = request.app.state.orchestrator
    status: dict = await orchestrator.get_status()
    return JSONResponse(status_code=200, content=status)


@admin_router.get("/health")
async def health_check() -> JSONResponse:
    """Simple health check endpoint.

    Returns ``{"status": "ok"}`` with HTTP 200.  No app state access is
    required — this endpoint is intentionally lightweight so that load
    balancers and container orchestrators can probe it cheaply.
    """
    return JSONResponse(status_code=200, content={"status": "ok"})
