"""
Pipeline Scheduler — full APScheduler implementation (Task 11.3).

Uses ``APScheduler``'s ``AsyncIOScheduler`` to trigger
``PipelineOrchestrator.run(mode="incremental")`` on a configurable interval.

Requirements: 8.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.pipeline.orchestrator import PipelineOrchestrator

logger = get_logger(__name__)


class PipelineScheduler:
    """Wraps APScheduler to trigger the pipeline on a configurable interval.

    Args:
        orchestrator: The :class:`~app.pipeline.orchestrator.PipelineOrchestrator`
            instance whose ``run`` method will be called on each scheduled tick.
    """

    def __init__(self, orchestrator: "PipelineOrchestrator") -> None:
        self._orchestrator = orchestrator
        self._scheduler = None  # Created lazily in start()

    def start(self, interval_hours: int = 24) -> None:
        """Start the interval scheduler.

        Creates an ``AsyncIOScheduler`` (if not already created), registers an
        interval job that calls ``orchestrator.run(mode="incremental")`` every
        *interval_hours* hours, and starts the scheduler.

        Calling ``start()`` when the scheduler is already running is a no-op.

        Args:
            interval_hours: How often (in hours) the pipeline should run.
                Defaults to 24.
        """
        if self._scheduler is not None and self._scheduler.running:
            logger.warning(
                "Pipeline scheduler already running — ignoring duplicate start()",
                extra={"interval_hours": interval_hours},
            )
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self._orchestrator.run,
            "interval",
            hours=interval_hours,
            kwargs={"mode": "incremental"},
        )
        self._scheduler.start()

        logger.info(
            "Pipeline scheduler started",
            extra={"interval_hours": interval_hours},
        )

    def shutdown(self) -> None:
        """Gracefully stop the scheduler.

        Calls ``scheduler.shutdown(wait=False)`` so that the FastAPI shutdown
        sequence is not blocked by in-flight jobs.
        """
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Pipeline scheduler stopped")
        else:
            logger.debug("Pipeline scheduler shutdown called but scheduler was not running")
