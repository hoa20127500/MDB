"""
Pipeline Orchestrator — full implementation (Task 11.1).

Coordinates extraction from Odoo ERP and VTM PostgreSQL sources, loading
into MongoDB, and embedding generation for a full or incremental pipeline run.

Requirements: 1.5, 2.3, 2.4, 8.3, 8.4
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from app.core.logging import get_logger
from app.models import PipelineRunResult, PipelineStatus

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorDatabase

    from app.core.config import Settings
    from app.pipeline.adapters.odoo import OdooAdapter
    from app.pipeline.adapters.vtm import VTMAdapter
    from app.pipeline.loader import Loader

logger = get_logger(__name__)


async def _await_if_needed(value):
    """Await *value* only if it is a coroutine/awaitable.

    Motor returns coroutines; mongomock returns plain values.  This helper
    makes the orchestrator compatible with both.
    """
    if inspect.isawaitable(value):
        return await value
    return value


class PipelineOrchestrator:
    """Coordinates extraction, loading, and embedding generation.

    Constructor parameters are all optional so that ``app/main.py`` can
    instantiate ``PipelineOrchestrator()`` with no arguments (stub mode).
    When any parameter is ``None`` the orchestrator logs and returns
    immediately without performing real work, maintaining backward
    compatibility with the existing application startup.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance.
        loader: A :class:`~app.pipeline.loader.Loader` instance.
        odoo_adapter: An :class:`~app.pipeline.adapters.odoo.OdooAdapter`.
        vtm_adapter: A :class:`~app.pipeline.adapters.vtm.VTMAdapter`.
        settings: Application :class:`~app.core.config.Settings`.
    """

    def __init__(
        self,
        db: "AsyncIOMotorDatabase | None" = None,
        loader: "Loader | None" = None,
        odoo_adapter: "OdooAdapter | None" = None,
        vtm_adapter: "VTMAdapter | None" = None,
        settings: "Settings | None" = None,
    ) -> None:
        self._db = db
        self._loader = loader
        self._odoo_adapter = odoo_adapter
        self._vtm_adapter = vtm_adapter
        self._settings = settings

        # Concurrency guard — prevents overlapping pipeline runs (Req 8.4).
        self._lock = asyncio.Lock()
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, mode: Literal["full", "incremental"] = "incremental") -> dict:
        """Trigger a pipeline run.

        Returns a dict so that the admin route handler can call
        ``result.get("status")`` directly.

        Args:
            mode: ``"full"`` extracts all products; ``"incremental"`` extracts
                only products modified since the last successful run.

        Returns:
            A dict with at minimum a ``"status"`` key.  When a run is already
            in progress the dict also contains a ``"message"`` key and the
            caller should return HTTP 409.
        """
        # ── Stub mode ────────────────────────────────────────────────────────
        if self._db is None or self._loader is None:
            logger.info(
                "Pipeline run triggered (stub mode — no DB/loader configured)",
                extra={"mode": mode},
            )
            return {
                "status": "ok",
                "message": f"Pipeline run ({mode}) started (stub).",
            }

        # ── Concurrency guard (Req 8.4) ──────────────────────────────────────
        async with self._lock:
            if self._running:
                logger.warning(
                    "Pipeline trigger rejected — run already in progress",
                    extra={"mode": mode},
                )
                return {
                    "status": "already_running",
                    "message": "A pipeline run is already in progress.",
                }
            self._running = True

        try:
            return await self._execute_run(mode)
        finally:
            self._running = False

    async def get_status(self) -> dict:
        """Return the current pipeline status.

        Returns:
            A dict with ``running`` (bool) and ``latest_run`` (the most recent
            ``pipeline_runs`` document, or ``None``).
        """
        # ── Stub mode ────────────────────────────────────────────────────────
        if self._db is None:
            return {"running": self._running, "latest_run": None}

        from app.core.db import get_pipeline_runs_collection

        collection = get_pipeline_runs_collection(self._db)

        cursor = collection.find(
            {"completed_at": {"$exists": True}},
            sort=[("completed_at", -1)],
        ).limit(1)

        docs = await _await_if_needed(cursor.to_list(length=1))

        latest_run = None
        if docs:
            doc = docs[0]
            doc.pop("_id", None)
            try:
                latest_run = PipelineRunResult(**doc).model_dump(mode="json")
            except Exception:
                latest_run = {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in doc.items()
                }

        return {"running": self._running, "latest_run": latest_run}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_run(self, mode: Literal["full", "incremental"]) -> dict:
        """Perform the actual ETL run and return a result dict."""
        from app.core.db import get_pipeline_runs_collection

        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        errors: list[str] = []
        records_processed: dict[str, int] = {}

        collection = get_pipeline_runs_collection(self._db)

        # ── Insert initial pipeline_runs document ────────────────────────────
        run_doc: dict = {
            "run_id": run_id,
            "mode": mode,
            "status": "running",
            "started_at": started_at,
        }
        insert_result = collection.insert_one(run_doc)
        await _await_if_needed(insert_result)

        # ── Determine batch size ─────────────────────────────────────────────
        batch_size: int = 100
        if self._settings is not None:
            batch_size = self._settings.BATCH_SIZE

        # ── Process each source ──────────────────────────────────────────────
        adapters = []
        if self._odoo_adapter is not None:
            adapters.append(self._odoo_adapter)
        if self._vtm_adapter is not None:
            adapters.append(self._vtm_adapter)

        for adapter in adapters:
            source_name: str = adapter.source_name
            logger.info(
                "Starting source extraction",
                extra={"source": source_name, "mode": mode},
            )

            try:
                since: datetime | None = None
                if mode == "incremental":
                    since = await self._get_last_run_ts(source_name)

                source_count = await self._process_source(
                    adapter=adapter,
                    since=since,
                    batch_size=batch_size,
                )
                records_processed[source_name] = source_count

                await self._loader.record_load_timestamp(source_name, datetime.utcnow())

                logger.info(
                    "Source extraction complete",
                    extra={"source": source_name, "records": source_count},
                )

            except Exception as exc:
                ts = datetime.utcnow().isoformat()
                error_msg = (
                    f"[{ts}] Source '{source_name}' failed: {type(exc).__name__}: {exc}"
                )
                logger.error(
                    "Source extraction failed — continuing to next source",
                    extra={"source": source_name, "timestamp": ts, "error": str(exc)},
                    exc_info=True,
                )
                errors.append(error_msg)
                records_processed[source_name] = 0

        # ── Determine overall status ─────────────────────────────────────────
        completed_at = datetime.utcnow()
        if any(v > 0 for v in records_processed.values()):
            final_status = "success"
        elif errors:
            final_status = "failure"
        else:
            final_status = "success"

        # ── Update pipeline_runs document ────────────────────────────────────
        update_op = collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": final_status,
                    "completed_at": completed_at,
                    "records_processed": records_processed,
                    "errors": errors,
                }
            },
        )
        await _await_if_needed(update_op)

        logger.info(
            "Pipeline run complete",
            extra={
                "run_id": run_id,
                "status": final_status,
                "records_processed": records_processed,
                "errors": errors,
            },
        )

        return {
            "status": final_status,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "records_processed": records_processed,
            "errors": errors,
        }

    async def _process_source(
        self,
        adapter,
        since: datetime | None,
        batch_size: int,
    ) -> int:
        """Extract records from *adapter* and load them in batches."""
        total = 0
        batch: list = []

        async_gen = adapter.extract(since=since)

        async for record in async_gen:
            batch.append(record)
            if len(batch) >= batch_size:
                await self._loader.upsert_batch(batch)
                total += len(batch)
                batch = []

        if batch:
            await self._loader.upsert_batch(batch)
            total += len(batch)

        return total

    async def _get_last_run_ts(self, source_name: str) -> datetime | None:
        """Return the timestamp of the last successful load for *source_name*."""
        from app.core.db import get_pipeline_runs_collection

        collection = get_pipeline_runs_collection(self._db)

        cursor = collection.find(
            {"source": source_name, "last_successful_load": {"$exists": True}},
            sort=[("last_successful_load", -1)],
        ).limit(1)

        docs = await _await_if_needed(cursor.to_list(length=1))

        if docs:
            return docs[0].get("last_successful_load")
        return None
