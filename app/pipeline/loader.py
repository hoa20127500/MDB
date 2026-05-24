"""
Loader — writes ``ProductRecord`` batches to MongoDB and generates embeddings.

The ``Loader`` class is responsible for:

1. Upserting product records into the ``products`` collection via Motor's
   ``bulk_write`` with ``UpdateOne(..., upsert=True)`` on ``source_id``.
2. Retrying failed bulk-write operations up to 3 times with exponential
   backoff (1 s, 2 s, 4 s) before logging and skipping the batch.
3. Calling ``EmbeddingService.encode_batch`` after a successful upsert and
   writing the resulting embeddings back to MongoDB.
4. Recording the timestamp of each successful load per source system in the
   ``pipeline_runs`` (metadata) collection.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from typing import TYPE_CHECKING

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from app.core.db import get_pipeline_runs_collection, get_products_collection
from app.core.logging import get_logger
from app.models import LoadResult, ProductRecord

if TYPE_CHECKING:
    from app.core.embedding import EmbeddingService

logger = get_logger(__name__)

# Maximum number of retry attempts for a failed bulk_write.
_MAX_RETRIES = 3


async def _bulk_write(collection, operations, ordered=False):
    """Call bulk_write on *collection*, awaiting the result only if it is a coroutine.

    Motor returns a coroutine; mongomock returns a plain BulkWriteResult.
    This helper makes the Loader compatible with both.
    """
    result = collection.bulk_write(operations, ordered=ordered)
    if inspect.isawaitable(result):
        result = await result
    return result

# Exponential backoff delays in seconds: attempt 1 → 1 s, 2 → 2 s, 3 → 4 s.
_BACKOFF_DELAYS = (1, 2, 4)


class Loader:
    """Writes product batches to MongoDB and generates vector embeddings.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance pointing at the target
            database.
        embedding_service: An ``EmbeddingService`` instance used to generate
            vector embeddings for each product after it is upserted.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        embedding_service: "EmbeddingService",
    ) -> None:
        self._db = db
        self._embedding_service = embedding_service
        self._products = get_products_collection(db)
        self._pipeline_runs = get_pipeline_runs_collection(db)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def upsert_batch(self, records: list[ProductRecord]) -> LoadResult:
        """Upsert a batch of ``ProductRecord`` objects into MongoDB.

        Each record is matched by ``source_id`` and either inserted (upsert)
        or updated in place.  After a successful bulk write the method calls
        ``EmbeddingService.encode_batch`` and writes the resulting embeddings
        back to MongoDB in a second bulk write.

        Retry policy:
        - Up to ``_MAX_RETRIES`` attempts on ``PyMongoError``.
        - Exponential backoff: 1 s, 2 s, 4 s between attempts.
        - After all retries are exhausted the batch is skipped and a
          ``LoadResult`` with ``failed=len(records)`` is returned.

        Args:
            records: List of ``ProductRecord`` instances to upsert.

        Returns:
            A ``LoadResult`` with ``upserted``, ``modified``, and ``failed``
            counts.
        """
        if not records:
            return LoadResult(upserted=0, modified=0, failed=0)

        # ── Build upsert operations ──────────────────────────────────────────
        now = datetime.utcnow()
        upsert_ops = [
            UpdateOne(
                filter={"source_id": r.source_id},
                update={
                    "$set": {
                        "source_id": r.source_id,
                        "source": r.source,
                        "name": r.name,
                        "description": r.description,
                        "category": r.category,
                        "price": r.price,
                        "availability": r.availability,
                        "metadata": r.metadata,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )
            for r in records
        ]

        # ── Attempt bulk write with retries ──────────────────────────────────
        upserted_count = 0
        modified_count = 0
        last_exc: PyMongoError | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = await _bulk_write(self._products, upsert_ops, ordered=False)
                upserted_count = result.upserted_count
                modified_count = result.modified_count
                last_exc = None
                break  # success — exit retry loop
            except PyMongoError as exc:
                last_exc = exc
                logger.warning(
                    "Bulk write attempt failed",
                    extra={
                        "attempt": attempt,
                        "max_retries": _MAX_RETRIES,
                        "batch_size": len(records),
                        "error": str(exc),
                    },
                )
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_DELAYS[attempt - 1]
                    await asyncio.sleep(delay)

        if last_exc is not None:
            # All retries exhausted — skip this batch.
            logger.error(
                "Batch upsert failed after all retries; skipping batch",
                extra={
                    "batch_size": len(records),
                    "error": str(last_exc),
                },
                exc_info=True,
            )
            return LoadResult(upserted=0, modified=0, failed=len(records))

        # ── Generate and store embeddings ────────────────────────────────────
        await self._update_embeddings(records)

        return LoadResult(
            upserted=upserted_count,
            modified=modified_count,
            failed=0,
        )

    async def record_load_timestamp(self, source_name: str, ts: datetime) -> None:
        """Record the timestamp of a successful load for *source_name*.

        Upserts a metadata document in the ``pipeline_runs`` collection with
        the structure::

            {"source": source_name, "last_successful_load": ts}

        Args:
            source_name: The name of the source system (e.g. ``"odoo"``).
            ts: The timestamp of the successful load operation.
        """
        try:
            await _bulk_write(
                self._pipeline_runs,
                [
                    UpdateOne(
                        filter={"source": source_name},
                        update={"$set": {"source": source_name, "last_successful_load": ts}},
                        upsert=True,
                    )
                ],
                ordered=False,
            )
            logger.info(
                "Recorded load timestamp",
                extra={"source": source_name, "timestamp": ts.isoformat()},
            )
        except PyMongoError as exc:
            logger.error(
                "Failed to record load timestamp",
                extra={"source": source_name, "error": str(exc)},
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _update_embeddings(self, records: list[ProductRecord]) -> None:
        """Generate embeddings for *records* and write them back to MongoDB.

        Calls ``EmbeddingService.encode_batch`` with the canonical product
        text for each record.  Documents whose encoding returned ``None`` are
        skipped (Requirement 3.5).

        Args:
            records: The same list of ``ProductRecord`` instances that were
                just upserted.
        """
        texts = [self._embedding_service.build_product_text(r) for r in records]
        product_ids = [r.source_id for r in records]

        embeddings = self._embedding_service.encode_batch(texts, product_ids=product_ids)

        model_name = self._embedding_service._model_name  # noqa: SLF001

        embedding_ops = []
        for record, emb in zip(records, embeddings):
            if emb is None:
                # Encoding failed for this product — do not store a partial
                # embedding (Requirement 3.5).
                logger.warning(
                    "Skipping embedding update for product (encoding returned None)",
                    extra={"source_id": record.source_id},
                )
                continue
            embedding_ops.append(
                UpdateOne(
                    filter={"source_id": record.source_id},
                    update={
                        "$set": {
                            "embedding": emb,
                            "embedding_model": model_name,
                        }
                    },
                )
            )

        if not embedding_ops:
            logger.warning(
                "No embeddings to write (all encoding results were None)",
                extra={"batch_size": len(records)},
            )
            return

        try:
            await _bulk_write(self._products, embedding_ops, ordered=False)
            logger.info(
                "Embedding update complete",
                extra={
                    "updated": len(embedding_ops),
                    "skipped": len(records) - len(embedding_ops),
                },
            )
        except PyMongoError as exc:
            logger.error(
                "Failed to write embeddings to MongoDB",
                extra={"error": str(exc)},
                exc_info=True,
            )
