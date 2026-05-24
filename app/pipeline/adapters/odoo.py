"""OdooAdapter — SourceAdapter implementation for the Odoo ERP PostgreSQL database.

Connects to the Odoo PostgreSQL instance via SQLAlchemy async engine and
yields :class:`~app.models.ProductRecord` objects for each product row.

Supports both full extraction (``since=None``) and incremental extraction
(``since=<datetime>``) by filtering on the ``updated_at`` column.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.logging import get_logger
from app.models import ProductRecord

logger = get_logger(__name__)

# SQL query for full extraction
_QUERY_FULL = text(
    """
    SELECT
        id,
        name,
        description,
        category,
        price,
        availability,
        updated_at
    FROM products
    ORDER BY id
    """
)

# SQL query for incremental extraction (products updated after a given timestamp)
_QUERY_INCREMENTAL = text(
    """
    SELECT
        id,
        name,
        description,
        category,
        price,
        availability,
        updated_at
    FROM products
    WHERE updated_at > :since
    ORDER BY id
    """
)


class OdooAdapter:
    """Concrete :class:`~app.pipeline.extractor.SourceAdapter` for Odoo ERP.

    Connects to the Odoo PostgreSQL database using a SQLAlchemy async engine
    and yields :class:`~app.models.ProductRecord` objects.

    Args:
        dsn: SQLAlchemy-compatible async DSN, e.g.
            ``"postgresql+asyncpg://user:pass@host:5432/odoo"``.
    """

    source_name: str = "odoo"

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._engine = create_async_engine(dsn, echo=False, future=True)

    async def extract(
        self,
        since: datetime | None = None,
    ) -> AsyncIterator[ProductRecord]:
        """Yield :class:`~app.models.ProductRecord` objects from the Odoo database.

        Args:
            since: When provided, only products with ``updated_at > since``
                are returned (incremental mode).  When ``None``, all products
                are returned (full extraction mode).

        Yields:
            :class:`~app.models.ProductRecord` for each product row.

        Raises:
            Exception: Re-raises any exception that occurs during extraction
                after logging it, so the pipeline orchestrator can continue
                with remaining sources.
        """
        logger.info(
            "Starting extraction",
            extra={"source": self.source_name, "since": since},
        )

        try:
            async with AsyncSession(self._engine) as session:
                if since is not None:
                    result = await session.execute(
                        _QUERY_INCREMENTAL, {"since": since}
                    )
                else:
                    result = await session.execute(_QUERY_FULL)

                rows = result.fetchall()
                count = 0

                for row in rows:
                    yield ProductRecord(
                        source_id=f"odoo:{row.id}",
                        source=self.source_name,
                        name=row.name or "",
                        description=row.description or "",
                        category=row.category or "",
                        price=float(row.price),
                        availability=bool(row.availability),
                    )
                    count += 1

                logger.info(
                    "Extraction complete",
                    extra={
                        "source": self.source_name,
                        "since": since,
                        "records_yielded": count,
                    },
                )

        except Exception as exc:
            logger.error(
                "Extraction failed",
                extra={
                    "source": self.source_name,
                    "since": since,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise
