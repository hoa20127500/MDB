"""OdooAdapter — SourceAdapter implementation for the Odoo ERP PostgreSQL database.

Queries the real Odoo schema: product_template joined with product_category.
Name and description fields are stored as JSONB {"en_US": "..."} in Odoo 16.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.logging import get_logger
from app.models import ProductRecord

logger = get_logger(__name__)

_QUERY_FULL = """
    SELECT
        pt.id,
        pt.name->>'en_US'                           AS name,
        COALESCE(pt.description_sale->>'en_US', '') AS description,
        COALESCE(pc.name, '')                        AS category,
        pt.list_price                                AS price,
        pt.active                                    AS availability,
        pt.write_date                                AS updated_at
    FROM product_template pt
    LEFT JOIN product_category pc ON pt.categ_id = pc.id
    WHERE pt.active = true
    ORDER BY pt.id
    {limit_clause}
"""

_QUERY_INCREMENTAL = """
    SELECT
        pt.id,
        pt.name->>'en_US'                           AS name,
        COALESCE(pt.description_sale->>'en_US', '') AS description,
        COALESCE(pc.name, '')                        AS category,
        pt.list_price                                AS price,
        pt.active                                    AS availability,
        pt.write_date                                AS updated_at
    FROM product_template pt
    LEFT JOIN product_category pc ON pt.categ_id = pc.id
    WHERE pt.active = true
      AND pt.write_date > :since
    ORDER BY pt.id
    {limit_clause}
"""


class OdooAdapter:
    """SourceAdapter for the Odoo ERP PostgreSQL database (GNF)."""

    source_name: str = "odoo"

    def __init__(self, dsn: str, limit: int | None = None) -> None:
        self._dsn = dsn
        self._limit = limit
        self._engine = create_async_engine(dsn, echo=False, future=True)

    async def extract(
        self,
        since: datetime | None = None,
    ) -> AsyncIterator[ProductRecord]:
        logger.info(
            "Starting extraction",
            extra={"source": self.source_name, "since": since, "limit": self._limit},
        )
        limit_clause = f"LIMIT {self._limit}" if self._limit else ""
        try:
            async with AsyncSession(self._engine) as session:
                if since is not None:
                    sql = text(_QUERY_INCREMENTAL.format(limit_clause=limit_clause))
                    result = await session.execute(sql, {"since": since})
                else:
                    sql = text(_QUERY_FULL.format(limit_clause=limit_clause))
                    result = await session.execute(sql)

                rows = result.fetchall()
                count = 0
                for row in rows:
                    name = row.name or ""
                    if not name:
                        continue
                    yield ProductRecord(
                        source_id=f"odoo:{row.id}",
                        source=self.source_name,
                        name=name,
                        description=row.description or "",
                        category=row.category or "",
                        price=float(row.price or 0),
                        availability=bool(row.availability),
                    )
                    count += 1

                logger.info(
                    "Extraction complete",
                    extra={"source": self.source_name, "records_yielded": count},
                )
        except Exception as exc:
            logger.error(
                "Extraction failed",
                extra={"source": self.source_name, "since": since, "error": str(exc)},
                exc_info=True,
            )
            raise
