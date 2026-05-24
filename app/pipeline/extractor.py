"""Extractor module for the data pipeline.

Defines the ``SourceAdapter`` Protocol that every PostgreSQL source adapter
must implement.  Concrete adapters (e.g. ``OdooAdapter``, ``VTMAdapter``)
live in ``app/pipeline/adapters/`` and are registered with the pipeline
orchestrator at startup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.models import ProductRecord


@runtime_checkable
class SourceAdapter(Protocol):
    """Interface that each PostgreSQL source must implement."""

    source_name: str

    async def extract(
        self,
        since: datetime | None = None,  # None = full extraction
    ) -> AsyncIterator[ProductRecord]: ...
