"""
Behavior Tracker — records and retrieves user behavioral events.

The ``BehaviorTracker`` class is responsible for:

1. Recording search, click, and purchase events into the ``user_behaviors``
   MongoDB collection as ``BehaviorEvent`` documents.
2. Supporting anonymous users by storing events under a ``session_id`` with
   ``user_id`` set to ``None`` (Requirement 4.6).
3. Retrieving the most recent behavioral events for a given user, sorted by
   timestamp descending and limited to a configurable count (Requirement 5.1).
"""

from __future__ import annotations

import inspect
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.db import get_user_behaviors_collection
from app.core.logging import get_logger
from app.models import BehaviorEvent

logger = get_logger(__name__)


async def _insert_one(collection, document: dict) -> None:
    """Call insert_one on *collection*, awaiting the result only if it is a coroutine.

    Motor returns a coroutine; mongomock returns a plain InsertOneResult.
    This helper makes the BehaviorTracker compatible with both.
    """
    result = collection.insert_one(document)
    if inspect.isawaitable(result):
        await result


class BehaviorTracker:
    """Records and retrieves user behavioral events in MongoDB.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance pointing at the target
            database.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = get_user_behaviors_collection(db)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def record_search(
        self,
        user_id: str | None,
        query: str,
        ts: datetime,
        session_id: str | None = None,
    ) -> None:
        """Record a search event.

        Args:
            user_id: The authenticated user's identifier, or ``None`` for
                anonymous users.
            query: The search query text submitted by the user.
            ts: The timestamp at which the search occurred.
            session_id: The session identifier, used when ``user_id`` is
                ``None`` (anonymous user).
        """
        event = BehaviorEvent(
            user_id=user_id,
            session_id=session_id,
            event_type="search",
            query=query,
            timestamp=ts,
        )
        doc = event.model_dump()
        await _insert_one(self._collection, doc)
        logger.info(
            "Recorded search event",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "query": query,
            },
        )

    async def record_click(
        self,
        user_id: str | None,
        product_id: str,
        ts: datetime,
        session_id: str | None = None,
    ) -> None:
        """Record a product click event.

        Args:
            user_id: The authenticated user's identifier, or ``None`` for
                anonymous users.
            product_id: The identifier of the product that was clicked.
            ts: The timestamp at which the click occurred.
            session_id: The session identifier, used when ``user_id`` is
                ``None`` (anonymous user).
        """
        event = BehaviorEvent(
            user_id=user_id,
            session_id=session_id,
            event_type="click",
            product_id=product_id,
            timestamp=ts,
        )
        doc = event.model_dump()
        await _insert_one(self._collection, doc)
        logger.info(
            "Recorded click event",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "product_id": product_id,
            },
        )

    async def record_purchase(
        self,
        user_id: str | None,
        product_ids: list[str],
        ts: datetime,
        session_id: str | None = None,
    ) -> None:
        """Record a purchase event.

        Args:
            user_id: The authenticated user's identifier, or ``None`` for
                anonymous users.
            product_ids: The list of product identifiers that were purchased.
            ts: The timestamp at which the purchase occurred.
            session_id: The session identifier, used when ``user_id`` is
                ``None`` (anonymous user).
        """
        event = BehaviorEvent(
            user_id=user_id,
            session_id=session_id,
            event_type="purchase",
            product_ids=product_ids,
            timestamp=ts,
        )
        doc = event.model_dump()
        await _insert_one(self._collection, doc)
        logger.info(
            "Recorded purchase event",
            extra={
                "user_id": user_id,
                "session_id": session_id,
                "product_ids": product_ids,
            },
        )

    async def get_recent_events(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[BehaviorEvent]:
        """Retrieve the most recent behavioral events for a user.

        Queries the ``user_behaviors`` collection for documents matching
        ``{"user_id": user_id}``, sorted by ``timestamp`` descending, and
        limited to ``limit`` results.

        Args:
            user_id: The user identifier to query events for.
            limit: Maximum number of events to return.  Defaults to 50.

        Returns:
            A list of ``BehaviorEvent`` objects ordered by timestamp
            descending (most recent first).
        """
        cursor = self._collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)

        events: list[BehaviorEvent] = []

        # Handle both Motor (async cursor) and mongomock (sync cursor).
        # Motor cursors support `async for`; mongomock cursors are synchronous.
        if hasattr(cursor, "__aiter__"):
            async for doc in cursor:
                doc.pop("_id", None)
                events.append(BehaviorEvent(**doc))
        else:
            for doc in cursor:
                doc.pop("_id", None)
                events.append(BehaviorEvent(**doc))

        logger.debug(
            "Retrieved recent events",
            extra={"user_id": user_id, "count": len(events), "limit": limit},
        )
        return events
