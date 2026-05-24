"""
Recommendation API — context embedding builder and recommendation engine.

This module provides:

- :func:`build_context_embedding`: Combines a query embedding with weighted
  behavioral event embeddings to produce a personalized context vector
  (Requirements 5.2, 5.3, 5.4).
- :class:`RecommendationEngine`: Orchestrates context building and vector
  search to return ranked product recommendations (Requirements 6.1–6.7, 7.5).
"""

from __future__ import annotations

import inspect

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.behavior import BehaviorTracker
from app.core.db import get_products_collection
from app.core.embedding import EmbeddingService
from app.core.logging import get_logger
from app.models import BehaviorEvent, RecommendedProduct, RecommendationResponse

logger = get_logger(__name__)

# Event type weights used when constructing the context embedding.
_EVENT_WEIGHTS: dict[str, float] = {
    "purchase": 3.0,
    "click": 2.0,
    "search": 1.0,
}


def build_context_embedding(
    query_embedding: list[float],
    events: list[BehaviorEvent],
    embedding_service: EmbeddingService,
) -> list[float]:
    """Build a context-aware embedding by combining query and behavioral signals.

    The context embedding is a weighted average of the query embedding and the
    embeddings derived from the user's recent behavioral events:

    - **purchase** events contribute with weight **3.0** (one contribution per
      product in ``event.product_ids``).
    - **click** events contribute with weight **2.0** (using ``event.product_id``
      as the text to encode).
    - **search** events contribute with weight **1.0** (using ``event.query`` as
      the text to encode).

    If ``events`` is empty the function returns ``query_embedding`` unchanged
    without performing any numpy operations (Requirement 5.4).

    Args:
        query_embedding: The embedding vector for the current search query.
        events: The user's recent behavioral events, ordered most-recent first.
            May be empty.
        embedding_service: An :class:`~app.core.embedding.EmbeddingService`
            instance used to encode event texts into vectors.

    Returns:
        A list of floats representing the context embedding.  When *events* is
        empty this is identical to *query_embedding*.  Otherwise it is the
        weighted average of the query embedding and all successfully encoded
        event embeddings.
    """
    if not events:
        return query_embedding

    # Start with the query embedding at weight 1.0.
    weighted_sum: np.ndarray = np.array(query_embedding, dtype=float) * 1.0
    total_weight: float = 1.0

    for event in events:
        event_type = event.event_type
        weight = _EVENT_WEIGHTS.get(event_type, 1.0)

        if event_type == "search":
            # Use the query text; skip if absent.
            if event.query is None:
                continue
            texts_and_weights = [(event.query, weight)]

        elif event_type == "click":
            # Use the product_id as the text; skip if absent.
            if event.product_id is None:
                continue
            texts_and_weights = [(event.product_id, weight)]

        elif event_type == "purchase":
            # Each purchased product contributes separately at weight 3.0.
            texts_and_weights = [
                (pid, weight) for pid in event.product_ids
            ]

        else:
            continue

        for text, w in texts_and_weights:
            try:
                embedding = embedding_service.encode(text)
                if embedding is None:
                    continue
            except Exception:  # noqa: BLE001
                # Skip events whose text cannot be encoded.
                continue

            weighted_sum += np.array(embedding, dtype=float) * w
            total_weight += w

    return (weighted_sum / total_weight).tolist()


class RecommendationEngine:
    """Orchestrates context building and vector search.

    Encodes the user's query, optionally enriches it with behavioral context,
    and executes a MongoDB Atlas ``$vectorSearch`` aggregation to return a
    ranked list of recommended products.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance pointing at the target
            database.
        embedding_service: An :class:`~app.core.embedding.EmbeddingService`
            used to encode queries and behavioral event texts.
        behavior_tracker: A :class:`~app.api.behavior.BehaviorTracker` used
            to retrieve recent behavioral events for a user.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        embedding_service: EmbeddingService,
        behavior_tracker: BehaviorTracker,
    ) -> None:
        self._db = db
        self._embedding_service = embedding_service
        self._behavior_tracker = behavior_tracker
        self._collection = get_products_collection(db)

    async def get_recommendations(
        self,
        query: str,
        user_id: str | None,
        k: int = 10,
        min_score: float | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> RecommendationResponse:
        """Return personalized product recommendations for a search query.

        Steps:
        1. Encode the query into a vector embedding.
        2. Fetch the user's last 50 behavioral events (if ``user_id`` is set).
        3. Build a context embedding that blends the query with behavioral
           signals.
        4. Execute a ``$vectorSearch`` aggregation against the ``products``
           collection.
        5. Optionally filter results by ``min_score``.
        6. Apply page/page_size slicing.
        7. Return a :class:`~app.models.RecommendationResponse`.

        Args:
            query: The user's search query text.
            user_id: The authenticated user's identifier, or ``None`` for
                anonymous / unauthenticated requests.
            k: Number of candidates to retrieve from the vector index.
                Defaults to 10.
            min_score: Optional minimum similarity score threshold.  Products
                with a score below this value are excluded.
            page: 1-based page number for pagination.  Defaults to 1.
            page_size: Number of results per page.  Defaults to 10.

        Returns:
            A :class:`~app.models.RecommendationResponse` containing the
            ranked, paginated list of recommended products.
        """
        # 1. Encode the query.
        query_embedding: list[float] = self._embedding_service.encode(query)

        # 2. Fetch recent behavioral events (skip for anonymous users).
        events: list[BehaviorEvent] = []
        if user_id is not None:
            events = await self._behavior_tracker.get_recent_events(
                user_id, limit=50
            )

        # 3. Build context embedding (falls back to query_embedding when
        #    events is empty — Requirement 5.4).
        context_embedding: list[float] = build_context_embedding(
            query_embedding, events, self._embedding_service
        )

        # 4. Build the $vectorSearch aggregation pipeline.
        pipeline: list[dict] = [
            {
                "$vectorSearch": {
                    "index": "product_embedding_index",
                    "path": "embedding",
                    "queryVector": context_embedding,
                    "numCandidates": k * 10,
                    "limit": k,
                }
            },
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
        ]

        # 5. Apply min_score filter when configured (Requirement 6.6).
        if min_score is not None:
            pipeline.append(
                {
                    "$match": {
                        "similarity_score": {"$gte": min_score}
                    }
                }
            )

        pipeline.append(
            {
                "$project": {
                    "source_id": 1,
                    "name": 1,
                    "description": 1,
                    "category": 1,
                    "price": 1,
                    "availability": 1,
                    "similarity_score": 1,
                }
            }
        )

        # Execute the aggregation and collect all results.
        cursor = self._collection.aggregate(pipeline)
        all_results: list[RecommendedProduct] = []

        # Handle both Motor (async cursor) and mongomock (sync cursor).
        if hasattr(cursor, "__aiter__"):
            async for doc in cursor:
                doc.pop("_id", None)
                all_results.append(RecommendedProduct(**doc))
        else:
            # mongomock returns a synchronous cursor; it may also be a
            # coroutine when the aggregate call itself is awaitable.
            if inspect.isawaitable(cursor):
                cursor = await cursor
            for doc in cursor:
                doc.pop("_id", None)
                all_results.append(RecommendedProduct(**doc))

        # Results from $vectorSearch are already ranked descending by score.
        # Sort explicitly to guarantee the invariant (Requirement 6.7).
        all_results.sort(key=lambda p: p.similarity_score, reverse=True)

        total = len(all_results)

        # 6. Apply pagination slicing (Requirement 7.5).
        offset = (page - 1) * page_size
        page_results = all_results[offset: offset + page_size]

        logger.info(
            "Recommendations generated",
            extra={
                "query": query,
                "user_id": user_id,
                "k": k,
                "total": total,
                "page": page,
                "page_size": page_size,
                "returned": len(page_results),
            },
        )

        # 7. Return the response.
        return RecommendationResponse(
            query=query,
            results=page_results,
            total=total,
            page=page,
            page_size=page_size,
        )
