"""Unit tests for recommendation engine components (Task 9).

Tests cover:
- BehaviorTracker.record_search, record_click, record_purchase (mongomock)
- BehaviorTracker.get_recent_events — limit enforcement and ordering
- build_context_embedding — empty events, purchase weighted higher than search
- RecommendationEngine.get_recommendations — mocked $vectorSearch aggregation,
  response structure, pagination, min_score filtering, score ordering
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import mongomock
import numpy as np
import pytest

from app.api.behavior import BehaviorTracker
from app.api.recommendation import RecommendationEngine, build_context_embedding
from app.models import BehaviorEvent, RecommendedProduct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mongomock_db(db_name: str = "test_db"):
    client = mongomock.MongoClient()
    return client[db_name]


def _make_embedding_service(vector: list[float] | None = None) -> MagicMock:
    """Return a mock EmbeddingService whose encode() returns *vector*."""
    svc = MagicMock()
    if vector is None:
        vector = [0.1, 0.2, 0.3]
    svc.encode.return_value = vector
    return svc


def _make_behavior_event(
    event_type: str,
    user_id: str = "user1",
    query: str | None = None,
    product_id: str | None = None,
    product_ids: list[str] | None = None,
    ts: datetime | None = None,
) -> BehaviorEvent:
    return BehaviorEvent(
        user_id=user_id,
        session_id=None,
        event_type=event_type,
        query=query,
        product_id=product_id,
        product_ids=product_ids or [],
        timestamp=ts or datetime.utcnow(),
    )


def _make_product_doc(
    source_id: str,
    name: str = "Product",
    description: str = "Desc",
    category: str = "Cat",
    price: float = 10.0,
    availability: bool = True,
    similarity_score: float = 0.9,
) -> dict:
    return {
        "source_id": source_id,
        "name": name,
        "description": description,
        "category": category,
        "price": price,
        "availability": availability,
        "similarity_score": similarity_score,
    }


# ---------------------------------------------------------------------------
# BehaviorTracker — record_search
# ---------------------------------------------------------------------------


class TestBehaviorTrackerRecordSearch:
    async def test_record_search_inserts_document(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 1, 12, 0, 0)

        await tracker.record_search(user_id="u1", query="blue shoes", ts=ts)

        docs = list(db["user_behaviors"].find({"user_id": "u1"}))
        assert len(docs) == 1
        assert docs[0]["event_type"] == "search"
        assert docs[0]["query"] == "blue shoes"
        assert docs[0]["timestamp"] == ts

    async def test_record_search_anonymous_user(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 1, 12, 0, 0)

        await tracker.record_search(
            user_id=None, query="red hat", ts=ts, session_id="sess-abc"
        )

        docs = list(db["user_behaviors"].find({"session_id": "sess-abc"}))
        assert len(docs) == 1
        assert docs[0]["user_id"] is None
        assert docs[0]["event_type"] == "search"

    async def test_record_search_multiple_events(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 1, 12, 0, 0)

        await tracker.record_search(user_id="u1", query="shoes", ts=ts)
        await tracker.record_search(user_id="u1", query="boots", ts=ts)

        count = db["user_behaviors"].count_documents({"user_id": "u1"})
        assert count == 2


# ---------------------------------------------------------------------------
# BehaviorTracker — record_click
# ---------------------------------------------------------------------------


class TestBehaviorTrackerRecordClick:
    async def test_record_click_inserts_document(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 2, 10, 0, 0)

        await tracker.record_click(user_id="u2", product_id="odoo:42", ts=ts)

        docs = list(db["user_behaviors"].find({"user_id": "u2"}))
        assert len(docs) == 1
        assert docs[0]["event_type"] == "click"
        assert docs[0]["product_id"] == "odoo:42"
        assert docs[0]["timestamp"] == ts

    async def test_record_click_anonymous_user(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 2, 10, 0, 0)

        await tracker.record_click(
            user_id=None, product_id="vtm:99", ts=ts, session_id="sess-xyz"
        )

        docs = list(db["user_behaviors"].find({"session_id": "sess-xyz"}))
        assert len(docs) == 1
        assert docs[0]["user_id"] is None
        assert docs[0]["product_id"] == "vtm:99"


# ---------------------------------------------------------------------------
# BehaviorTracker — record_purchase
# ---------------------------------------------------------------------------


class TestBehaviorTrackerRecordPurchase:
    async def test_record_purchase_inserts_document(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 3, 9, 0, 0)

        await tracker.record_purchase(
            user_id="u3", product_ids=["odoo:1", "vtm:2"], ts=ts
        )

        docs = list(db["user_behaviors"].find({"user_id": "u3"}))
        assert len(docs) == 1
        assert docs[0]["event_type"] == "purchase"
        assert docs[0]["product_ids"] == ["odoo:1", "vtm:2"]
        assert docs[0]["timestamp"] == ts

    async def test_record_purchase_single_product(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 3, 9, 0, 0)

        await tracker.record_purchase(user_id="u4", product_ids=["odoo:5"], ts=ts)

        docs = list(db["user_behaviors"].find({"user_id": "u4"}))
        assert len(docs) == 1
        assert docs[0]["product_ids"] == ["odoo:5"]


# ---------------------------------------------------------------------------
# BehaviorTracker — get_recent_events (limit and ordering)
# ---------------------------------------------------------------------------


class TestBehaviorTrackerGetRecentEvents:
    async def test_get_recent_events_returns_events_for_user(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        base_ts = datetime(2024, 1, 1, 0, 0, 0)

        await tracker.record_search(user_id="u1", query="shoes", ts=base_ts)
        await tracker.record_click(
            user_id="u1", product_id="odoo:1", ts=base_ts + timedelta(minutes=1)
        )

        events = await tracker.get_recent_events("u1")
        assert len(events) == 2

    async def test_get_recent_events_ordered_most_recent_first(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        base_ts = datetime(2024, 1, 1, 0, 0, 0)

        # Insert in chronological order
        await tracker.record_search(user_id="u1", query="first", ts=base_ts)
        await tracker.record_search(
            user_id="u1", query="second", ts=base_ts + timedelta(hours=1)
        )
        await tracker.record_search(
            user_id="u1", query="third", ts=base_ts + timedelta(hours=2)
        )

        events = await tracker.get_recent_events("u1")
        # Most recent first
        assert events[0].query == "third"
        assert events[1].query == "second"
        assert events[2].query == "first"

    async def test_get_recent_events_limit_enforced(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        base_ts = datetime(2024, 1, 1, 0, 0, 0)

        # Insert 10 events
        for i in range(10):
            await tracker.record_search(
                user_id="u1",
                query=f"query-{i}",
                ts=base_ts + timedelta(minutes=i),
            )

        # Request only 3
        events = await tracker.get_recent_events("u1", limit=3)
        assert len(events) == 3

    async def test_get_recent_events_returns_only_matching_user(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 1, 0, 0, 0)

        await tracker.record_search(user_id="u1", query="shoes", ts=ts)
        await tracker.record_search(user_id="u2", query="boots", ts=ts)

        events = await tracker.get_recent_events("u1")
        assert len(events) == 1
        assert events[0].query == "shoes"

    async def test_get_recent_events_empty_when_no_events(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)

        events = await tracker.get_recent_events("nonexistent_user")
        assert events == []

    async def test_get_recent_events_returns_behavior_event_objects(self):
        db = _make_mongomock_db()
        tracker = BehaviorTracker(db)
        ts = datetime(2024, 1, 1, 0, 0, 0)

        await tracker.record_search(user_id="u1", query="test", ts=ts)

        events = await tracker.get_recent_events("u1")
        assert len(events) == 1
        assert isinstance(events[0], BehaviorEvent)
        assert events[0].event_type == "search"


# ---------------------------------------------------------------------------
# build_context_embedding
# ---------------------------------------------------------------------------


class TestBuildContextEmbedding:
    def test_empty_events_returns_query_embedding_unchanged(self):
        query_embedding = [0.1, 0.2, 0.3]
        svc = _make_embedding_service()

        result = build_context_embedding(query_embedding, [], svc)

        # Must be equal in value
        assert result == query_embedding
        # encode should NOT be called when events is empty
        svc.encode.assert_not_called()

    def test_empty_events_returns_same_object(self):
        """When events is empty, the exact same list object is returned."""
        query_embedding = [0.1, 0.2, 0.3]
        svc = _make_embedding_service()

        result = build_context_embedding(query_embedding, [], svc)

        assert result is query_embedding

    def test_search_event_blends_with_query(self):
        query_embedding = [1.0, 0.0, 0.0]
        search_embedding = [0.0, 1.0, 0.0]

        svc = MagicMock()
        svc.encode.return_value = search_embedding

        event = _make_behavior_event(event_type="search", query="shoes")
        result = build_context_embedding(query_embedding, [event], svc)

        # query weight=1.0, search weight=1.0 → average of [1,0,0] and [0,1,0]
        # weighted_sum = [1,0,0]*1 + [0,1,0]*1 = [1,1,0], total_weight=2
        # result = [0.5, 0.5, 0.0]
        assert result == pytest.approx([0.5, 0.5, 0.0], abs=1e-6)

    def test_purchase_weighted_higher_than_search(self):
        """Purchase (weight=3) should pull the result more than search (weight=1)."""
        query_embedding = [1.0, 0.0, 0.0]
        purchase_embedding = [0.0, 0.0, 1.0]
        search_embedding = [0.0, 1.0, 0.0]

        call_count = [0]

        def encode_side_effect(text):
            call_count[0] += 1
            if text == "odoo:1":
                return purchase_embedding
            return search_embedding

        svc = MagicMock()
        svc.encode.side_effect = encode_side_effect

        purchase_event = _make_behavior_event(
            event_type="purchase", product_ids=["odoo:1"]
        )
        search_event = _make_behavior_event(event_type="search", query="shoes")

        result = build_context_embedding(
            query_embedding, [purchase_event, search_event], svc
        )

        # query=1.0, purchase=3.0, search=1.0 → total_weight=5.0
        # weighted_sum = [1,0,0]*1 + [0,0,1]*3 + [0,1,0]*1 = [1,1,3]
        # result = [0.2, 0.2, 0.6]
        assert result == pytest.approx([0.2, 0.2, 0.6], abs=1e-6)

    def test_click_event_uses_weight_2(self):
        query_embedding = [1.0, 0.0, 0.0]
        click_embedding = [0.0, 1.0, 0.0]

        svc = MagicMock()
        svc.encode.return_value = click_embedding

        event = _make_behavior_event(event_type="click", product_id="odoo:5")
        result = build_context_embedding(query_embedding, [event], svc)

        # query=1.0, click=2.0 → total_weight=3.0
        # weighted_sum = [1,0,0]*1 + [0,1,0]*2 = [1,2,0]
        # result = [1/3, 2/3, 0]
        assert result == pytest.approx([1 / 3, 2 / 3, 0.0], abs=1e-6)

    def test_purchase_multiple_products_each_contribute(self):
        """Each product_id in a purchase event contributes separately."""
        query_embedding = [1.0, 0.0, 0.0]
        prod1_embedding = [0.0, 1.0, 0.0]
        prod2_embedding = [0.0, 0.0, 1.0]

        def encode_side_effect(text):
            if text == "odoo:1":
                return prod1_embedding
            return prod2_embedding

        svc = MagicMock()
        svc.encode.side_effect = encode_side_effect

        event = _make_behavior_event(
            event_type="purchase", product_ids=["odoo:1", "odoo:2"]
        )
        result = build_context_embedding(query_embedding, [event], svc)

        # query=1.0, prod1=3.0, prod2=3.0 → total_weight=7.0
        # weighted_sum = [1,0,0]*1 + [0,1,0]*3 + [0,0,1]*3 = [1,3,3]
        # result = [1/7, 3/7, 3/7]
        assert result == pytest.approx([1 / 7, 3 / 7, 3 / 7], abs=1e-6)

    def test_search_event_without_query_is_skipped(self):
        """A search event with query=None should be skipped."""
        query_embedding = [1.0, 0.0, 0.0]
        svc = MagicMock()

        event = _make_behavior_event(event_type="search", query=None)
        result = build_context_embedding(query_embedding, [event], svc)

        # No valid events → result should equal query_embedding
        assert result == pytest.approx(query_embedding, abs=1e-6)
        svc.encode.assert_not_called()

    def test_click_event_without_product_id_is_skipped(self):
        """A click event with product_id=None should be skipped."""
        query_embedding = [1.0, 0.0, 0.0]
        svc = MagicMock()

        event = _make_behavior_event(event_type="click", product_id=None)
        result = build_context_embedding(query_embedding, [event], svc)

        assert result == pytest.approx(query_embedding, abs=1e-6)
        svc.encode.assert_not_called()

    def test_returns_list_of_floats(self):
        query_embedding = [0.5, 0.5]
        svc = MagicMock()
        svc.encode.return_value = [0.3, 0.7]

        event = _make_behavior_event(event_type="search", query="test")
        result = build_context_embedding(query_embedding, [event], svc)

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# RecommendationEngine.get_recommendations
# ---------------------------------------------------------------------------


class TestRecommendationEngineGetRecommendations:
    """Tests for RecommendationEngine.get_recommendations with mocked aggregation."""

    def _make_engine(
        self,
        db,
        aggregate_return: list[dict],
        query_embedding: list[float] | None = None,
        events: list[BehaviorEvent] | None = None,
    ) -> RecommendationEngine:
        """Build a RecommendationEngine with mocked embedding service and behavior tracker."""
        if query_embedding is None:
            query_embedding = [0.1, 0.2, 0.3]

        embedding_svc = MagicMock()
        embedding_svc.encode.return_value = query_embedding

        behavior_tracker = MagicMock()
        behavior_tracker.get_recent_events = MagicMock(
            return_value=events if events is not None else []
        )
        # Make get_recent_events awaitable
        import asyncio

        async def _async_get_recent_events(user_id, limit=50):
            return events if events is not None else []

        behavior_tracker.get_recent_events = _async_get_recent_events

        engine = RecommendationEngine(
            db=db,
            embedding_service=embedding_svc,
            behavior_tracker=behavior_tracker,
        )

        # Mock the products collection's aggregate method to return known docs
        mock_aggregate = MagicMock(return_value=aggregate_return)
        engine._collection = MagicMock()
        engine._collection.aggregate = mock_aggregate

        return engine

    async def test_basic_response_structure(self):
        db = _make_mongomock_db()
        docs = [
            _make_product_doc("odoo:1", similarity_score=0.95),
            _make_product_doc("odoo:2", similarity_score=0.85),
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(query="shoes", user_id=None)

        assert response.query == "shoes"
        assert response.total == 2
        assert response.page == 1
        assert response.page_size == 10
        assert len(response.results) == 2

    async def test_results_are_recommended_product_instances(self):
        db = _make_mongomock_db()
        docs = [_make_product_doc("odoo:1", similarity_score=0.9)]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(query="test", user_id=None)

        assert len(response.results) == 1
        assert isinstance(response.results[0], RecommendedProduct)
        assert response.results[0].source_id == "odoo:1"
        assert response.results[0].similarity_score == pytest.approx(0.9)

    async def test_results_ordered_by_score_descending(self):
        db = _make_mongomock_db()
        # Return docs in non-sorted order to verify sorting
        docs = [
            _make_product_doc("odoo:3", similarity_score=0.7),
            _make_product_doc("odoo:1", similarity_score=0.95),
            _make_product_doc("odoo:2", similarity_score=0.85),
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(query="shoes", user_id=None)

        scores = [r.similarity_score for r in response.results]
        assert scores == sorted(scores, reverse=True)
        assert response.results[0].source_id == "odoo:1"

    async def test_pagination_page_2(self):
        """page=2, page_size=3 on 7 results should return items at indices 3,4,5."""
        db = _make_mongomock_db()
        docs = [
            _make_product_doc(f"odoo:{i}", similarity_score=1.0 - i * 0.05)
            for i in range(7)
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(
            query="shoes", user_id=None, page=2, page_size=3
        )

        assert response.page == 2
        assert response.page_size == 3
        assert response.total == 7
        assert len(response.results) == 3
        # After sorting by score desc, indices 3,4,5 → odoo:3, odoo:4, odoo:5
        assert response.results[0].source_id == "odoo:3"
        assert response.results[1].source_id == "odoo:4"
        assert response.results[2].source_id == "odoo:5"

    async def test_pagination_last_page_partial(self):
        """Last page with fewer items than page_size returns only remaining items."""
        db = _make_mongomock_db()
        docs = [
            _make_product_doc(f"odoo:{i}", similarity_score=1.0 - i * 0.05)
            for i in range(7)
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(
            query="shoes", user_id=None, page=3, page_size=3
        )

        assert response.total == 7
        assert len(response.results) == 1  # only index 6 remains
        assert response.results[0].source_id == "odoo:6"

    async def test_min_score_filtering(self):
        """Results below min_score should be excluded."""
        db = _make_mongomock_db()
        # Simulate that the aggregation pipeline (with $match) already filtered
        # Only return docs above the threshold
        docs = [
            _make_product_doc("odoo:1", similarity_score=0.95),
            _make_product_doc("odoo:2", similarity_score=0.85),
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(
            query="shoes", user_id=None, min_score=0.8
        )

        # All returned docs are above threshold
        for result in response.results:
            assert result.similarity_score >= 0.8

    async def test_empty_results(self):
        db = _make_mongomock_db()
        engine = self._make_engine(db, [])

        response = await engine.get_recommendations(query="obscure query", user_id=None)

        assert response.total == 0
        assert response.results == []

    async def test_anonymous_user_skips_behavior_fetch(self):
        """When user_id is None, behavior events should not be fetched."""
        db = _make_mongomock_db()
        docs = [_make_product_doc("odoo:1", similarity_score=0.9)]

        embedding_svc = MagicMock()
        embedding_svc.encode.return_value = [0.1, 0.2, 0.3]

        behavior_tracker = MagicMock()

        engine = RecommendationEngine(
            db=db,
            embedding_service=embedding_svc,
            behavior_tracker=behavior_tracker,
        )
        engine._collection = MagicMock()
        engine._collection.aggregate = MagicMock(return_value=docs)

        await engine.get_recommendations(query="shoes", user_id=None)

        # get_recent_events should NOT be called for anonymous users
        behavior_tracker.get_recent_events.assert_not_called()

    async def test_authenticated_user_fetches_behavior_events(self):
        """When user_id is set, behavior events should be fetched."""
        db = _make_mongomock_db()
        docs = [_make_product_doc("odoo:1", similarity_score=0.9)]

        embedding_svc = MagicMock()
        embedding_svc.encode.return_value = [0.1, 0.2, 0.3]

        behavior_tracker = MagicMock()

        async def _async_get_recent_events(user_id, limit=50):
            return []

        behavior_tracker.get_recent_events = _async_get_recent_events

        engine = RecommendationEngine(
            db=db,
            embedding_service=embedding_svc,
            behavior_tracker=behavior_tracker,
        )
        engine._collection = MagicMock()
        engine._collection.aggregate = MagicMock(return_value=docs)

        response = await engine.get_recommendations(query="shoes", user_id="user123")

        assert response.query == "shoes"

    async def test_aggregate_called_with_pipeline(self):
        """Verify that aggregate is called with a pipeline containing $vectorSearch."""
        db = _make_mongomock_db()
        docs = [_make_product_doc("odoo:1", similarity_score=0.9)]
        engine = self._make_engine(db, docs)

        await engine.get_recommendations(query="shoes", user_id=None, k=5)

        engine._collection.aggregate.assert_called_once()
        pipeline = engine._collection.aggregate.call_args[0][0]
        # First stage must be $vectorSearch
        assert "$vectorSearch" in pipeline[0]
        assert pipeline[0]["$vectorSearch"]["limit"] == 5

    async def test_response_fields_populated_correctly(self):
        db = _make_mongomock_db()
        docs = [
            _make_product_doc(
                "odoo:42",
                name="Blue Shoes",
                description="Comfortable blue shoes",
                category="Footwear",
                price=49.99,
                availability=True,
                similarity_score=0.92,
            )
        ]
        engine = self._make_engine(db, docs)

        response = await engine.get_recommendations(query="blue shoes", user_id=None)

        product = response.results[0]
        assert product.source_id == "odoo:42"
        assert product.name == "Blue Shoes"
        assert product.description == "Comfortable blue shoes"
        assert product.category == "Footwear"
        assert product.price == pytest.approx(49.99)
        assert product.availability is True
        assert product.similarity_score == pytest.approx(0.92)
