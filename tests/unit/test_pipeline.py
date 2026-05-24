"""Unit tests for pipeline components (Tasks 1-5).

Tests cover:
- app/models.py — Pydantic models and validators
- app/core/config.py — Settings loading
- app/core/db.py — MongoDB client factory and collection accessors
- app/core/logging.py — JSON logging setup
- app/core/embedding.py — EmbeddingService (model mocked)
- app/pipeline/extractor.py — SourceAdapter Protocol
- app/pipeline/loader.py — Loader (mongomock)
- app/pipeline/adapters/odoo.py — OdooAdapter (SQLAlchemy mocked)
- app/pipeline/adapters/vtm.py — VTMAdapter (SQLAlchemy mocked)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import mongomock
import numpy as np
import pytest
from pydantic import ValidationError

from app.models import (
    BehaviorEvent,
    LoadResult,
    ProductRecord,
    RecommendationRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(
    source_id: str = "odoo:123",
    source: str = "odoo",
    name: str = "Widget",
    description: str = "A great widget",
    category: str = "Widgets",
    price: float = 9.99,
    availability: bool = True,
) -> ProductRecord:
    return ProductRecord(
        source_id=source_id,
        source=source,
        name=name,
        description=description,
        category=category,
        price=price,
        availability=availability,
    )


def _make_mongomock_db(db_name: str = "test_db"):
    client = mongomock.MongoClient()
    return client[db_name]


# ---------------------------------------------------------------------------
# 1. ProductRecord validation
# ---------------------------------------------------------------------------


class TestProductRecordValidation:
    def test_valid_source_id_passes(self):
        record = _make_product(source_id="odoo:12345")
        assert record.source_id == "odoo:12345"

    def test_valid_vtm_source_id_passes(self):
        record = _make_product(source_id="vtm:abc-99", source="vtm")
        assert record.source_id == "vtm:abc-99"

    def test_invalid_source_id_no_colon_raises(self):
        with pytest.raises(ValidationError):
            _make_product(source_id="odoo12345")

    def test_invalid_source_id_uppercase_prefix_raises(self):
        with pytest.raises(ValidationError):
            _make_product(source_id="ODOO:123")

    def test_invalid_source_id_whitespace_in_id_raises(self):
        with pytest.raises(ValidationError):
            _make_product(source_id="odoo:123 456")

    def test_invalid_source_id_empty_raises(self):
        with pytest.raises(ValidationError):
            _make_product(source_id="")

    def test_invalid_source_id_colon_only_raises(self):
        # "odoo:" has nothing after the colon — should fail \S+
        with pytest.raises(ValidationError):
            _make_product(source_id="odoo:")


# ---------------------------------------------------------------------------
# 2. RecommendationRequest validation
# ---------------------------------------------------------------------------


class TestRecommendationRequestValidation:
    def test_valid_request_passes(self):
        req = RecommendationRequest(query="blue shoes", k=5, page=1)
        assert req.query == "blue shoes"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="")

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="   ")

    def test_k_less_than_1_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="shoes", k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="shoes", k=-1)

    def test_page_less_than_1_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="shoes", page=0)

    def test_page_size_less_than_1_raises(self):
        with pytest.raises(ValidationError):
            RecommendationRequest(query="shoes", page_size=0)

    def test_defaults_are_valid(self):
        req = RecommendationRequest(query="test")
        assert req.k == 10
        assert req.page == 1
        assert req.page_size == 10


# ---------------------------------------------------------------------------
# 3. BehaviorEvent validation
# ---------------------------------------------------------------------------


class TestBehaviorEventValidation:
    def test_valid_search_event(self):
        event = BehaviorEvent(
            user_id="u1",
            session_id=None,
            event_type="search",
            query="red shoes",
            timestamp=datetime.utcnow(),
        )
        assert event.event_type == "search"

    def test_valid_click_event(self):
        event = BehaviorEvent(
            user_id=None,
            session_id="sess-abc",
            event_type="click",
            product_id="odoo:42",
            timestamp=datetime.utcnow(),
        )
        assert event.event_type == "click"

    def test_valid_purchase_event(self):
        event = BehaviorEvent(
            user_id="u2",
            session_id=None,
            event_type="purchase",
            product_ids=["odoo:1", "vtm:2"],
            timestamp=datetime.utcnow(),
        )
        assert event.event_type == "purchase"

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValidationError):
            BehaviorEvent(
                user_id="u1",
                session_id=None,
                event_type="view",  # not a valid Literal
                timestamp=datetime.utcnow(),
            )


# ---------------------------------------------------------------------------
# 4-6. EmbeddingService
# ---------------------------------------------------------------------------


class TestEmbeddingService:
    """Tests for EmbeddingService — model is mocked to avoid loading weights."""

    def _make_service(self, mock_model: MagicMock):
        """Construct an EmbeddingService with a pre-injected mock model."""
        from app.core.embedding import EmbeddingService

        with patch("app.core.embedding.SentenceTransformer", return_value=mock_model):
            svc = EmbeddingService(model_name="mock-model")
        return svc

    def test_build_product_text_concatenates_fields(self):
        from app.core.embedding import EmbeddingService

        mock_model = MagicMock()
        svc = self._make_service(mock_model)

        record = _make_product(name="Widget", description="A great widget", category="Widgets")
        text = svc.build_product_text(record)
        assert text == "Widget A great widget Widgets"

    def test_encode_returns_list_of_floats(self):
        from app.core.embedding import EmbeddingService

        fake_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vector

        svc = self._make_service(mock_model)
        result = svc.encode("hello world")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert result == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)

    def test_encode_batch_returns_list_of_floats_per_item(self):
        from app.core.embedding import EmbeddingService

        fake_batch = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_batch

        svc = self._make_service(mock_model)
        results = svc.encode_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0] == pytest.approx([0.1, 0.2], abs=1e-5)
        assert results[1] == pytest.approx([0.3, 0.4], abs=1e-5)

    def test_encode_batch_returns_none_for_failed_items(self):
        from app.core.embedding import EmbeddingService

        mock_model = MagicMock()
        # Simulate the entire batch call raising an exception
        mock_model.encode.side_effect = RuntimeError("model exploded")

        svc = self._make_service(mock_model)
        results = svc.encode_batch(["text one", "text two"])

        assert results == [None, None]

    def test_encode_batch_empty_input_returns_empty(self):
        from app.core.embedding import EmbeddingService

        mock_model = MagicMock()
        svc = self._make_service(mock_model)
        results = svc.encode_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# 7. OdooAdapter.source_name
# ---------------------------------------------------------------------------


class TestOdooAdapter:
    def test_source_name_is_odoo(self):
        from app.pipeline.adapters.odoo import OdooAdapter

        # Patch create_async_engine so no real DB connection is attempted
        with patch("app.pipeline.adapters.odoo.create_async_engine"):
            adapter = OdooAdapter(dsn="postgresql+asyncpg://user:pass@localhost/odoo")

        assert adapter.source_name == "odoo"

    def test_source_name_class_attribute(self):
        from app.pipeline.adapters.odoo import OdooAdapter

        assert OdooAdapter.source_name == "odoo"


# ---------------------------------------------------------------------------
# 8. VTMAdapter.source_name
# ---------------------------------------------------------------------------


class TestVTMAdapter:
    def test_source_name_is_vtm(self):
        from app.pipeline.adapters.vtm import VTMAdapter

        with patch("app.pipeline.adapters.vtm.create_async_engine"):
            adapter = VTMAdapter(dsn="postgresql+asyncpg://user:pass@localhost/vtm")

        assert adapter.source_name == "vtm"

    def test_source_name_class_attribute(self):
        from app.pipeline.adapters.vtm import VTMAdapter

        assert VTMAdapter.source_name == "vtm"


# ---------------------------------------------------------------------------
# 9. Loader.upsert_batch — idempotence with mongomock
# ---------------------------------------------------------------------------


class TestLoaderUpsertBatch:
    """Tests for Loader.upsert_batch using mongomock."""

    def _make_loader(self, db):
        from app.pipeline.loader import Loader

        mock_embedding_svc = MagicMock()
        # encode_batch returns a list of None so embedding writes are skipped
        mock_embedding_svc.encode_batch.return_value = [None]
        mock_embedding_svc.build_product_text.return_value = "Widget A great widget Widgets"
        mock_embedding_svc._model_name = "mock-model"
        return Loader(db=db, embedding_service=mock_embedding_svc)

    @pytest.mark.asyncio
    async def test_upsert_idempotence(self):
        """Upserting the same record twice should leave exactly one document."""
        db = _make_mongomock_db()
        loader = self._make_loader(db)

        record = _make_product(source_id="odoo:999")

        # First upsert
        result1 = await loader.upsert_batch([record])
        # Second upsert (same source_id)
        result2 = await loader.upsert_batch([record])

        count = db["products"].count_documents({})
        assert count == 1, f"Expected 1 document, got {count}"

    @pytest.mark.asyncio
    async def test_upsert_multiple_distinct_records(self):
        """Upserting N distinct records should produce N documents."""
        db = _make_mongomock_db()

        mock_embedding_svc = MagicMock()
        mock_embedding_svc.encode_batch.return_value = [None, None, None]
        mock_embedding_svc.build_product_text.return_value = "text"
        mock_embedding_svc._model_name = "mock-model"

        from app.pipeline.loader import Loader

        loader = Loader(db=db, embedding_service=mock_embedding_svc)

        records = [
            _make_product(source_id="odoo:1"),
            _make_product(source_id="odoo:2"),
            _make_product(source_id="odoo:3"),
        ]
        await loader.upsert_batch(records)

        count = db["products"].count_documents({})
        assert count == 3

    @pytest.mark.asyncio
    async def test_upsert_empty_batch_returns_zero_counts(self):
        db = _make_mongomock_db()
        loader = self._make_loader(db)

        result = await loader.upsert_batch([])
        assert result.upserted == 0
        assert result.modified == 0
        assert result.failed == 0


# ---------------------------------------------------------------------------
# 10. Loader.record_load_timestamp — with mongomock
# ---------------------------------------------------------------------------


class TestLoaderRecordLoadTimestamp:
    @pytest.mark.asyncio
    async def test_record_load_timestamp_creates_document(self):
        """record_load_timestamp should upsert a document in pipeline_runs."""
        db = _make_mongomock_db()

        mock_embedding_svc = MagicMock()
        mock_embedding_svc._model_name = "mock-model"

        from app.pipeline.loader import Loader

        loader = Loader(db=db, embedding_service=mock_embedding_svc)

        ts = datetime(2024, 1, 15, 12, 0, 0)
        await loader.record_load_timestamp("odoo", ts)

        doc = db["pipeline_runs"].find_one({"source": "odoo"})
        assert doc is not None, "Expected a pipeline_runs document for 'odoo'"
        assert doc["source"] == "odoo"
        assert doc["last_successful_load"] == ts

    @pytest.mark.asyncio
    async def test_record_load_timestamp_upserts_on_second_call(self):
        """Calling record_load_timestamp twice should update, not duplicate."""
        db = _make_mongomock_db()

        mock_embedding_svc = MagicMock()
        mock_embedding_svc._model_name = "mock-model"

        from app.pipeline.loader import Loader

        loader = Loader(db=db, embedding_service=mock_embedding_svc)

        ts1 = datetime(2024, 1, 15, 12, 0, 0)
        ts2 = datetime(2024, 1, 16, 12, 0, 0)

        await loader.record_load_timestamp("odoo", ts1)
        await loader.record_load_timestamp("odoo", ts2)

        count = db["pipeline_runs"].count_documents({"source": "odoo"})
        assert count == 1, "Expected exactly one pipeline_runs document for 'odoo'"

        doc = db["pipeline_runs"].find_one({"source": "odoo"})
        assert doc["last_successful_load"] == ts2


# ---------------------------------------------------------------------------
# Bonus: core/logging.py — JSON formatter
# ---------------------------------------------------------------------------


class TestJsonLogging:
    def test_configure_logging_emits_json(self):
        from app.core.logging import configure_logging

        stream = StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        logger = logging.getLogger("test.json_logging")
        logger.info("hello world")

        output = stream.getvalue().strip()
        assert output, "Expected log output"

        record = json.loads(output)
        assert record["message"] == "hello world"
        assert record["level"] == "INFO"
        assert "timestamp" in record

    def test_configure_logging_includes_extra_fields(self):
        from app.core.logging import configure_logging

        stream = StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        logger = logging.getLogger("test.extra_fields")
        logger.info("test message", extra={"custom_key": "custom_value"})

        output = stream.getvalue().strip()
        record = json.loads(output)
        assert record.get("custom_key") == "custom_value"


# ---------------------------------------------------------------------------
# Bonus: core/db.py — collection accessors
# ---------------------------------------------------------------------------


class TestDbCollectionAccessors:
    def test_get_products_collection_name(self):
        from app.core.db import COLLECTION_PRODUCTS, get_products_collection

        db = _make_mongomock_db()
        col = get_products_collection(db)
        assert col.name == COLLECTION_PRODUCTS

    def test_get_user_behaviors_collection_name(self):
        from app.core.db import COLLECTION_USER_BEHAVIORS, get_user_behaviors_collection

        db = _make_mongomock_db()
        col = get_user_behaviors_collection(db)
        assert col.name == COLLECTION_USER_BEHAVIORS

    def test_get_pipeline_runs_collection_name(self):
        from app.core.db import COLLECTION_PIPELINE_RUNS, get_pipeline_runs_collection

        db = _make_mongomock_db()
        col = get_pipeline_runs_collection(db)
        assert col.name == COLLECTION_PIPELINE_RUNS


# ---------------------------------------------------------------------------
# Bonus: SourceAdapter Protocol structural check
# ---------------------------------------------------------------------------


class TestSourceAdapterProtocol:
    def test_odoo_adapter_satisfies_protocol(self):
        from app.pipeline.adapters.odoo import OdooAdapter
        from app.pipeline.extractor import SourceAdapter

        with patch("app.pipeline.adapters.odoo.create_async_engine"):
            adapter = OdooAdapter(dsn="postgresql+asyncpg://u:p@h/db")

        assert isinstance(adapter, SourceAdapter)

    def test_vtm_adapter_satisfies_protocol(self):
        from app.pipeline.adapters.vtm import VTMAdapter
        from app.pipeline.extractor import SourceAdapter

        with patch("app.pipeline.adapters.vtm.create_async_engine"):
            adapter = VTMAdapter(dsn="postgresql+asyncpg://u:p@h/db")

        assert isinstance(adapter, SourceAdapter)
