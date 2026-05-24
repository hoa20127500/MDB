"""
MongoDB index setup for the Product Recommendation Engine.

Call ``ensure_indexes(db)`` once at application startup to create all
required indexes on the ``products`` and ``user_behaviors`` collections.

NOTE — Atlas Vector Search index
---------------------------------
The HNSW vector search index on ``products.embedding`` **cannot** be created
programmatically via Motor or pymongo.  It must be created manually through
the MongoDB Atlas UI or the Atlas Admin API.

Use the following JSON definition when creating the index:

    Collection : products
    Index name : product_embedding_index

    {
      "name": "product_embedding_index",
      "type": "vectorSearch",
      "definition": {
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine"
          }
        ]
      }
    }

Steps (Atlas UI):
  1. Open your Atlas cluster → Browse Collections → products collection.
  2. Click "Search Indexes" → "Create Search Index".
  3. Choose "JSON Editor", paste the definition above, and click "Create".

Steps (Atlas Admin API):
  POST /api/atlas/v2/groups/{groupId}/clusters/{clusterName}/search/indexes
  Body: <JSON definition above>

Requirements: 3.6
"""

from __future__ import annotations

import inspect
import json

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.core.db import get_products_collection, get_user_behaviors_collection
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Atlas Vector Search index definition (for manual creation)
# ---------------------------------------------------------------------------

VECTOR_SEARCH_INDEX_DEFINITION: dict = {
    "name": "product_embedding_index",
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine",
            }
        ]
    },
}


async def _create_index(collection, *args, **kwargs) -> None:
    """Call create_index on *collection*, awaiting only if the result is a coroutine.

    Motor returns a coroutine; mongomock returns a plain string.
    This helper makes ``ensure_indexes`` compatible with both.
    """
    result = collection.create_index(*args, **kwargs)
    if inspect.isawaitable(result):
        await result


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create all required MongoDB indexes at application startup.

    Creates the following indexes (idempotent — safe to call on every startup):

    **products collection**
    - Unique index on ``source_id`` — prevents duplicate product documents
      from the same source system.

    **user_behaviors collection**
    - Compound descending index on ``(user_id, timestamp)`` — supports
      efficient retrieval of a user's most recent events.
    - Index on ``session_id`` — supports anonymous user lookups.

    The Atlas Vector Search index on ``products.embedding`` is **not** created
    here (Motor cannot create HNSW indexes programmatically).  Its definition
    is logged at startup so operators know what to apply manually.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance.
    """
    products = get_products_collection(db)
    behaviors = get_user_behaviors_collection(db)

    # ── products.source_id — unique ──────────────────────────────────────────
    await _create_index(products, "source_id", unique=True)
    logger.info("Index ensured: products.source_id (unique)")

    # ── user_behaviors.(user_id, timestamp) — compound descending ────────────
    await _create_index(
        behaviors,
        [("user_id", DESCENDING), ("timestamp", DESCENDING)],
    )
    logger.info("Index ensured: user_behaviors.(user_id, timestamp) (compound desc)")

    # ── user_behaviors.session_id ────────────────────────────────────────────
    await _create_index(behaviors, "session_id")
    logger.info("Index ensured: user_behaviors.session_id")

    # ── Atlas Vector Search index — manual creation required ─────────────────
    logger.info(
        "ACTION REQUIRED: Create the Atlas Vector Search index manually.\n"
        "Collection : products\n"
        "Index name : product_embedding_index\n"
        "Definition :\n%s\n"
        "Steps — Atlas UI  : Cluster → Browse Collections → products → "
        "Search Indexes → Create Search Index → JSON Editor → paste definition.\n"
        "Steps — Admin API : POST /api/atlas/v2/groups/{groupId}/clusters/"
        "{clusterName}/search/indexes  (body = definition above).",
        json.dumps(VECTOR_SEARCH_INDEX_DEFINITION, indent=2),
    )
