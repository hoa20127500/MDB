"""
Async MongoDB client factory and collection accessors (Motor).

Usage
-----
Call ``get_client()`` once at application startup (e.g. in the FastAPI lifespan
hook) and store the result.  Pass the client (or the database) through
dependency injection rather than calling ``get_client()`` repeatedly.

Collection accessors accept an ``AsyncIOMotorDatabase`` instance so they can be
used with both the real Motor client and a ``mongomock`` database in tests.
"""

from __future__ import annotations

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

# ── Collection name constants ────────────────────────────────────────────────

COLLECTION_PRODUCTS = "products"
COLLECTION_USER_BEHAVIORS = "user_behaviors"
COLLECTION_PIPELINE_RUNS = "pipeline_runs"

# Default database name; can be overridden by callers.
DEFAULT_DB_NAME = "recommendation_engine"


# ── Client factory ───────────────────────────────────────────────────────────


def create_motor_client(mongodb_uri: str) -> AsyncIOMotorClient:
    """Create and return an ``AsyncIOMotorClient`` for the given URI.

    The client is *not* connected until the first operation is performed
    (Motor uses lazy connection by default).  Call ``client.close()`` during
    application shutdown to release connection-pool resources.

    Args:
        mongodb_uri: A valid MongoDB connection string, e.g.
            ``mongodb+srv://user:pass@cluster.mongodb.net/``.

    Returns:
        A configured ``AsyncIOMotorClient`` instance.
    """
    return AsyncIOMotorClient(mongodb_uri)


def get_database(
    client: AsyncIOMotorClient,
    db_name: str = DEFAULT_DB_NAME,
) -> AsyncIOMotorDatabase:
    """Return the Motor database object for *db_name*.

    Args:
        client: An ``AsyncIOMotorClient`` created by :func:`create_motor_client`.
        db_name: Name of the MongoDB database.  Defaults to
            ``DEFAULT_DB_NAME``.

    Returns:
        An ``AsyncIOMotorDatabase`` instance.
    """
    return client[db_name]


# ── Collection accessors ─────────────────────────────────────────────────────


def get_products_collection(db: AsyncIOMotorDatabase) -> AsyncIOMotorCollection:
    """Return the ``products`` collection from *db*.

    The ``products`` collection stores product records together with their
    vector embeddings.  A unique index on ``source_id`` and an Atlas Vector
    Search index on ``embedding`` are expected to exist (see
    ``app/core/indexes.py``).

    Args:
        db: An ``AsyncIOMotorDatabase`` instance.

    Returns:
        The ``products`` ``AsyncIOMotorCollection``.
    """
    return db[COLLECTION_PRODUCTS]


def get_user_behaviors_collection(db: AsyncIOMotorDatabase) -> AsyncIOMotorCollection:
    """Return the ``user_behaviors`` collection from *db*.

    The ``user_behaviors`` collection stores search, click, and purchase events
    keyed by ``user_id`` (or ``session_id`` for anonymous users).

    Args:
        db: An ``AsyncIOMotorDatabase`` instance.

    Returns:
        The ``user_behaviors`` ``AsyncIOMotorCollection``.
    """
    return db[COLLECTION_USER_BEHAVIORS]


def get_pipeline_runs_collection(db: AsyncIOMotorDatabase) -> AsyncIOMotorCollection:
    """Return the ``pipeline_runs`` collection from *db*.

    The ``pipeline_runs`` collection stores metadata about each ETL pipeline
    execution: status, timing, record counts, and any per-source errors.

    Args:
        db: An ``AsyncIOMotorDatabase`` instance.

    Returns:
        The ``pipeline_runs`` ``AsyncIOMotorCollection``.
    """
    return db[COLLECTION_PIPELINE_RUNS]
