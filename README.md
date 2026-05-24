# Product Recommendation Engine

A Python/FastAPI backend service that delivers personalized product recommendations using MongoDB Atlas Vector Search and sentence-transformers embeddings.

Built for the **MUG VN - MDB Hackathon**.

---

## Overview

The system operates as two loosely coupled subsystems:

1. **Data Pipeline** — ETL process that extracts product records from Odoo ERP and VTM PostgreSQL databases, loads them into MongoDB, and generates vector embeddings for each product.
2. **Recommendation API** — FastAPI service that accepts search queries, builds a user-context embedding from behavioral history, and executes a MongoDB Atlas Vector Search to return ranked product recommendations.

---

## Architecture

```
Odoo ERP (PostgreSQL) ──┐
                        ├──► Extractor ──► Loader ──► MongoDB Atlas
VTM System (PostgreSQL) ┘                    │
                                             └──► EmbeddingService (all-MiniLM-L6-v2)

Client ──► FastAPI ──► BehaviorTracker ──► user_behaviors collection
               └──► RecommendationEngine ──► $vectorSearch ──► products collection
```

---

## Project Structure

```
.
├── app/
│   ├── main.py                    # FastAPI app factory, lifespan hooks
│   ├── models.py                  # Pydantic data models
│   ├── api/
│   │   ├── routes.py              # POST /recommendations, POST /events/*
│   │   ├── admin.py               # POST /admin/pipeline/trigger, GET /admin/pipeline/status, GET /health
│   │   ├── behavior.py            # BehaviorTracker
│   │   └── recommendation.py     # RecommendationEngine, build_context_embedding
│   ├── core/
│   │   ├── config.py              # Settings (pydantic-settings)
│   │   ├── db.py                  # Motor client factory, collection accessors
│   │   ├── embedding.py           # EmbeddingService (sentence-transformers)
│   │   ├── indexes.py             # ensure_indexes() — MongoDB index setup
│   │   └── logging.py             # JSON structured logging, request-ID support
│   └── pipeline/
│       ├── extractor.py           # SourceAdapter Protocol
│       ├── loader.py              # Loader (bulk upsert + embedding update)
│       ├── orchestrator.py        # PipelineOrchestrator
│       ├── scheduler.py           # PipelineScheduler (APScheduler)
│       └── adapters/
│           ├── odoo.py            # OdooAdapter
│           └── vtm.py             # VTMAdapter
├── tests/
│   ├── unit/
│   │   ├── test_pipeline.py       # 43 unit tests — models, db, logging, embedding, loader, adapters
│   │   └── test_recommendation.py # 30 unit tests — BehaviorTracker, context embedding, RecommendationEngine
│   └── integration/
│       ├── test_pipeline.py       # 4 integration tests — full pipeline run with mongomock
│       └── test_routes.py         # 3 smoke tests — route registration via TestClient
├── pyproject.toml
└── README.md
```

---

## Requirements

- Python 3.11+
- MongoDB Atlas cluster (for vector search in production)
- PostgreSQL instances for Odoo ERP and VTM (for the data pipeline)

---

## Installation

```bash
# Install runtime dependencies
pip install -e .

# Install dev/test dependencies
pip install -e ".[dev]"
```

---

## Configuration

All configuration is loaded from environment variables (or a `.env` file in the project root).

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGODB_URI` | ✅ | — | MongoDB Atlas connection URI (`mongodb+srv://...`) |
| `ODOO_DSN` | ✅ | — | SQLAlchemy async DSN for Odoo PostgreSQL (`postgresql+asyncpg://user:pass@host/db`) |
| `VTM_DSN` | ✅ | — | SQLAlchemy async DSN for VTM PostgreSQL |
| `EMBEDDING_MODEL` | | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `SCHEDULER_INTERVAL_HOURS` | | `24` | How often the pipeline runs automatically (hours) |
| `DEFAULT_K` | | `10` | Default number of recommendations to return |
| `MIN_SCORE` | | `None` | Global minimum similarity score threshold (0.0–1.0) |
| `BATCH_SIZE` | | `100` | Records per MongoDB bulk-write batch |

Create a `.env` file:

```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
ODOO_DSN=postgresql+asyncpg://user:pass@localhost:5432/odoo
VTM_DSN=postgresql+asyncpg://user:pass@localhost:5432/vtm
```

---

## MongoDB Atlas Vector Search Index

> ⚠️ **Manual step required.** Motor cannot create HNSW indexes programmatically. You must create the vector search index via the Atlas UI or Admin API before recommendations will work.

**Index definition:**

```json
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
```

**Atlas UI steps:**
1. Open your cluster → Browse Collections → `products` collection
2. Click **Search Indexes** → **Create Search Index**
3. Choose **JSON Editor**, paste the definition above, click **Create**

---

## Running the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### Recommendations

#### `POST /recommendations`

Get personalized product recommendations for a search query.

**Request body:**
```json
{
  "query": "wireless keyboard",
  "user_id": "user_abc",
  "k": 10,
  "min_score": 0.7,
  "page": 1,
  "page_size": 10
}
```

**Response:**
```json
{
  "query": "wireless keyboard",
  "results": [
    {
      "source_id": "odoo:12345",
      "name": "Wireless Keyboard",
      "description": "Compact wireless keyboard with backlight",
      "category": "Electronics",
      "price": 49.99,
      "availability": true,
      "similarity_score": 0.94
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 10
}
```

### Behavioral Events

#### `POST /events/search`
```json
{ "user_id": "user_abc", "query": "wireless keyboard" }
```

#### `POST /events/click`
```json
{ "user_id": "user_abc", "product_id": "odoo:12345" }
```

#### `POST /events/purchase`
```json
{ "user_id": "user_abc", "product_ids": ["odoo:12345", "vtm:67890"] }
```

### Admin / Pipeline

#### `POST /admin/pipeline/trigger?mode=incremental`

Manually trigger a pipeline run. Returns HTTP 409 if a run is already in progress.

#### `GET /admin/pipeline/status`

Get the current pipeline run status.

#### `GET /health`

Health check. Returns `{"status": "ok"}`.

---

## Running Tests

```bash
# Full test suite (80 tests)
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v
```

---

## How Recommendations Work

1. **Query encoding** — The search query is encoded into a 384-dim vector using `all-MiniLM-L6-v2`.
2. **Behavioral context** — The user's last 50 events are fetched and blended into the query embedding using weighted averaging:
   - Purchase events: weight **3.0**
   - Click events: weight **2.0**
   - Search events: weight **1.0**
3. **Vector search** — The context embedding is used to query MongoDB Atlas `$vectorSearch` (cosine similarity, HNSW index).
4. **Ranking** — Results are returned sorted by similarity score descending, with optional `min_score` filtering and pagination.

---

## Data Pipeline

The pipeline extracts products from Odoo ERP and VTM PostgreSQL, upserts them into MongoDB, and generates embeddings.

- **Full mode** — extracts all products from both sources
- **Incremental mode** — extracts only products modified since the last successful run
- **Scheduled** — runs automatically every `SCHEDULER_INTERVAL_HOURS` hours (default: 24)
- **On-demand** — trigger via `POST /admin/pipeline/trigger`
- **Retry logic** — failed MongoDB batches are retried up to 3 times with exponential backoff (1s, 2s, 4s)
- **Fault tolerant** — a failure in one source does not stop processing of the other
# MDB
# MDB
