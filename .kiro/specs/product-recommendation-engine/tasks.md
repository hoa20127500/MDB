# Implementation Plan: Product Recommendation Engine

## Overview

Implement a Python/FastAPI backend service that delivers personalized product recommendations via semantic vector search. The implementation proceeds in six phases: project scaffolding, data models and configuration, the ETL data pipeline, the recommendation API, the FastAPI application layer, and finally scheduling and pipeline orchestration. Each phase builds directly on the previous one, ending with all components wired together and verified.

## Tasks

- [ ] 1. Project scaffolding and configuration
  - Create the top-level package structure: `app/`, `app/pipeline/`, `app/api/`, `app/core/`, `tests/`
  - Add `pyproject.toml` (or `requirements.txt`) with pinned dependencies: `fastapi`, `uvicorn`, `motor`, `pymongo`, `sqlalchemy[asyncio]`, `asyncpg`, `sentence-transformers`, `apscheduler`, `pydantic`, `pydantic-settings`, `hypothesis`, `pytest`, `pytest-asyncio`, `mongomock`
  - Create `app/core/config.py` using `pydantic-settings` to load all environment variables: MongoDB URI, Odoo DSN, VTM DSN, embedding model name, scheduler interval, default K, min score, batch size
  - Create `app/core/db.py` with async MongoDB client factory (Motor) and collection accessors for `products`, `user_behaviors`, and `pipeline_runs`
  - Create `app/core/logging.py` with structured logging setup (JSON formatter, request-id support)
  - _Requirements: 1.5, 2.4, 3.4, 8.1_

- [ ] 2. Pydantic data models
  - [x] 2.1 Implement all Pydantic models in `app/models.py`
    - `ProductRecord`, `BehaviorEvent`, `RecommendationRequest`, `RecommendedProduct`, `RecommendationResponse`, `PipelineRunResult`, `PipelineStatus`, `LoadResult`
    - Add field validators: `source_id` must match `^[a-z]+:\S+$`; `event_type` must be one of `search`, `click`, `purchase`; `k` must be ≥ 1; `page` and `page_size` must be ≥ 1
    - _Requirements: 1.3, 4.1, 4.2, 4.3, 6.3, 7.5_

  - [ ]* 2.2 Write property test for source ID format invariant
    - **Property 1: Source ID Format Invariant**
    - Use `hypothesis` `st.text()` strategies for source names and original IDs; verify `source_id == f"{source_name}:{original_id}"` and that distinct original IDs produce distinct `source_id` values
    - Tag: `# Feature: product-recommendation-engine, Property 1: Source ID Format Invariant`
    - **Validates: Requirements 1.3**

- [ ] 3. Embedding Service
  - [x] 3.1 Implement `app/core/embedding.py` — `EmbeddingService` class
    - Load `all-MiniLM-L6-v2` once at construction; expose `encode(text: str) -> list[float]` and `encode_batch(texts: list[str]) -> list[list[float]]`
    - Implement `build_product_text(record: ProductRecord) -> str` concatenating `name`, `description`, and `category`
    - Catch model inference exceptions per item; log `product_id` and error; return `None` for failed items (caller must not store partial embeddings)
    - _Requirements: 3.1, 3.4, 3.5_

  - [ ]* 3.2 Write property test for embedding dimensionality consistency
    - **Property 10: Query Embedding Dimensionality Consistency**
    - Use `hypothesis` `st.text(min_size=1)` to generate arbitrary query strings; assert `len(encode(query)) == 384`
    - Tag: `# Feature: product-recommendation-engine, Property 10: Query Embedding Dimensionality Consistency`
    - **Validates: Requirements 6.1**

  - [ ]* 3.3 Write property test for product data round-trip embedding determinism
    - **Property 4 (embedding portion): Product Data Round-Trip**
    - Generate random `ProductRecord` instances; assert two products with identical `name`, `description`, `category` produce identical embeddings regardless of other fields
    - Tag: `# Feature: product-recommendation-engine, Property 4: Product Data Round-Trip`
    - **Validates: Requirements 3.1, 3.4**

- [ ] 4. Data Pipeline — Extractor
  - [x] 4.1 Define `SourceAdapter` Protocol in `app/pipeline/extractor.py`
    - Protocol with `source_name: str` and `async def extract(since: datetime | None) -> AsyncIterator[ProductRecord]`
    - _Requirements: 1.1, 1.2, 1.6_

  - [x] 4.2 Implement `OdooAdapter` in `app/pipeline/adapters/odoo.py`
    - Use SQLAlchemy async session; `SELECT` product rows (name, description, category, price, availability) from Odoo schema
    - Apply `WHERE updated_at > since` when `since` is not None (incremental mode)
    - Assign `source_id = f"odoo:{row.id}"`; wrap connection in try/except; log failures with source name, timestamp, and reason; re-raise to allow orchestrator to continue
    - _Requirements: 1.1, 1.3, 1.5, 1.6_

  - [x] 4.3 Implement `VTMAdapter` in `app/pipeline/adapters/vtm.py`
    - Mirror `OdooAdapter` structure for the VTM PostgreSQL schema
    - Assign `source_id = f"vtm:{row.id}"`
    - _Requirements: 1.2, 1.3, 1.5, 1.6_

  - [ ]* 4.4 Write property test for incremental extraction filter
    - **Property 3: Incremental Extraction Filter**
    - Generate random lists of `ProductRecord`-like dicts with random `updated_at` values and a random cutoff; assert extracted set equals exactly those with `updated_at > cutoff`
    - Tag: `# Feature: product-recommendation-engine, Property 3: Incremental Extraction Filter`
    - **Validates: Requirements 1.6**

- [ ] 5. Data Pipeline — Loader
  - [x] 5.1 Implement `Loader` in `app/pipeline/loader.py`
    - `upsert_batch(records: list[ProductRecord]) -> LoadResult`: use Motor `bulk_write` with `UpdateOne(filter={"source_id": r.source_id}, update={"$set": {...}}, upsert=True)` for each record
    - Retry up to 3 times with exponential backoff (1 s, 2 s, 4 s) on `PyMongoError`; after 3 failures log and skip the batch
    - After a successful upsert, call `EmbeddingService.encode_batch` on the batch; update each document's `embedding` and `embedding_model` fields; skip documents where encoding returned `None`
    - `record_load_timestamp(source_name: str, ts: datetime)`: upsert a metadata document in `pipeline_runs` or a dedicated metadata collection
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.5_

  - [ ]* 5.2 Write property test for upsert idempotence
    - **Property 2: Upsert Idempotence**
    - Use `mongomock` in-memory MongoDB; generate random `ProductRecord` instances; upsert each twice; assert collection document count equals number of distinct `source_id` values
    - Tag: `# Feature: product-recommendation-engine, Property 2: Upsert Idempotence`
    - **Validates: Requirements 1.4**

  - [ ]* 5.3 Write property test for product data round-trip field preservation
    - **Property 4 (storage portion): Product Data Round-Trip**
    - Use `mongomock`; generate random `ProductRecord` instances; write and read back; assert all original fields are present with original values and `embedding` is a non-empty list
    - Tag: `# Feature: product-recommendation-engine, Property 4: Product Data Round-Trip`
    - **Validates: Requirements 2.2, 3.2**

  - [ ]* 5.4 Write property test for embedding regeneration on update
    - **Property 5: Embedding Regeneration on Update**
    - Use `mongomock`; store a product; update `name`/`description`/`category`; re-run embedding step; assert exactly one `embedding` field exists and it differs from the original
    - Tag: `# Feature: product-recommendation-engine, Property 5: Embedding Regeneration on Update`
    - **Validates: Requirements 3.3**

- [x] 6. Checkpoint — pipeline unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Behavior Tracker
  - [x] 7.1 Implement `BehaviorTracker` in `app/api/behavior.py`
    - `record_search(user_id, query, ts)`, `record_click(user_id, product_id, ts)`, `record_purchase(user_id, product_ids, ts)`: insert `BehaviorEvent` documents into `user_behaviors`; when `user_id` is None use `session_id` and set `user_id=null`
    - `get_recent_events(user_id, limit=50)`: query `user_behaviors` with `{"user_id": user_id}` sorted by `timestamp` descending, limit to `limit`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1_

  - [ ]* 7.2 Write property test for behavioral event storage round-trip
    - **Property 6: Behavioral Event Storage Round-Trip**
    - Use `mongomock`; generate random events of all three types with random `user_id`/`session_id` combinations; record each; query back by `user_id` or `session_id`; assert all original fields are present; assert anonymous events have `user_id=null` and non-null `session_id`
    - Tag: `# Feature: product-recommendation-engine, Property 6: Behavioral Event Storage Round-Trip`
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**

  - [ ]* 7.3 Write property test for recent events limit
    - **Property 7: Recent Events Limit**
    - Use `mongomock`; generate users with random event counts > 50; call `get_recent_events(user_id, limit=50)`; assert result length ≤ 50 and events are the 50 most recent by timestamp
    - Tag: `# Feature: product-recommendation-engine, Property 7: Recent Events Limit`
    - **Validates: Requirements 5.1**

- [ ] 8. Recommendation Engine
  - [x] 8.1 Implement `build_context_embedding` in `app/api/recommendation.py`
    - Weight events: purchase = 3.0, click = 2.0, search = 1.0
    - Encode each event's text (query for search events, product name for click/purchase events); compute weighted average of query embedding and event embeddings
    - If `events` is empty, return `query_embedding` unchanged
    - _Requirements: 5.2, 5.3, 5.4_

  - [ ]* 8.2 Write property test for context embedding weighting
    - **Property 8: Context Embedding Weighting**
    - Construct controlled event sets with known product embeddings (one purchase event, one search event referencing different products); assert cosine similarity of context embedding to purchased product embedding > cosine similarity to searched product embedding
    - Tag: `# Feature: product-recommendation-engine, Property 8: Context Embedding Weighting`
    - **Validates: Requirements 5.2, 5.3**

  - [ ]* 8.3 Write property test for empty history fallback
    - **Property 9: Empty History Falls Back to Query Embedding**
    - Generate random 384-dim float vectors as query embeddings; call `build_context_embedding(query_embedding, events=[])`; assert result equals `query_embedding` within floating-point tolerance (`numpy.allclose`)
    - Tag: `# Feature: product-recommendation-engine, Property 9: Empty History Falls Back to Query Embedding`
    - **Validates: Requirements 5.4**

  - [x] 8.4 Implement `RecommendationEngine.get_recommendations` in `app/api/recommendation.py`
    - Encode query via `EmbeddingService`; fetch last 50 behavioral events via `BehaviorTracker`; call `build_context_embedding`; execute `$vectorSearch` aggregation pipeline against `products` collection
    - Apply `$match` on `similarity_score >= min_score` when `min_score` is set; apply pagination offset/limit in the aggregation or post-query
    - Return `RecommendationResponse` with results sorted descending by `similarity_score`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.5_

  - [ ]* 8.5 Write property test for recommendation response invariants
    - **Property 11: Recommendation Response Invariants**
    - Mock `$vectorSearch` to return random score lists; call `get_recommendations` with varying `k` and `min_score`; assert `len(results) <= k`, all scores non-null, scores in non-increasing order, no score below `min_score`
    - Tag: `# Feature: product-recommendation-engine, Property 11: Recommendation Response Invariants`
    - **Validates: Requirements 6.3, 6.4, 6.6, 6.7**

  - [ ]* 8.6 Write property test for pagination correctness
    - **Property 14: Pagination Correctness**
    - Generate random result sets of size N with random `page` and `page_size` values; assert returned slice length equals `min(page_size, max(0, N - (page-1)*page_size))` and items correspond to the correct offset in the full ranked list
    - Tag: `# Feature: product-recommendation-engine, Property 14: Pagination Correctness`
    - **Validates: Requirements 7.5**

- [x] 9. Checkpoint — recommendation engine unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. FastAPI application and routes
  - [x] 10.1 Create `app/main.py` — FastAPI app factory
    - Instantiate `FastAPI`; register startup/shutdown lifespan hooks to initialize Motor client, `EmbeddingService`, `BehaviorTracker`, `RecommendationEngine`, `PipelineOrchestrator`, and `PipelineScheduler`
    - Register a global exception handler that catches unhandled exceptions, logs full stack trace with request ID, and returns HTTP 500 with `{"error_code": "INTERNAL_ERROR"}`
    - _Requirements: 7.4, 8.1_

  - [x] 10.2 Implement recommendation and event routes in `app/api/routes.py`
    - `POST /recommendations`: validate `RecommendationRequest`; call `BehaviorTracker.record_search`; call `RecommendationEngine.get_recommendations`; return `RecommendationResponse`
    - `POST /events/search`, `POST /events/click`, `POST /events/purchase`: validate request body; call corresponding `BehaviorTracker` method; return HTTP 200
    - Return HTTP 400 with descriptive `detail` for missing/invalid parameters (FastAPI validation handles this automatically via Pydantic)
    - _Requirements: 7.1, 7.2, 7.3, 4.1, 4.2, 4.3_

  - [x] 10.3 Implement admin pipeline routes in `app/api/admin.py`
    - `POST /admin/pipeline/trigger`: call `PipelineOrchestrator.run(mode)`; return HTTP 409 with status message if already running
    - `GET /admin/pipeline/status`: call `PipelineOrchestrator.get_status()`; return current run status
    - `GET /health`: return `{"status": "ok"}` with HTTP 200
    - _Requirements: 8.2, 8.4_

  - [ ]* 10.4 Write property test for valid API response schema
    - **Property 12: Valid API Response Schema**
    - Use FastAPI `TestClient`; generate random valid `RecommendationRequest` payloads with `hypothesis`; assert HTTP 200 and response body contains `results` array where each element has all required fields (`source_id`, `name`, `description`, `category`, `price`, `availability`, `similarity_score`)
    - Tag: `# Feature: product-recommendation-engine, Property 12: Valid API Response Schema`
    - **Validates: Requirements 7.2**

  - [ ]* 10.5 Write property test for missing parameters return 400
    - **Property 13: Missing Parameters Return 400**
    - Use FastAPI `TestClient`; generate recommendation requests with `query` field omitted or set to empty string; assert HTTP 400 and response body contains non-empty `detail` field
    - Tag: `# Feature: product-recommendation-engine, Property 13: Missing Parameters Return 400`
    - **Validates: Requirements 7.3**

- [ ] 11. Pipeline Orchestrator and Scheduler
  - [x] 11.1 Implement `PipelineOrchestrator` in `app/pipeline/orchestrator.py`
    - `run(mode: Literal["full", "incremental"]) -> PipelineRunResult`: check `_running` flag; if True return HTTP 409 message; set `_running = True`; insert `pipeline_runs` document with `status="running"`
    - Loop over `[OdooAdapter, VTMAdapter]`; call `extractor.extract(since=last_run_ts if incremental)`; batch records; call `Loader.upsert_batch`; catch per-source exceptions, log with source name and timestamp, append to `errors`, continue to next source
    - On completion update `pipeline_runs` document with `status`, `completed_at`, `records_processed`, `errors`; call `Loader.record_load_timestamp` per source; set `_running = False`
    - `get_status() -> PipelineStatus`: return current `_running` flag and latest `pipeline_runs` document
    - _Requirements: 1.5, 2.3, 2.4, 8.3, 8.4_

  - [ ]* 11.2 Write property test for pipeline run record completeness
    - **Property 15: Pipeline Run Record Completeness**
    - Use `mongomock`; simulate pipeline runs with random source counts and random success/failure outcomes; assert every completed `pipeline_runs` document contains `status` ∈ {"success", "failure"}, non-null `started_at`, non-null `completed_at`, and `records_processed` map with an entry for each processed source
    - Tag: `# Feature: product-recommendation-engine, Property 15: Pipeline Run Record Completeness`
    - **Validates: Requirements 8.3**

  - [x] 11.3 Implement `PipelineScheduler` in `app/pipeline/scheduler.py`
    - Use `APScheduler` `AsyncIOScheduler`; `start(interval_hours: int = 24)` adds an interval job calling `PipelineOrchestrator.run(mode="incremental")`; `shutdown()` gracefully stops the scheduler
    - Wire scheduler startup/shutdown into the FastAPI lifespan hooks in `app/main.py`
    - _Requirements: 8.1_

- [x] 12. MongoDB index setup
  - Create `app/core/indexes.py` with an `ensure_indexes()` coroutine called at startup
  - Create unique index on `products.source_id`
  - Create compound descending index on `user_behaviors.(user_id, timestamp)`
  - Create index on `user_behaviors.session_id`
  - Print Atlas Vector Search index JSON definition to stdout/logs at startup with instructions to apply it via Atlas UI or API (the HNSW index cannot be created programmatically via Motor; document the `product_embedding_index` definition from the design)
  - _Requirements: 3.6_

- [x] 13. Checkpoint — full test suite passes
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Wire everything together and validate end-to-end
  - [x] 14.1 Write `tests/integration/test_pipeline.py`
    - Seed a test PostgreSQL instance (via Docker Compose or `pytest` fixtures with `asyncpg`); run `PipelineOrchestrator.run(mode="full")` against a `mongomock` or local MongoDB; assert products are present with non-empty `embedding` fields
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2_

  - [ ]* 14.2 Write integration test for end-to-end recommendation flow
    - Seed `products` collection with known embeddings; POST to `/recommendations` via `TestClient`; assert HTTP 200, non-empty `results`, scores in descending order
    - _Requirements: 6.2, 6.5, 7.2_

  - [x] 14.3 Confirm all routes are registered and reachable via `TestClient`
    - Assert `/health` returns 200; assert `/admin/pipeline/status` returns 200; assert `/recommendations` with missing `query` returns 400
    - _Requirements: 7.1, 7.3, 8.2_

- [x] 15. Final checkpoint — all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use `@settings(max_examples=100)` and are tagged with `# Feature: product-recommendation-engine, Property N: <title>`
- The Atlas Vector Search index (`product_embedding_index`) must be created manually via the Atlas UI or Atlas Admin API — Motor cannot create HNSW indexes programmatically
- `mongomock` is used for all unit/property tests that touch MongoDB; real MongoDB Atlas is required only for integration tests
- The `EmbeddingService` loads the model once at startup; tests that do not need real embeddings should mock `encode()` to return deterministic 384-dim vectors for speed
