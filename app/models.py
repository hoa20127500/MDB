from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# Regex pattern for source_id: lowercase letters, colon, then non-whitespace characters
_SOURCE_ID_PATTERN = re.compile(r"^[a-z]+:\S+$")


class ProductRecord(BaseModel):
    """A product record extracted from a source system (Odoo ERP or VTM)."""

    model_config = ConfigDict(populate_by_name=True)

    source_id: str  # e.g. "odoo:12345" or "vtm:67890"
    source: str  # "odoo" | "vtm"
    name: str
    description: str
    category: str
    price: float
    availability: bool
    metadata: dict = {}

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if not _SOURCE_ID_PATTERN.match(v):
            raise ValueError(
                f"source_id must match pattern '^[a-z]+:\\S+$', got: {v!r}"
            )
        return v


class BehaviorEvent(BaseModel):
    """A user behavioral event (search, click, or purchase)."""

    user_id: str | None
    session_id: str | None
    event_type: Literal["search", "click", "purchase"]
    query: str | None = None
    product_id: str | None = None
    product_ids: list[str] = []
    timestamp: datetime


class RecommendationRequest(BaseModel):
    """Request payload for the recommendation endpoint."""

    query: str
    user_id: str | None = None
    session_id: str | None = None
    k: int = 10
    min_score: float | None = None
    page: int = 1
    page_size: int = 10

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must be a non-empty string")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("k must be >= 1")
        return v

    @field_validator("page")
    @classmethod
    def validate_page(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page must be >= 1")
        return v

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("page_size must be >= 1")
        return v


class RecommendedProduct(BaseModel):
    """A single product in a recommendation response, with its similarity score."""

    source_id: str
    name: str
    description: str
    category: str
    price: float
    availability: bool
    similarity_score: float


class RecommendationResponse(BaseModel):
    """Response payload for the recommendation endpoint."""

    query: str
    results: list[RecommendedProduct]
    total: int
    page: int
    page_size: int


class PipelineRunResult(BaseModel):
    """Result of a completed data pipeline run."""

    run_id: str
    status: Literal["success", "failure"]
    started_at: datetime
    completed_at: datetime
    records_processed: dict[str, int]
    errors: list[str]


class PipelineStatus(BaseModel):
    """Current status of the data pipeline."""

    running: bool
    latest_run: PipelineRunResult | None = None


class LoadResult(BaseModel):
    """Result of a batch upsert operation to MongoDB."""

    upserted: int
    modified: int
    failed: int
