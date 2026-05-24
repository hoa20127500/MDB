"""
HTTP route definitions for the Product Recommendation Engine API.

Endpoints:
- POST /recommendations  — accept a search query, record the search event,
                           and return ranked product recommendations.
- POST /events/search    — record a search behavioral event.
- POST /events/click     — record a click behavioral event.
- POST /events/purchase  — record a purchase behavioral event.

All endpoints access shared services (BehaviorTracker, RecommendationEngine)
from ``request.app.state``, which is populated during the application lifespan
startup sequence in ``app/main.py``.

FastAPI + Pydantic handle HTTP 400 responses automatically for missing or
invalid request parameters (Requirement 7.3).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.models import RecommendationRequest, RecommendationResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Event request models
# ---------------------------------------------------------------------------


class SearchEventRequest(BaseModel):
    """Request body for POST /events/search."""

    user_id: str | None = None
    session_id: str | None = None
    query: str
    timestamp: datetime | None = None


class ClickEventRequest(BaseModel):
    """Request body for POST /events/click."""

    user_id: str | None = None
    session_id: str | None = None
    product_id: str
    timestamp: datetime | None = None


class PurchaseEventRequest(BaseModel):
    """Request body for POST /events/purchase."""

    user_id: str | None = None
    session_id: str | None = None
    product_ids: list[str]
    timestamp: datetime | None = None


# ---------------------------------------------------------------------------
# Recommendation endpoint
# ---------------------------------------------------------------------------


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    body: RecommendationRequest,
    request: Request,
) -> RecommendationResponse:
    """Return personalized product recommendations for a search query.

    Steps:
    1. Record the search event via BehaviorTracker (Requirement 4.1).
    2. Delegate to RecommendationEngine to build context and run vector search.
    3. Return the ranked RecommendationResponse (Requirements 7.1, 7.2).

    FastAPI automatically returns HTTP 400 when Pydantic validation fails on
    the request body (Requirement 7.3).
    """
    behavior_tracker = request.app.state.behavior_tracker
    recommendation_engine = request.app.state.recommendation_engine

    ts = datetime.now(timezone.utc)

    # Record the search event before generating recommendations (Req 4.1).
    await behavior_tracker.record_search(
        user_id=body.user_id,
        query=body.query,
        ts=ts,
        session_id=body.session_id,
    )

    # Generate and return recommendations.
    return await recommendation_engine.get_recommendations(
        query=body.query,
        user_id=body.user_id,
        k=body.k,
        min_score=body.min_score,
        page=body.page,
        page_size=body.page_size,
    )


# ---------------------------------------------------------------------------
# Behavioral event endpoints
# ---------------------------------------------------------------------------


@router.post("/events/search", status_code=200)
async def record_search_event(
    body: SearchEventRequest,
    request: Request,
) -> dict:
    """Record a search behavioral event (Requirement 4.1).

    Returns HTTP 200 on success.  FastAPI returns HTTP 400 automatically when
    required fields are missing or invalid.
    """
    behavior_tracker = request.app.state.behavior_tracker

    ts = body.timestamp if body.timestamp is not None else datetime.now(timezone.utc)

    await behavior_tracker.record_search(
        user_id=body.user_id,
        query=body.query,
        ts=ts,
        session_id=body.session_id,
    )

    return {"status": "ok"}


@router.post("/events/click", status_code=200)
async def record_click_event(
    body: ClickEventRequest,
    request: Request,
) -> dict:
    """Record a product click behavioral event (Requirement 4.2).

    Returns HTTP 200 on success.  FastAPI returns HTTP 400 automatically when
    required fields are missing or invalid.
    """
    behavior_tracker = request.app.state.behavior_tracker

    ts = body.timestamp if body.timestamp is not None else datetime.now(timezone.utc)

    await behavior_tracker.record_click(
        user_id=body.user_id,
        product_id=body.product_id,
        ts=ts,
        session_id=body.session_id,
    )

    return {"status": "ok"}


@router.post("/events/purchase", status_code=200)
async def record_purchase_event(
    body: PurchaseEventRequest,
    request: Request,
) -> dict:
    """Record a purchase behavioral event (Requirement 4.3).

    Returns HTTP 200 on success.  FastAPI returns HTTP 400 automatically when
    required fields are missing or invalid.
    """
    behavior_tracker = request.app.state.behavior_tracker

    ts = body.timestamp if body.timestamp is not None else datetime.now(timezone.utc)

    await behavior_tracker.record_purchase(
        user_id=body.user_id,
        product_ids=body.product_ids,
        ts=ts,
        session_id=body.session_id,
    )

    return {"status": "ok"}
