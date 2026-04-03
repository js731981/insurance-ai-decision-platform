from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_vector_store
from app.services.analytics import (
    build_analytics_summary,
    build_anomaly_alerts,
    build_fraud_leaderboard,
)
from app.services.vector_store import VectorStore

router = APIRouter(tags=["analytics"])


@router.get("/analytics/summary")
async def analytics_summary(
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    return build_analytics_summary(vector_store)


@router.get("/analytics/anomalies")
async def analytics_anomalies(
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    return build_anomaly_alerts(vector_store)


@router.get("/analytics/leaderboard")
async def analytics_leaderboard(
    limit: int = Query(10, ge=1, le=200),
    min_fraud_score: Optional[float] = Query(None),
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    return build_fraud_leaderboard(
        vector_store, limit=limit, min_fraud_score=min_fraud_score
    )
