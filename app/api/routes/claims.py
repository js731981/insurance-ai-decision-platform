from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import InsurFlowOrchestrator, get_insurflow_orchestrator
from app.core.dependencies import get_vector_store
from app.models.schemas import (
    ClaimListItem,
    ClaimProcessResponse,
    ClaimRequest,
    ClaimReviewRequest,
)
from app.services.metrics import metrics
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["claims"])


@router.post("/claims", response_model=ClaimProcessResponse)
async def create_claim(
    body: ClaimRequest,
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> ClaimProcessResponse:
    result = await orchestrator.process_claim(body.model_dump(exclude_none=True))
    return ClaimProcessResponse.model_validate(result)


@router.get("/claims", response_model=list[ClaimListItem])
async def list_claims(
    vector_store: VectorStore = Depends(get_vector_store),
) -> list[ClaimListItem]:
    rows = vector_store.list_claims(limit=200, offset=0)
    out: list[ClaimListItem] = []
    for row in rows:
        meta = row.get("metadata") or {}
        entities = None
        ej = meta.get("entities_json")
        if isinstance(ej, str) and ej.strip():
            try:
                entities = json.loads(ej)
            except json.JSONDecodeError:
                entities = None
        rs_raw = meta.get("review_status") or meta.get("reviewed_action")
        rs = rs_raw if rs_raw in ("APPROVED", "REJECTED") else None
        out.append(
            ClaimListItem(
                claim_id=str(row.get("claim_id") or ""),
                claim_description=str(row.get("claim_description") or ""),
                fraud_score=float(meta.get("fraud_score") or 0.0),
                decision=meta.get("decision"),
                confidence=meta.get("confidence"),
                explanation=meta.get("explanation"),
                review_status=rs,
                hitl_needed=bool(meta.get("hitl_needed") or False),
                reviewed_action=meta.get("reviewed_action"),
                reviewed_at=meta.get("reviewed_at"),
                reviewed_by=meta.get("reviewed_by"),
                entities=entities,
            )
        )
    return out


@router.post("/claims/{claim_id}/review")
async def review_claim(
    claim_id: str,
    body: ClaimReviewRequest,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    action = body.action
    try:
        existing = vector_store.get_claim(claim_id)
        if not existing:
            raise KeyError(f"Claim '{claim_id}' not found.")

        meta = dict(existing.get("metadata") or {})
        meta.setdefault("explanation", "")
        if not str(meta.get("explanation") or "").strip():
            meta["explanation"] = json.dumps(
                {
                    "summary": "Legacy record; explanation was missing.",
                    "key_factors": ["Restored when saving human review."],
                    "similar_case_reference": "",
                },
                ensure_ascii=False,
            )
        meta.setdefault("entities_json", "{}")
        meta.setdefault("fraud_score", 0.0)
        meta.setdefault("decision", "")
        meta.setdefault("confidence", 0.0)
        meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        meta.setdefault("review_status", "")
        meta.setdefault("case_status", "NEW")
        meta.setdefault("assigned_to", "")
        meta.setdefault("assigned_at", "")
        meta.setdefault("updated_at", meta.get("timestamp") or datetime.now(timezone.utc).isoformat())
        now = datetime.now(timezone.utc).isoformat()
        meta.update(
            {
                "review_status": action,
                "reviewed_action": action,
                "reviewed_by": body.reviewed_by or "human_reviewer",
                "reviewed_at": now,
                "hitl_needed": False,
                "hitl_reason": "",
                "updated_at": now,
            }
        )
        vector_store.store_claim(
            claim_id=str(existing.get("claim_id") or claim_id),
            claim_description=str(existing.get("claim_description") or ""),
            embedding=list(existing.get("embedding") or []),
            metadata=meta,
        )
        metrics.record_review()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("claim_review_failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail=f"Failed to review claim: {type(exc).__name__}") from exc
    return {"ok": True, "claim_id": claim_id, "action": action}
