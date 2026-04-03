from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import get_vector_store
from app.models.schemas import (
    CaseAssignRequest,
    CaseListItem,
    CaseListResponse,
    CaseStatusUpdateRequest,
    ClaimDecision,
)
from app.services.analytics import risk_level_from_claim_metadata
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cases"])

_VALID_CASE_STATUS = frozenset({"NEW", "ASSIGNED", "IN_PROGRESS", "RESOLVED"})
_REVIEW_OUT = frozenset({"APPROVED", "REJECTED"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_case_status(meta: dict) -> str:
    s = str(meta.get("case_status") or "").strip().upper()
    return s if s in _VALID_CASE_STATUS else "NEW"


def _review_status_for_case(meta: dict) -> str:
    rs = str(meta.get("review_status") or "").strip().upper()
    if rs in _REVIEW_OUT:
        return rs
    ra = str(meta.get("reviewed_action") or "").strip().upper()
    if ra in _REVIEW_OUT:
        return ra
    return ""


def _decision_for_case(meta: dict) -> ClaimDecision | None:
    d = str(meta.get("decision") or "").strip().upper()
    if d in ("APPROVED", "REJECTED", "INVESTIGATE"):
        return cast(ClaimDecision, d)
    return None


@router.get("/cases", response_model=CaseListResponse)
async def list_cases(
    *,
    case_status: str | None = Query(default=None, description="Filter by case_status"),
    assigned_to: str | None = Query(default=None, description="Filter by assigned investigator id/name"),
    unassigned_only: bool = Query(default=False, description="Only claims in NEW (unassigned) state"),
    vector_store: VectorStore = Depends(get_vector_store),
) -> CaseListResponse:
    rows = vector_store.list_claims(limit=500, offset=0)
    want_status = (case_status or "").strip().upper() or None
    if want_status and want_status not in _VALID_CASE_STATUS:
        raise HTTPException(status_code=400, detail=f"Invalid case_status filter: {case_status!r}")
    want_assignee = (assigned_to or "").strip() or None

    cases: list[CaseListItem] = []
    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        cid = str(row.get("claim_id") or "").strip()
        if not cid:
            continue
        cs = _normalize_case_status(meta)
        ato = str(meta.get("assigned_to") or "").strip()

        if unassigned_only and cs != "NEW":
            continue
        if want_status and cs != want_status:
            continue
        if want_assignee is not None and ato != want_assignee:
            continue

        fs = float(meta.get("fraud_score") or 0.0)
        cases.append(
            CaseListItem(
                claim_id=cid,
                case_status=cs,
                assigned_to=ato,
                decision=_decision_for_case(meta),  # type: ignore[arg-type]
                fraud_score=fs,
                risk_level=risk_level_from_claim_metadata(meta),
                review_status=_review_status_for_case(meta),
                timestamp=str(meta.get("timestamp") or "").strip(),
            )
        )

    cases.sort(key=lambda c: (c.timestamp or "", c.claim_id))
    return CaseListResponse(cases=cases)


@router.post("/cases/{claim_id}/assign")
async def assign_case(
    claim_id: str,
    body: CaseAssignRequest,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    existing = vector_store.get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found.")

    meta = dict(existing.get("metadata") or {})
    cs = _normalize_case_status(meta)
    if cs != "NEW":
        raise HTTPException(
            status_code=400,
            detail=f"Can only assign when case_status is NEW (current: {cs}).",
        )

    assignee = body.assigned_to.strip()
    now = _utc_now_iso()
    meta["assigned_to"] = assignee
    meta["assigned_at"] = now
    meta["case_status"] = "ASSIGNED"
    meta["updated_at"] = now

    try:
        vector_store.store_claim(
            claim_id=str(existing.get("claim_id") or claim_id),
            claim_description=str(existing.get("claim_description") or ""),
            embedding=list(existing.get("embedding") or []),
            metadata=meta,
        )
    except Exception as exc:
        logger.exception("case_assign_failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail=f"Failed to assign case: {type(exc).__name__}") from exc

    return {"ok": True, "claim_id": claim_id, "case_status": "ASSIGNED", "assigned_to": assignee}


@router.post("/cases/{claim_id}/status")
async def update_case_status(
    claim_id: str,
    body: CaseStatusUpdateRequest,
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    new_s = body.case_status
    if new_s == "ASSIGNED":
        raise HTTPException(
            status_code=400,
            detail="Use POST /cases/{claim_id}/assign with assigned_to to move NEW → ASSIGNED.",
        )

    existing = vector_store.get_claim(claim_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id!r} not found.")

    meta = dict(existing.get("metadata") or {})
    cur = _normalize_case_status(meta)

    if new_s == "IN_PROGRESS":
        if cur != "ASSIGNED":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transition to IN_PROGRESS (current case_status: {cur}).",
            )
    elif new_s == "RESOLVED":
        if cur != "IN_PROGRESS":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transition to RESOLVED (current case_status: {cur}).",
            )
    elif new_s == "NEW":
        raise HTTPException(status_code=400, detail="Cannot set case_status back to NEW via this endpoint.")

    now = _utc_now_iso()
    meta["case_status"] = new_s
    meta["updated_at"] = now

    try:
        vector_store.store_claim(
            claim_id=str(existing.get("claim_id") or claim_id),
            claim_description=str(existing.get("claim_description") or ""),
            embedding=list(existing.get("embedding") or []),
            metadata=meta,
        )
    except Exception as exc:
        logger.exception("case_status_failed", extra={"claim_id": claim_id})
        raise HTTPException(status_code=500, detail=f"Failed to update case status: {type(exc).__name__}") from exc

    return {"ok": True, "claim_id": claim_id, "case_status": new_s}
