from __future__ import annotations

from typing import Any

# Lightweight in-memory case list (post-decision add-on).
# NOTE: the production MVP already persists case status in Chroma metadata via `/cases` routes.
CASES: list[dict[str, Any]] = []


def create_case(claim_id: str, decision: str, score: float) -> dict[str, Any]:
    item = {
        "id": str(claim_id),
        "decision": str(decision),
        "score": float(score),
        "status": "OPEN",
    }
    CASES.append(item)
    return item


def update_case(claim_id: str, status: str) -> dict[str, Any] | None:
    cid = str(claim_id)
    for c in CASES:
        if str(c.get("id")) == cid:
            c["status"] = str(status)
            return c
    return None


__all__ = ["CASES", "create_case", "update_case"]

