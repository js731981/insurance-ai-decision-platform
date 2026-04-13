"""Vector retrieval for RAG.

Delegates to :class:`app.services.vector_store.VectorStore` using the same weighted query
path as :meth:`~app.services.vector_store.VectorStore.query_similar_for_context`
(:meth:`~app.services.vector_store.VectorStore.query_similar_hits`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from app.services.vector_store import SimilarHit, VectorStore

logger = logging.getLogger(__name__)


def build_chroma_where(
    *,
    decision_equal: Optional[str] = None,
    metadata_equal: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Map simple equality filters to a Chroma ``where`` clause (extensible for Qdrant later)."""
    clauses: list[dict[str, Any]] = []

    d = (decision_equal or "").strip().upper()
    if d in ("APPROVED", "REJECTED", "INVESTIGATE"):
        clauses.append({"decision": d})

    for k, v in (metadata_equal or {}).items():
        if v is None or v == "" or not str(k).strip():
            continue
        key = str(k).strip()
        if isinstance(v, bool):
            clauses.append({key: v})
        elif isinstance(v, (int, float)):
            clauses.append({key: v})
        else:
            clauses.append({key: str(v)})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _hit_matches_product_code(hit: SimilarHit, product_code: str) -> bool:
    code = (product_code or "").strip()
    if not code:
        return True
    ej = str(hit.metadata.get("entities_json") or "")
    if code in ej:
        return True
    try:
        parsed = json.loads(ej) if ej else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    for key in ("product", "product_code", "productCode"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip().upper() == code.upper():
            return True
        if val is not None and str(val).strip() == code:
            return True
    return False


@dataclass(frozen=True)
class RetrievalParams:
    """Inputs for vector retrieval (embedding is computed once upstream)."""

    claim_description: str
    query_embedding: list[float]
    exclude_claim_id: Optional[str] = None
    top_k: int = 3
    decision_equal: Optional[str] = None
    metadata_equal: Optional[dict[str, Any]] = None
    product_code_equal: Optional[str] = None


class ClaimRetriever:
    """Vector retriever abstraction over :class:`VectorStore`.

    Uses the same query path as :meth:`VectorStore.query_similar_for_context`
    (:meth:`VectorStore.query_similar_hits` + weighting), optionally narrowed by metadata.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(self, params: RetrievalParams) -> list[SimilarHit]:
        desc = (params.claim_description or "").strip()
        top_k = max(1, min(25, int(params.top_k)))
        where = build_chroma_where(
            decision_equal=params.decision_equal,
            metadata_equal=params.metadata_equal,
        )
        product_code = (params.product_code_equal or "").strip()

        n_fetch = top_k
        if product_code:
            n_fetch = min(25, max(top_k * 4, top_k))

        hits = self._vector_store.query_similar_hits(
            query_embedding=params.query_embedding,
            exclude_claim_id=params.exclude_claim_id,
            n_results=n_fetch,
            where=where,
        )

        if product_code:
            filtered = [h for h in hits if _hit_matches_product_code(h, product_code)]
            if not filtered and hits:
                logger.info(
                    "retriever_product_filter_no_match",
                    extra={"product_code": product_code, "candidates": len(hits)},
                )
            hits = filtered

        out = hits[:top_k]
        logger.debug(
            "retriever_complete",
            extra={
                "claim_description_len": len(desc),
                "where": bool(where),
                "product_filter": bool(product_code),
                "returned": len(out),
            },
        )
        return out
