from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.services.vector_store import SimilarHit

logger = logging.getLogger(__name__)


class LightweightReranker:
    """Cheap re-ranking on top of vector scores (no cross-encoder).

    Boosts hits whose stored entities mention the same product code as the inbound claim.
    """

    def rerank(
        self,
        hits: list[SimilarHit],
        *,
        claim: Optional[dict[str, Any]] = None,
        product_code: Optional[str] = None,
    ) -> list[SimilarHit]:
        code = (product_code or "").strip()
        if not code and claim is not None:
            raw = claim.get("product_code")
            code = str(raw).strip() if raw is not None else ""

        if not code or not hits:
            return list(hits)

        def bonus(hit: SimilarHit) -> float:
            b = 0.0
            ej = str(hit.metadata.get("entities_json") or "")
            if code in ej:
                b += 0.03
            try:
                ent = json.loads(ej) if ej else {}
            except json.JSONDecodeError:
                ent = {}
            if isinstance(ent, dict):
                for key in ("product", "product_code", "productCode"):
                    val = ent.get(key)
                    if isinstance(val, str) and val.strip().upper() == code.upper():
                        b += 0.04
                        break
                    if val is not None and str(val).strip() == code:
                        b += 0.04
                        break
            return b

        scored = [(h.adjusted_score + bonus(h), h) for h in hits]
        scored.sort(key=lambda t: t[0], reverse=True)
        out = [h for _, h in scored]
        logger.debug("reranker_applied", extra={"product_code": code, "count": len(out)})
        return out
