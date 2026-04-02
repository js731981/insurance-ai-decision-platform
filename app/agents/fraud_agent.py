from __future__ import annotations

import json
import logging
from typing import Any, Optional, Tuple

from app.agents.base_agent import BaseAgent
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _default_explanation() -> dict[str, Any]:
    """Shown when the model response is missing, non-JSON, or fails schema checks (aligns with DecisionAgent fallback)."""
    return {
        "summary": (
            "Fraud model did not return usable structured output; treat as escalate for manual review."
        ),
        "key_factors": [
            "Response was not valid JSON or omitted required fields (e.g. explanation.summary, two or more key_factors).",
            "Automated triage cannot rely on this output; same outcome as fraud model unavailable.",
        ],
        "similar_case_reference": "",
    }


class FraudAgent(BaseAgent):
    """Fraud signals for micro-insurance claims (LLM-assisted, JSON score + structured explanation)."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
    ) -> None:
        super().__init__()
        self._llm_service = llm_service

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        claim_id = str(input_data.get("claim_id") or "").strip() or "unknown"
        similar = str(input_data.get("similar_claims_context") or "").strip()
        similar_block = (
            f"\n\nSimilar claims from your organisation's memory (use for patterns only):\n{similar}\n"
            if similar
            else ""
        )

        prompt = (
            "You are a fraud analyst for a micro-insurance product (small sums, high volume, "
            "often mobile-first or instant payouts).\n"
            "Use the claim details below. If similar past claims are provided, compare patterns "
            "and anomalies; give more weight to cases that already have a human review. "
            "In that case, explicitly reference the pattern in similar_case_reference using wording like "
            "\"This pattern is similar to previous approved claims\" or "
            "\"This pattern is similar to previous rejected claims\" (choose approved vs rejected to match "
            "what the similar cases indicate).\n"
            "Analyze the claim and output ONLY valid JSON.\n"
            "The JSON schema is exactly:\n"
            "{\n"
            '  "fraud_score": number between 0 and 1,\n'
            '  "decision": one of APPROVED, REJECTED, INVESTIGATE (your fraud triage recommendation),\n'
            '  "confidence": number between 0 and 1 (your confidence in fraud_score and decision),\n'
            '  "entities": object with short string values, e.g. '
            '{"product": "...", "amount_band": "...", "flags": "..."},\n'
            '  "explanation": {\n'
            '    "summary": "1-2 short lines summarizing fraud risk for a non-technical reader",\n'
            '    "key_factors": ["2 to 4 short bullet strings: concrete factors, no repetition"],\n'
            '    "similar_case_reference": "If similar claims were provided: one line, e.g. '
            'This pattern is similar to previous approved/rejected claims; else empty string"\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- Output JSON only (no markdown fences, no surrounding text).\n"
            "- fraud_score=0.0 means no fraud signals; 1.0 means strong fraud signals.\n"
            "- summary must be 1-2 lines; key_factors must have 2-4 items.\n"
            "- Routine legitimate micro-claims should score low unless something is off.\n"
            "- If information is missing, lower fraud_score and explain uncertainty in key_factors.\n"
            "- If similar claims are provided in context, you MUST include similar_case_reference with phrasing "
            "such as \"This pattern is similar to previous approved claims\" or "
            "\"This pattern is similar to previous rejected claims\" (aligned with those cases).\n"
        )

        claim_json = json.dumps(
            {k: v for k, v in input_data.items() if k != "similar_claims_context"},
            ensure_ascii=False,
        )

        try:
            completion = await self._llm_service.generate(
                prompt=prompt,
                context=f"Claim payload (JSON):\n{claim_json}{similar_block}",
                generation_kwargs={"temperature": 0.2},
                claim_id=claim_id,
            )
        except BaseException as exc:  # noqa: BLE001
            logger.exception(
                "fraud_llm_failed",
                extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
            )
            return {
                "fraud_score": 0.5,
                "decision": "INVESTIGATE",
                "confidence": 0.5,
                "entities": {},
                "explanation": {
                    "summary": "Automated fraud analysis failed; escalate for manual review.",
                    "key_factors": [
                        f"LLM or transport error: {type(exc).__name__}.",
                        "No structured fraud assessment was produced.",
                    ],
                    "similar_case_reference": "",
                },
                "_llm_failed": True,
            }

        parsed = self._parse_fraud_json(completion.text)
        fraud_score, decision, confidence, explanation, entities, parse_ok = parsed
        out: dict[str, Any] = {
            "fraud_score": fraud_score,
            "decision": decision,
            "confidence": confidence,
            "entities": entities if isinstance(entities, dict) else {},
            "explanation": explanation,
        }
        if not parse_ok:
            out["_llm_failed"] = True
            out["decision"] = "INVESTIGATE"
            out["confidence"] = 0.5
        return out

    def _parse_fraud_json(self, text: str) -> Tuple[float, str, float, dict[str, Any], dict[str, Any], bool]:
        candidate = self._extract_json_object(text)
        if not candidate:
            d = _default_explanation()
            return 0.5, "INVESTIGATE", 0.5, d, {}, False

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            d = _default_explanation()
            return 0.5, "INVESTIGATE", 0.5, d, {}, False

        fraud_score_raw: Optional[Any] = payload.get("fraud_score")
        decision_raw: Optional[Any] = payload.get("decision")
        confidence_raw: Optional[Any] = payload.get("confidence")
        explanation_raw: Optional[Any] = payload.get("explanation")
        entities_raw: Optional[Any] = payload.get("entities")

        fraud_score = 0.0
        try:
            if isinstance(fraud_score_raw, (int, float, str)):
                fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0
        fraud_score = min(1.0, max(0.0, fraud_score))

        decision = "INVESTIGATE"
        if isinstance(decision_raw, str) and decision_raw.strip():
            d_u = decision_raw.strip().upper()
            if d_u in ("APPROVED", "REJECTED", "INVESTIGATE"):
                decision = d_u

        confidence = 0.5
        try:
            if isinstance(confidence_raw, (int, float, str)):
                confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))

        explanation = _normalize_explanation(explanation_raw, payload)

        entities: dict[str, Any] = {}
        if isinstance(entities_raw, dict):
            entities = {str(k): v for k, v in entities_raw.items()}

        factors_list = explanation.get("key_factors")
        ok = bool(
            str(explanation.get("summary") or "").strip()
            and isinstance(factors_list, list)
            and len(factors_list) >= 2
        )
        return fraud_score, decision, confidence, explanation, entities, ok

    def _extract_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]


def _normalize_explanation(explanation_raw: Any, payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(explanation_raw, dict):
        summary = str(explanation_raw.get("summary") or "").strip()
        kf = explanation_raw.get("key_factors")
        factors: list[str] = []
        if isinstance(kf, list):
            factors = [str(x).strip() for x in kf if str(x).strip()][:8]
        ref = str(explanation_raw.get("similar_case_reference") or "").strip()
        if not summary:
            summary = "No summary provided."
        if len(factors) < 2:
            legacy = payload.get("explanation")
            if isinstance(legacy, str) and legacy.strip():
                factors = [legacy.strip()]
            if len(factors) < 2:
                factors = (factors + ["Insufficient structured explanation from model."])[:4]
        return {"summary": summary, "key_factors": factors[:4], "similar_case_reference": ref}

    legacy = payload.get("explanation")
    if isinstance(legacy, str) and legacy.strip():
        return {
            "summary": legacy.strip()[:300],
            "key_factors": [legacy.strip()[:500]],
            "similar_case_reference": "",
        }
    return _default_explanation()
