from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from app.agents.base import BaseAgent
from app.models.schemas import FraudAnalysisResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class FraudAgent(BaseAgent):
    """Fraud analysis agent.

    For now it uses the LLM as the analysis engine and expects JSON output:
    `{"fraud_score": <0..1>, "reason": "<string>"}`
    """

    def __init__(self, *, llm_service: LLMService) -> None:
        super().__init__(llm_service=llm_service)

    async def execute(self, input_data: Dict[str, Any]) -> FraudAnalysisResponse:
        prompt = (
            "You are an insurance claim fraud analyst.\n"
            "Analyze the given claim payload and output ONLY valid JSON.\n"
            "The JSON schema is exactly:\n"
            "{\n"
            '  "fraud_score": a number between 0 and 1,\n'
            '  "reason": a concise explanation (1-3 sentences) grounded in the claim fields.\n'
            "}\n"
            "Rules:\n"
            "- Output JSON only (no markdown, no surrounding text).\n"
            "- fraud_score=0.0 means no fraud signals; 1.0 means strong fraud signals.\n"
            "- If information is missing, lower fraud_score and explain uncertainty.\n"
        )

        claim_json = json.dumps(input_data, ensure_ascii=False)

        try:
            completion = await self._llm_service.generate(
                prompt=prompt,
                context=f"Claim payload (JSON):\n{claim_json}",
                generation_kwargs={"temperature": 0.2},
            )
        except BaseException as exc:  # noqa: BLE001 - return a safe structured response
            logger.exception("LLM call failed in FraudAgent")
            return FraudAnalysisResponse(
                fraud_score=0.0,
                reason=f"LLM call failed: {type(exc).__name__}: {exc}",
            )

        fraud_score, reason = self._parse_fraud_json(completion.text)
        return FraudAnalysisResponse(fraud_score=fraud_score, reason=reason)

    def _parse_fraud_json(self, text: str) -> Tuple[float, str]:
        candidate = self._extract_json_object(text)
        if not candidate:
            return 0.0, "Could not parse LLM response as JSON."

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return 0.0, "LLM response contained invalid JSON."

        fraud_score_raw: Optional[Any] = payload.get("fraud_score")
        reason_raw: Optional[Any] = payload.get("reason")

        # Be defensive: coerce score to float in range [0,1].
        fraud_score = 0.0
        try:
            if isinstance(fraud_score_raw, (int, float, str)):
                fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0

        fraud_score = min(1.0, max(0.0, fraud_score))

        reason = "No reason provided by model."
        if isinstance(reason_raw, str) and reason_raw.strip():
            reason = reason_raw.strip()
        else:
            # Keep a readable error trail without leaking raw model text.
            reason = "Model did not provide a valid `reason` field."

        return fraud_score, reason

    def _extract_json_object(self, text: str) -> Optional[str]:
        # Prefer the first top-level JSON object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

