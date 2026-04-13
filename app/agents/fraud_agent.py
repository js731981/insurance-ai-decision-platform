from __future__ import annotations

import json
import logging
import re
from typing import Any, NamedTuple, Optional, Tuple, TypedDict

from app.agents.base_agent import BaseAgent
from app.core.config import settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

_RAW_LOG_MAX_CHARS = 12_000


def _truncate_for_log(text: str, max_chars: int = _RAW_LOG_MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


class FraudParseResult(NamedTuple):
    fraud_score: float
    decision: str
    confidence: float
    explanation: dict[str, Any]
    entities: dict[str, Any]
    ok: bool
    error: Optional[str]


class FraudAgentInput(TypedDict, total=False):
    """Orchestrator-supplied payload (arbitrary claim keys allowed via ``total=False`` + extras).

    ``similar_claims_context`` is produced by the RAG :class:`~app.services.context_builder.ContextBuilder`.
    """

    claim_id: str
    description: str
    claim_amount: float
    policy_limit: float
    currency: str
    product_code: str
    incident_date: str
    policyholder_id: str
    rag_filter_decision: str
    rag_metadata_filter: dict[str, Any]
    similar_claims_context: str


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


def _claim_description_for_llm(input_data: dict[str, Any]) -> str:
    """Minimal claim text for the fraud prompt (description only)."""
    desc = str(input_data.get("description") or "").strip()
    if desc:
        return desc
    cid = str(input_data.get("claim_id") or "").strip()
    if cid:
        return f"(No claim description provided; claim_id={cid}.)"
    return "(No claim description provided.)"


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _extract_json_balanced(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    quote_ch = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_ch:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote_ch = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_object_loose(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _repair_json_candidate(candidate: str) -> str:
    t = candidate.strip()
    if t.startswith("\ufeff"):
        t = t[1:]
    for _ in range(8):
        n = re.sub(r",\s*}", "}", t)
        n = re.sub(r",\s*]", "]", n)
        if n == t:
            break
        t = n
    return t


class FraudAgent(BaseAgent):
    """Fraud signals for micro-insurance claims (LLM-assisted, JSON score + structured explanation)."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
    ) -> None:
        super().__init__()
        self._llm_service = llm_service

    def _build_fraud_prompt(self, *, strict_retry: bool) -> str:
        retry_prefix = ""
        if strict_retry:
            retry_prefix = (
                "Your previous response was invalid. Return ONLY valid JSON.\n\n"
            )
        schema_example = (
            "Your JSON MUST match this shape (replace example values; keep all keys):\n"
            "{\n"
            '  "fraud_score": 0.0,\n'
            '  "decision": "APPROVED",\n'
            '  "confidence": 0.0,\n'
            '  "entities": {},\n'
            '  "explanation": {\n'
            '    "summary": "string",\n'
            '    "key_factors": ["string", "string"]\n'
            "  }\n"
            "}\n"
            'Use "decision" as exactly one of: "APPROVED", "INVESTIGATE", "REJECTED".\n'
            "Optional inside explanation: similar_case_reference (string; empty if none).\n"
        )
        instructions = (
            "You are a micro-insurance fraud analyst.\n\n"
            "You MUST return ONLY valid JSON.\n"
            "Do NOT include any explanation outside JSON.\n"
            "Do NOT include markdown or text.\n"
            "If unsure, still return valid JSON.\n\n"
            f"{schema_example}\n"
            "Rules: fraud_score 0=low risk, 1=high risk. Missing info → lower score, note in key_factors. "
            "explanation.summary must be non-empty. explanation.key_factors must have at least 2 strings. "
            "With RAG context lines, you may set similar_case_reference to a one-line echo of those priors.\n"
        )
        return retry_prefix + instructions

    def _build_fraud_context(self, *, claim_description: str, similar_claims_context: str) -> str:
        parts = [f"Claim description:\n{claim_description}"]
        similar = similar_claims_context.strip()
        if similar:
            parts.append(f"Similar claims (compact retrieval context; not ground truth):\n{similar}")
        return "\n\n".join(parts)

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        claim_id = str(input_data.get("claim_id") or "").strip() or "unknown"
        claim_description = _claim_description_for_llm(input_data)
        similar = str(input_data.get("similar_claims_context") or "").strip()

        max_parse_retries = settings.max_llm_retries if settings.strict_json_mode else 0
        total_attempts = 1 + max(0, max_parse_retries)

        total_latency_ms = 0
        last_raw = ""

        for attempt in range(total_attempts):
            strict_retry = attempt > 0
            prompt = self._build_fraud_prompt(strict_retry=strict_retry)
            context = self._build_fraud_context(
                claim_description=claim_description,
                similar_claims_context=similar,
            )
            gen_kw: dict[str, Any] = {"temperature": 0.1 if strict_retry else 0.2}

            try:
                completion = await self._llm_service.generate(
                    prompt=prompt,
                    context=context,
                    generation_kwargs=gen_kw,
                    claim_id=claim_id,
                )
            except Exception as exc:
                logger.exception(
                    "fraud_llm_failed",
                    extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                is_timeout = type(exc).__name__ == "TimeoutError" or "timeout" in str(exc).lower()
                summary = (
                    "Automated fraud analysis timed out; escalate for manual review."
                    if is_timeout
                    else "Automated fraud analysis failed; escalate for manual review."
                )
                return {
                    "fraud_score": 0.5,
                    "decision": "INVESTIGATE",
                    "confidence": 0.5,
                    "entities": {},
                    "explanation": {
                        "summary": summary,
                        "key_factors": [
                            f"LLM or transport error: {type(exc).__name__}.",
                            "No structured fraud assessment was produced.",
                        ],
                        "similar_case_reference": "",
                    },
                    "_llm_failed": True,
                    "_llm_latency_ms": 0,
                }

            last_raw = completion.text or ""
            total_latency_ms += int(completion.latency_ms)

            raw_preview, raw_trunc = _truncate_for_log(last_raw)
            logger.info(
                "fraud_llm_raw_output",
                extra={
                    "claim_id": claim_id,
                    "attempt": attempt + 1,
                    "strict_retry": strict_retry,
                    "raw_char_len": len(last_raw),
                    "raw_truncated": raw_trunc,
                    "raw_text": raw_preview,
                },
            )

            prepared = self._prepare_model_text(last_raw)
            result = self._parse_fraud_response(prepared)
            if result.ok:
                out: dict[str, Any] = {
                    "fraud_score": result.fraud_score,
                    "decision": result.decision,
                    "confidence": result.confidence,
                    "entities": result.entities,
                    "explanation": result.explanation,
                    "_llm_latency_ms": total_latency_ms,
                }
                return out

            logger.warning(
                "fraud_llm_json_parse_failed",
                extra={
                    "claim_id": claim_id,
                    "attempt": attempt + 1,
                    "error": result.error,
                },
            )

            if attempt < total_attempts - 1:
                logger.info(
                    "fraud_llm_json_retry_scheduled",
                    extra={
                        "claim_id": claim_id,
                        "from_attempt": attempt + 1,
                        "to_attempt": attempt + 2,
                        "max_attempts": total_attempts,
                    },
                )

        logger.warning(
            "fraud_llm_json_parse_exhausted",
            extra={
                "claim_id": claim_id,
                "attempts": total_attempts,
                "last_error": result.error,
            },
        )

        d = _default_explanation()
        return {
            "fraud_score": 0.5,
            "decision": "INVESTIGATE",
            "confidence": 0.5,
            "entities": {},
            "explanation": d,
            "_llm_failed": True,
            "_llm_latency_ms": total_latency_ms,
        }

    def _prepare_model_text(self, text: str) -> str:
        t = _strip_markdown_json_fence(text)
        return t.strip()

    def _extract_json_candidate(self, text: str) -> Optional[str]:
        balanced = _extract_json_balanced(text)
        if balanced:
            return balanced
        return _extract_json_object_loose(text)

    def _loads_json_with_repair(self, candidate: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            return None, "json_root_not_object"
        except json.JSONDecodeError as first_exc:
            repaired = _repair_json_candidate(candidate)
            if repaired != candidate:
                try:
                    obj = json.loads(repaired)
                    if isinstance(obj, dict):
                        return obj, None
                    return None, "json_root_not_object_after_repair"
                except json.JSONDecodeError as exc:
                    return None, f"json_decode_error_after_repair: {exc}"
            return None, f"json_decode_error: {first_exc}"

    def _parse_fraud_response(self, prepared_text: str) -> FraudParseResult:
        candidate = self._extract_json_candidate(prepared_text)
        if not candidate:
            return FraudParseResult(
                0.5,
                "INVESTIGATE",
                0.5,
                _default_explanation(),
                {},
                False,
                "no_json_object_extracted",
            )

        payload, err = self._loads_json_with_repair(candidate)
        if payload is None:
            return FraudParseResult(
                0.5,
                "INVESTIGATE",
                0.5,
                _default_explanation(),
                {},
                False,
                err or "json_load_failed",
            )

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
        if not ok:
            return FraudParseResult(
                fraud_score,
                decision,
                confidence,
                explanation,
                entities,
                False,
                "schema_incomplete_explanation",
            )

        return FraudParseResult(
            fraud_score,
            decision,
            confidence,
            explanation,
            entities,
            True,
            None,
        )

    def _parse_fraud_json(self, text: str) -> Tuple[float, str, float, dict[str, Any], dict[str, Any], bool]:
        """Back-compat tuple parse (used internally; same semantics as :meth:`_parse_fraud_response`)."""
        r = self._parse_fraud_response(self._prepare_model_text(text))
        return r.fraud_score, r.decision, r.confidence, r.explanation, r.entities, r.ok


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


__all__ = ["FraudAgent", "FraudAgentInput"]
