from __future__ import annotations

from typing import Any, Literal

from app.agents.base_agent import BaseAgent

DecisionLiteral = Literal["APPROVED", "REJECTED", "INVESTIGATE"]

# Micro-insurance: above this fraud score, queue for human / secondary review.
FRAUD_INVESTIGATE_THRESHOLD = 0.6
# When similar claims' human majority is APPROVED, require a higher bar to escalate (lean approve).
FRAUD_APPROVED_PATTERN_ESCALATE_THRESHOLD = 0.72
# Regardless of approved pattern, always investigate if fraud signals are this strong.
FRAUD_STRONG_CONTRADICTION_THRESHOLD = 0.84


class DecisionAgent(BaseAgent):
    """Single triage outcome for a micro-insurance claim from fraud + policy signals."""

    def __init__(self, *, fraud_investigate_threshold: float = FRAUD_INVESTIGATE_THRESHOLD) -> None:
        super().__init__()
        self._fraud_threshold = fraud_investigate_threshold

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        fraud_block: dict[str, Any] = input_data.get("fraud") or {}
        policy_block: dict[str, Any] = input_data.get("policy") or {}
        similar_majority = input_data.get("similar_majority_review")
        if similar_majority is not None:
            similar_majority = str(similar_majority).strip().upper() or None
        if similar_majority not in (None, "APPROVED", "REJECTED"):
            similar_majority = None

        fraud_score_raw = fraud_block.get("fraud_score", 0.0)
        try:
            fraud_score = float(fraud_score_raw)
        except (TypeError, ValueError):
            fraud_score = 0.0
        fraud_score = min(1.0, max(0.0, fraud_score))

        policy_valid = bool(policy_block.get("policy_valid"))
        policy_reason = str(policy_block.get("policy_reason") or "")
        fraud_explanation = _fraud_explanation_text(fraud_block)

        if not policy_valid:
            return {
                "decision": "REJECTED",
                "confidence_score": 0.9,
                "explanation": f"Policy check failed. {policy_reason}".strip(),
            }

        if fraud_score >= FRAUD_STRONG_CONTRADICTION_THRESHOLD:
            conf = min(1.0, max(0.5, fraud_score))
            return {
                "decision": "INVESTIGATE",
                "confidence_score": round(conf, 3),
                "explanation": (
                    f"Strong fraud signals (score={fraud_score:.2f}) override similar-case approval pattern. "
                    f"{fraud_explanation}".strip()
                ),
            }

        escalate_threshold = self._fraud_threshold
        pattern_note = ""
        if similar_majority == "APPROVED":
            escalate_threshold = FRAUD_APPROVED_PATTERN_ESCALATE_THRESHOLD
            pattern_note = " Similar reviewed claims were mostly approved; higher bar to escalate."

        if fraud_score >= escalate_threshold:
            conf = min(1.0, max(0.5, fraud_score))
            return {
                "decision": "INVESTIGATE",
                "confidence_score": round(conf, 3),
                "explanation": (
                    f"Elevated fraud signals (score={fraud_score:.2f}).{pattern_note} {fraud_explanation}".strip()
                ),
            }

        conf = min(1.0, max(0.5, 1.0 - fraud_score))
        approve_note = ""
        if similar_majority == "APPROVED":
            approve_note = (
                " Reviewed similar claims were mostly approved; leaning approve without strong fraud contradiction."
            )
            conf = min(1.0, conf * 1.06)
        expl = (
            f"Policy valid and fraud score acceptable ({fraud_score:.2f}).{approve_note} "
            f"{fraud_explanation}".strip()
        )
        return {
            "decision": "APPROVED",
            "confidence_score": round(conf, 3),
            "explanation": expl.strip(),
        }


def _fraud_explanation_text(fraud_block: dict[str, Any]) -> str:
    raw = fraud_block.get("explanation")
    if isinstance(raw, dict):
        summary = str(raw.get("summary") or "").strip()
        factors = raw.get("key_factors")
        lines: list[str] = []
        if summary:
            lines.append(summary)
        if isinstance(factors, list):
            for f in factors[:4]:
                t = str(f).strip()
                if t:
                    lines.append(f"- {t}")
        ref = str(raw.get("similar_case_reference") or "").strip()
        if ref:
            lines.append(f"Similar cases: {ref}")
        if lines:
            return "\n".join(lines)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return str(fraud_block.get("fraud_reason") or "").strip()
