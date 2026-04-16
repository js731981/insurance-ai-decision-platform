from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from utils.memory_store import get_similar_claims, store_claim
from utils.explanation_engine import build_production_explanation, format_explanation_ui
from utils.formatters import sanitize_output


@dataclass(frozen=True)
class PipelineResult:
    _data: Mapping[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def run_demo_logic(claim_id, description, amount, policy_limit, image=None) -> PipelineResult:
    desc = (description or "").lower()

    similar_claims = get_similar_claims(description or "", float(amount or 0.0))
    similar_claims = sanitize_output(similar_claims)

    if "major" in desc:
        severity = "high"
        fraud_score = 0.7
        cnn_conf = 0.88
    elif "minor" in desc:
        severity = "medium"
        fraud_score = 0.4
        cnn_conf = 0.72
    else:
        severity = "low"
        fraud_score = 0.2
        cnn_conf = 0.55

    decision = "APPROVED" if fraud_score < 0.5 else "INVESTIGATE"
    if fraud_score < 0.3:
        risk_level = "LOW"
    elif fraud_score < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    exp = build_production_explanation(
        {
            "claim_id": claim_id,
            "severity": severity,
            "amount": amount,
            "policy_limit": policy_limit,
            "decision": decision,
        },
        similar_claims,
        fraud_score,
    )
    formatted_exp = format_explanation_ui(exp)

    store_claim(
        {
            "claim_id": claim_id,
            "description": description,
            "amount": amount,
            "decision": decision,
            "fraud_score": fraud_score,
        }
    )

    data = sanitize_output(
        {
            "decision": decision,
            "risk_level": risk_level,
            "fraud_score": fraud_score,
            "explanation": formatted_exp,
        }
    )
    return PipelineResult(data)

