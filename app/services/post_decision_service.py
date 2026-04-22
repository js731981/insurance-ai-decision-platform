from __future__ import annotations

import json
from typing import Any

from app.agents.post_decision_agent import plan_steps, reflect
from app.services.llm_service import generate_explanation
from app.services.rag_service import retrieve_similar, store_claim


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def enhance_after_decision(*, input_data: dict[str, Any], core_result: dict[str, Any]) -> dict[str, Any]:
    """POST-DECISION enhancement layer (non-blocking, modular).

    Does NOT change the decision; it only adds trace, optional RAG context, and optional LLM explanation.
    """
    desc = str(input_data.get("description") or "").strip()
    claim_id = str(input_data.get("claim_id") or core_result.get("claim_id") or "").strip() or "unknown"

    decision = str(core_result.get("decision") or "").strip() or "UNKNOWN"
    fraud_score = (
        core_result.get("agent_outputs", {})
        .get("decision", {})
        .get("fused_fraud_score", None)
    )
    if fraud_score is None:
        fraud_score = core_result.get("agent_outputs", {}).get("fraud", {}).get("fraud_score", None)
    fraud_score_f = _safe_float(fraud_score, default=_safe_float(core_result.get("fraud_score"), default=0.0))

    trace: list[str] = []
    rag_data: list[dict[str, Any]] = []

    steps = plan_steps(desc)
    for step in steps:
        if step == "core_decision":
            trace.append(f"Core → {decision}")
        elif step == "rag":
            rag_data = retrieve_similar(desc, k=3)
            trace.append(f"RAG → {len(rag_data)} matches")
        elif step == "llm_explain":
            trace.append("LLM → explanation requested")

    if reflect(fraud_score_f) == "retry_rag":
        rag_data = retrieve_similar(desc, k=3)
        trace.append("Reflection → RAG retry")

    # Explanation input sources: keep stable and compact.
    summary = ""
    expl = core_result.get("agent_outputs", {}).get("fraud", {}).get("explanation")
    if isinstance(expl, dict):
        summary = str(expl.get("summary") or "").strip()
    elif isinstance(expl, str):
        summary = expl.strip()
    if not summary:
        summary = str(core_result.get("metadata", {}).get("explanation") or "").strip() or "No summary available."

    rules = {
        "decision_source": core_result.get("decision_source"),
        "contributors": (core_result.get("metadata") or {}).get("contributors"),
        "hitl_needed": core_result.get("hitl_needed"),
        "fraud_signal": core_result.get("fraud_signal"),
    }
    rules_text = json.dumps(rules, ensure_ascii=False)

    llm_text = generate_explanation(summary, rules_text, rag_data)

    # Best-effort post-decision memory store (separate collection).
    store_claim(
        {
            "claim_id": claim_id,
            "description": desc,
            "decision": decision,
            "fraud_score": fraud_score_f,
        }
    )

    return {
        **core_result,
        "fraud_score": fraud_score_f,
        "trace": trace,
        "rag": rag_data,
        "llm": llm_text,
    }


__all__ = ["enhance_after_decision"]

