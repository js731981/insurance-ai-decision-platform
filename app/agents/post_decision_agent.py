from __future__ import annotations

from typing import Any


def plan_steps(description: str) -> list[str]:
    """Tiny, pluggable post-decision planner.

    Note: the core pipeline is owned by `InsurFlowOrchestrator`. This planner is only
    for post-decision enrichment (trace, optional RAG/LLM explanation).
    """
    steps = ["core_decision"]
    desc_l = str(description or "").lower()
    if "damage" in desc_l or "crack" in desc_l:
        steps.append("rag")
    steps.append("llm_explain")
    return steps


def reflect(fraud_score: Any) -> str:
    """Simple reflection hook for retries (post-decision only)."""
    try:
        s = float(fraud_score)
    except (TypeError, ValueError):
        s = 0.5
    if s < 0.5:
        return "retry_rag"
    return "done"


__all__ = ["plan_steps", "reflect"]

