from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitlDecision:
    needs_hitl: bool
    reason: str | None = None


class HitlService:
    """Human-in-the-loop policy (kept out of agents)."""

    def __init__(self, *, confidence_threshold: float = 0.75) -> None:
        self._confidence_threshold = confidence_threshold

    def evaluate(self, *, decision: str, confidence: float) -> HitlDecision:
        if (decision or "").strip().upper() == "INVESTIGATE":
            return HitlDecision(needs_hitl=True, reason="Decision escalated to INVESTIGATE.")
        if confidence < self._confidence_threshold:
            return HitlDecision(needs_hitl=True, reason=f"Low confidence (< {self._confidence_threshold}).")
        return HitlDecision(needs_hitl=False)

