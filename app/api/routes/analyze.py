from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agents.orchestrator import InsurFlowOrchestrator, get_insurflow_orchestrator
from app.services.post_decision_service import enhance_after_decision

router = APIRouter(tags=["post-decision"])


@router.post("/analyze")
async def analyze(
    data: dict[str, Any],
    orchestrator: InsurFlowOrchestrator = Depends(get_insurflow_orchestrator),
) -> dict[str, Any]:
    """Compatibility endpoint for "enhanced" analysis (POST-DECISION only).

    This does not replace `/claims` and must not change the core decision logic.
    """
    try:
        core = await orchestrator.process_claim(dict(data or {}))
        return enhance_after_decision(input_data=dict(data or {}), core_result=core)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analyze failed: {type(exc).__name__}") from exc

