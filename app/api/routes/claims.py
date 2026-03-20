from fastapi import APIRouter, Depends

from app.agents.orchestrator import get_insurflow_orchestrator
from app.models.schemas import FraudAnalysisResponse

router = APIRouter()


@router.post("/claim", response_model=FraudAnalysisResponse, tags=["claims"])
async def process_claim(data: dict, orchestrator=Depends(get_insurflow_orchestrator)) -> FraudAnalysisResponse:
    return await orchestrator.run(data)

