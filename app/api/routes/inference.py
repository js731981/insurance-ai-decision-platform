from fastapi import APIRouter, Depends

from app.agents.orchestrator import get_insurflow_orchestrator
from app.models.schemas import InferenceRequest, InferenceResponse

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse, tags=["inference"])
async def inference(
    request: InferenceRequest,
    orchestrator=Depends(get_insurflow_orchestrator),
) -> InferenceResponse:
    # API layer should stay thin: validate/parse request -> call orchestrator -> return response.
    return await orchestrator.run(request)

