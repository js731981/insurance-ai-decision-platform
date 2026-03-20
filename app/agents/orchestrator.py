from typing import Any

from fastapi import Depends

from app.agents.fraud_agent import FraudAgent
from app.core.dependencies import get_llm_service
from app.models.schemas import FraudAnalysisResponse, InferenceRequest, InferenceResponse
from app.services.llm_service import LLMService


class InsurFlowOrchestrator:
    """Decision-making layer.

    Orchestrators decide *what* to do (e.g., which model, which inputs, which workflow),
    while `app.services` executes the actual work (LLM calls, DB calls, external APIs).
    """

    def __init__(self, llm_service: LLMService):
        self._llm_service = llm_service
        # Decision-makers for each claim workflow step.
        self._fraud_agent = FraudAgent(llm_service=llm_service)

    async def run(self, data: Any) -> InferenceResponse | FraudAnalysisResponse:
        """Run the appropriate workflow for the incoming payload."""

        # 1) Existing inference workflow (kept for compatibility with /inference).
        if isinstance(data, InferenceRequest):
            request = data

            # Decision makers pick parameters and workflow; executors do the I/O.
            provider_override: str | None = None
            if request.task:
                task_l = request.task.strip().lower()
                # Simple cost-optimization hook (can be expanded later).
                if task_l == "cheap":
                    provider_override = "ollama"
                elif task_l == "complex":
                    provider_override = "openai"

            completion = await self._llm_service.generate(
                prompt=request.prompt,
                context=request.context,
                model=request.model,
                provider=provider_override,
            )
            return InferenceResponse(
                text=completion.text,
                provider=completion.provider,
                model=completion.model,
                tokens=completion.tokens,
                cost=completion.cost,
                latency=completion.latency_ms,
                confidence=completion.confidence,
            )

        # 2) Claim workflow (currently: fraud analysis).
        if isinstance(data, dict):
            return await self._fraud_agent.execute(data)

        raise TypeError(f"Unsupported orchestrator input type: {type(data).__name__}")


def get_insurflow_orchestrator(
    llm_service: LLMService = Depends(get_llm_service),
) -> InsurFlowOrchestrator:
    return InsurFlowOrchestrator(llm_service=llm_service)

