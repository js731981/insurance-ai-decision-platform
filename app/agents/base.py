from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from app.services.llm_service import LLMService


class BaseAgent(ABC):
    """Base class for all agents (decision makers).

    Agents should be transport-agnostic and return structured outputs.
    """

    def __init__(self, *, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Any:
        """Run the agent on the given input payload."""
        raise NotImplementedError

