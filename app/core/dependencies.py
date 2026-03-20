from app.core.config import settings
from app.services.llm_service import LLMService


def get_llm_service() -> LLMService:
    # Factory so each request gets its own service instance (swap later if you want pooling).
    return LLMService(model_name=settings.model_name)

