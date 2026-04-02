from typing import Any, Dict, Optional

from app.core.config import settings
from app.services.llm.providers.openai_provider import OpenAIProvider
from app.services.llm.providers.openrouter_provider import OpenRouterProvider
from app.services.llm.providers.ollama_provider import OllamaProvider
from app.services.llm.router import LLMCompletion, LLMRouter, RetryPolicy


class LLMService:
    """Executor for LLM calls.

    Per layering rules:
    - `app.agents` decide what to do (decision makers).
    - `app.services` execute the work (LLM, DB, external APIs).
    """

    def __init__(self, model_name: str):
        self._default_model_name = model_name

        fallback = [p.strip() for p in settings.llm_fallback_providers.split(",") if p.strip()]

        providers: Dict[str, Any] = {
            "ollama": OllamaProvider(
                base_url=settings.ollama_base_url,
                timeout_s=settings.llm_timeout_s,
            ),
        }
        if settings.openai_api_key:
            providers["openai"] = OpenAIProvider(
                api_key=settings.openai_api_key,
                timeout_s=settings.llm_timeout_s,
            )
        if settings.openrouter_api_key:
            providers["openrouter"] = OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                timeout_s=settings.llm_timeout_s,
            )

        self._router = LLMRouter(
            primary_provider=settings.llm_provider,
            fallback_providers=fallback,
            providers=providers,
            timeout_s=settings.llm_timeout_s,
            retry_policy=RetryPolicy(
                max_attempts=settings.llm_retries,
                base_delay_s=settings.llm_base_delay_s,
                max_delay_s=settings.llm_max_delay_s,
            ),
        )

    async def generate(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> LLMCompletion:
        # Production note: keep prompt construction here so all agents remain transport-agnostic.
        model_to_use = model or self._default_model_name
        if context:
            full_prompt = f"{prompt}\n\nContext:\n{context}"
        else:
            full_prompt = prompt

        return await self._router.complete(
            prompt=full_prompt,
            model=model_to_use,
            generation_kwargs=generation_kwargs,
            provider=provider,
            claim_id=claim_id,
        )

