from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

from app.services.llm.providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama `/api/generate` using a shared async HTTP client (no per-request model reload on client side)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._provider_name = "ollama"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("ollama_client_close_failed")

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if generation_kwargs:
            payload.update(generation_kwargs)

        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:2000]
            except Exception:
                body = str(exc)
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama HTTP error ({exc.response.status_code})",
                status_code=exc.response.status_code,
                response_body=body,
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                provider=self._provider_name,
                message=f"Ollama network error: {exc}",
            ) from exc

        response = data.get("response")
        if not isinstance(response, str):
            raise LLMProviderError(
                provider=self._provider_name,
                message="Unexpected Ollama response format",
                response_body=json.dumps(data)[:2000],
            )
        return response
