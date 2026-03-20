from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.llm.providers.base import LLMProvider, LLMProviderError


class OllamaProvider(LLMProvider):
    """Ollama provider using the `/api/generate` endpoint (non-streaming)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._provider_name = "ollama"

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
            # Ollama supports many options; pass-through keeps this extensible.
            payload.update(generation_kwargs)

        url = f"{self._base_url}/api/generate"
        headers = {"Content-Type": "application/json"}

        data = await asyncio.to_thread(
            _post_json,
            url,
            headers,
            payload,
            self._provider_name,
            self._timeout_s,
        )

        # Ollama returns {"response": "...", "done": true, ...}
        response = data.get("response")
        if not isinstance(response, str):
            raise LLMProviderError(
                provider=self._provider_name,
                message="Unexpected Ollama response format",
                response_body=json.dumps(data)[:2000],
            )
        return response


def _post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    provider_name: str,
    timeout_s: float,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # nosec B310 - controlled by provider usage
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8")
        except Exception:
            raw = str(e)
        raise LLMProviderError(
            provider=provider_name,
            message=f"Ollama HTTP error ({e.code})",
            status_code=e.code,
            response_body=raw[:2000],
        ) from e
    except URLError as e:
        raise LLMProviderError(
            provider=provider_name,
            message=f"Ollama network error: {e.reason}",
        ) from e

