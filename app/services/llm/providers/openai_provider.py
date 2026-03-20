from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.llm.providers.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI chat completions provider (non-streaming)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._provider_name = "openai"

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self._api_key:
            raise LLMProviderError(provider=self._provider_name, message="Missing OPENAI_API_KEY")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if generation_kwargs:
            payload.update(generation_kwargs)

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        data = await asyncio.to_thread(
            _post_json,
            url,
            headers,
            payload,
            self._timeout_s,
        )

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                provider=self._provider_name,
                message="Unexpected OpenAI response format",
                response_body=json.dumps(data)[:2000],
            ) from exc


def _post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
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
            provider="openai",
            message=f"OpenAI HTTP error ({e.code})",
            status_code=e.code,
            response_body=raw[:2000],
        ) from e
    except URLError as e:
        raise LLMProviderError(
            provider="openai",
            message=f"OpenAI network error: {e.reason}",
        ) from e

