from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Ollama embeddings client with a reusable async HTTP session."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("embedding_client_close_failed")

    async def embed(self, text: str) -> list[float]:
        text_n = (text or "").strip()
        if not text_n:
            return []

        payload = {"model": self._model, "prompt": text_n}

        try:
            resp = await self._client.post("/api/embeddings", json=payload)
            resp.raise_for_status()
            data: Any = resp.json()
        except Exception:
            logger.exception(
                "embedding_request_failed",
                extra={"base_url": self._base_url, "model": self._model},
            )
            raise

        embedding = _coerce_embedding(data)
        if not embedding:
            raise ValueError("Ollama embeddings response did not contain an embedding vector.")
        return embedding


def _coerce_embedding(data: Any) -> list[float]:
    if isinstance(data, dict):
        emb = data.get("embedding")
        if isinstance(emb, Sequence) and not isinstance(emb, (str, bytes)):
            return [float(x) for x in emb]
        embs = data.get("embeddings")
        if (
            isinstance(embs, Sequence)
            and len(embs) >= 1
            and isinstance(embs[0], Sequence)
            and not isinstance(embs[0], (str, bytes))
        ):
            return [float(x) for x in embs[0]]
    return []


__all__ = ["EmbeddingService"]
