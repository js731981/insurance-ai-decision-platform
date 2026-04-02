from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Ollama embeddings client (local-only, reusable)."""

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

    async def embed(self, text: str) -> list[float]:
        text_n = (text or "").strip()
        if not text_n:
            return []

        url = f"{self._base_url}/api/embeddings"
        payload = {"model": self._model, "prompt": text_n}

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data: Any = resp.json()
        except Exception:
            logger.exception("embedding_request_failed", extra={"url": url, "model": self._model})
            raise

        embedding = _coerce_embedding(data)
        if not embedding:
            raise ValueError("Ollama embeddings response did not contain an embedding vector.")
        return embedding


def _coerce_embedding(data: Any) -> list[float]:
    if isinstance(data, dict):
        # Common Ollama response shapes:
        # - {"embedding": [...]}
        # - {"embeddings": [[...]]}
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

