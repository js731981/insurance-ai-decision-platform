from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


def _persist_dir() -> str:
    # Keep aligned with repo convention: ./chroma_db (relative to cwd), but normalize to absolute
    # to avoid Chroma singleton collisions between different working directories.
    return str(Path("chroma_db").expanduser().resolve())


def _get_client() -> chromadb.PersistentClient:
    # Disable Chroma telemetry (matches existing repo behavior).
    return chromadb.PersistentClient(path=_persist_dir(), settings=Settings(anonymized_telemetry=False))


def _get_collection() -> Any:
    # IMPORTANT: keep this a separate collection so we do NOT mix embedding dimensions
    # with the platform's primary `claims` collection (which uses Ollama embeddings).
    client = _get_client()
    return client.get_or_create_collection(name="claims_post_decision_sbert")


def _get_embedder():
    # sentence-transformers is optional at runtime; we degrade gracefully.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        logger.warning("sbert_embedder_unavailable", extra={"error": f"{type(exc).__name__}: {exc}"})
        return None


def store_claim(claim: dict[str, Any]) -> None:
    """Store a minimal claim record for post-decision RAG (best-effort)."""
    try:
        claim_id = str(claim.get("claim_id") or "").strip()
        desc = str(claim.get("description") or "").strip()
        if not claim_id or not desc:
            return

        embedder = _get_embedder()
        if embedder is None:
            return

        emb = embedder.encode(desc).tolist()
        collection = _get_collection()
        # Use upsert semantics (Chroma add() fails on duplicate ids).
        collection.upsert(
            ids=[claim_id],
            documents=[desc],
            embeddings=[emb],
            metadatas=[dict(claim)],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rag_store_failed", extra={"error": f"{type(exc).__name__}: {exc}"})


def retrieve_similar(description: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve k similar historical claim metadatas (best-effort)."""
    try:
        desc = str(description or "").strip()
        if not desc:
            return []

        embedder = _get_embedder()
        if embedder is None:
            return []

        emb = embedder.encode(desc).tolist()
        collection = _get_collection()
        res = collection.query(query_embeddings=[emb], n_results=max(1, min(10, int(k))), include=["metadatas"])
        metas = res.get("metadatas") if isinstance(res, dict) else None
        if not metas or not isinstance(metas, list) or not metas[0]:
            return []
        first = metas[0]
        return [m for m in first if isinstance(m, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.exception("rag_retrieve_failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        return []


__all__ = ["store_claim", "retrieve_similar"]

