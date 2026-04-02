from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.hitl_service import HitlService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore


def get_llm_service() -> LLMService:
    # Factory so each request gets its own service instance (swap later if you want pooling).
    return LLMService(model_name=settings.model_name)


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def get_vector_store() -> VectorStore:
    # Embedded persistent local store; safe to construct per-request for MVP.
    return VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )


def get_hitl_service() -> HitlService:
    return HitlService(confidence_threshold=0.75)

