import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables from a .env file if present.
load_dotenv()


def _default_llm_timeout_seconds() -> float:
    """Wall-clock cap per LLM attempt; local Ollama often needs much more than cloud APIs."""
    prov = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    raw = os.getenv("LLM_TIMEOUT_S") or os.getenv("LLM_TIMEOUT")
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    return 120.0 if prov == "ollama" else 60.0


class Settings(BaseModel):
    """Minimal runtime configuration.

    Keep this lightweight until you add more config sources.
    """

    app_name: str = os.getenv("APP_NAME", "Insurance AI Decision Platform")
    debug: bool = os.getenv("DEBUG", "false").strip().lower() == "true"
    # Confirmed local LLM default for MVP.
    model_name: str = os.getenv("MODEL_NAME", "phi3")

    # LLM routing/execution configuration
    # Confirmed local-only default for MVP (Ollama).
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    llm_fallback_providers: str = os.getenv("LLM_FALLBACK_PROVIDERS", "").strip().lower()
    # Total attempts per provider (initial + retries). "Retry up to 2 times" => 3 attempts.
    llm_retries: int = int(os.getenv("LLM_RETRIES", "3"))
    llm_base_delay_s: float = float(os.getenv("LLM_BASE_DELAY_S", "0.5"))
    llm_max_delay_s: float = float(os.getenv("LLM_MAX_DELAY_S", "5.0"))
    # Per-attempt wall-clock cap (asyncio.wait_for + provider HTTP client). Alias: LLM_TIMEOUT.
    llm_timeout_s: float = Field(default_factory=_default_llm_timeout_seconds)

    # Estimated cost tracking (heuristic; providers do not currently return usage).
    # Provide USD rates per 1k tokens. If zero, cost will generally compute as 0.
    llm_cost_usd_per_1k_input_tokens: float = float(os.getenv("LLM_COST_USD_PER_1K_INPUT_TOKENS", "0"))
    llm_cost_usd_per_1k_output_tokens: float = float(os.getenv("LLM_COST_USD_PER_1K_OUTPUT_TOKENS", "0"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Embeddings + local vector store (fully embedded / local persistence)
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "claims")

    # RAG (retrieval-augmented fraud context). TOP_K = RAG_TOP_K (similar claims count).
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").strip().lower() == "true"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    rag_rerank_enabled: bool = os.getenv("RAG_RERANK_ENABLED", "false").strip().lower() == "true"
    rag_context_max_tokens: int = int(os.getenv("RAG_CONTEXT_MAX_TOKENS", "256"))

    enable_parallel_execution: bool = (
        os.getenv("ENABLE_PARALLEL_EXECUTION", "true").strip().lower() == "true"
    )

    embedding_timeout_s: float = float(os.getenv("EMBEDDING_TIMEOUT_S", "30"))

    # Lightweight DL fraud head (optional torch; logistic fallback without it)
    dl_fraud_enabled: bool = os.getenv("DL_FRAUD_ENABLED", "false").strip().lower() == "true"
    dl_fraud_fusion_llm_weight: float = float(os.getenv("DL_FRAUD_FUSION_LLM_WEIGHT", "0.7"))
    dl_fraud_fusion_dl_weight: float = float(os.getenv("DL_FRAUD_FUSION_DL_WEIGHT", "0.3"))

    # FraudAgent: JSON parse recovery (extra LLM calls after invalid structured output).
    max_llm_retries: int = int(os.getenv("MAX_LLM_RETRIES", "1"))
    strict_json_mode: bool = os.getenv("STRICT_JSON_MODE", "true").strip().lower() == "true"


settings = Settings()

