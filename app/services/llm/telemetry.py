from __future__ import annotations

from typing import Tuple

from app.core.config import settings


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator.

    Many LLM tokenizers behave roughly like "1 token ~= 4 chars" for English text.
    This is intentionally approximate since providers currently do not return usage.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _rates_usd_per_1k_tokens(provider: str, model: str) -> Tuple[float, float]:
    """Return (input_rate_usd_per_1k, output_rate_usd_per_1k)."""
    provider_l = (provider or "").lower()
    model_l = (model or "").lower()

    # If you set explicit rates, prefer them.
    if settings.llm_cost_usd_per_1k_input_tokens > 0 or settings.llm_cost_usd_per_1k_output_tokens > 0:
        return (settings.llm_cost_usd_per_1k_input_tokens, settings.llm_cost_usd_per_1k_output_tokens)

    # Default heuristics for common OpenAI-ish model names.
    if provider_l in {"openai", "openrouter"}:
        if "gpt-4o-mini" in model_l or "4o-mini" in model_l:
            # $0.15 / 1M input, $0.60 / 1M output
            return (0.15 / 1000, 0.60 / 1000)
        if "gpt-4o" in model_l or "4o-" in model_l:
            # $5 / 1M input, $15 / 1M output
            return (5.00 / 1000, 15.00 / 1000)
        if "gpt-3.5" in model_l or "gpt-3.5-turbo" in model_l:
            # $0.50 / 1M input, $1.50 / 1M output
            return (0.50 / 1000, 1.50 / 1000)

    # For local models (ollama) we don't have a reliable USD pricing model.
    return (0.0, 0.0)


def estimate_cost_usd(*, prompt: str, completion: str, provider: str, model: str) -> Tuple[int, float]:
    """Return (total_tokens_estimate, estimated_cost_usd)."""
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    input_rate, output_rate = _rates_usd_per_1k_tokens(provider=provider, model=model)

    cost = (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate
    total_tokens = prompt_tokens + completion_tokens
    return total_tokens, cost

