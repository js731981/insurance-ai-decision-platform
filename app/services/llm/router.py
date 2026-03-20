from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.services.llm.providers.base import LLMProvider, LLMProviderError
from app.services.llm.telemetry import estimate_cost_usd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 5.0
    jitter_s: float = 0.1
    # HTTP status codes that are usually transient.
    retry_status_codes: Sequence[int] = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    provider: str
    model: str
    tokens: int
    cost: float
    latency_ms: int
    confidence: float


class LLMRouter:
    """Routes LLM completions across providers with retries + fallback.

    Strategy pattern:
    - Each provider implements the same `LLMProvider` interface.
    - The router decides which provider to execute and how to recover from failures.
    """

    def __init__(
        self,
        *,
        primary_provider: str,
        fallback_providers: Optional[Sequence[str]] = None,
        providers: Mapping[str, LLMProvider],
        timeout_s: float,
        retry_policy: Optional[RetryPolicy] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._primary_provider = primary_provider
        self._fallback_providers = list(fallback_providers or [])
        self._providers = dict(providers)
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy or RetryPolicy()
        self._logger = logger_ or logger

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        provider: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> LLMCompletion:
        provider_chain: List[str] = []

        if provider:
            provider_chain.append(provider)
            provider_chain.extend(self._fallback_providers)
        else:
            provider_chain.append(self._primary_provider)
            provider_chain.extend(self._fallback_providers)

        # Preserve order but remove duplicates.
        deduped_chain: List[str] = []
        seen = set()
        for p in provider_chain:
            if p not in seen:
                deduped_chain.append(p)
                seen.add(p)

        last_error: Optional[BaseException] = None

        for provider_name in deduped_chain:
            provider_obj = self._providers.get(provider_name)
            if provider_obj is None:
                self._logger.warning("LLM provider not configured: %s", provider_name)
                continue

            self._logger.info("LLM routing to provider=%s model=%s", provider_name, model)
            try:
                return await self._complete_with_retries(
                    provider_name=provider_name,
                    provider=provider_obj,
                    prompt=prompt,
                    model=model,
                    generation_kwargs=generation_kwargs,
                )
            except BaseException as exc:  # noqa: BLE001
                last_error = exc
                self._logger.error(
                    "LLM provider failed provider=%s err=%s",
                    provider_name,
                    exc,
                )

        # If we got here, all configured providers failed or were missing.
        if last_error:
            raise last_error
        raise RuntimeError("No configured LLM providers available")

    async def _complete_with_retries(
        self,
        *,
        provider_name: str,
        provider: LLMProvider,
        prompt: str,
        model: str,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> LLMCompletion:
        last_retryable_error: Optional[BaseException] = None

        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                start_s = time.perf_counter()
                text = await asyncio.wait_for(
                    provider.complete(
                        prompt=prompt,
                        model=model,
                        generation_kwargs=generation_kwargs,
                    ),
                    timeout=self._timeout_s,
                )
                latency_ms = int((time.perf_counter() - start_s) * 1000)

                estimated_tokens, estimated_cost = estimate_cost_usd(
                    prompt=prompt,
                    completion=text,
                    provider=provider_name,
                    model=model,
                )
                return LLMCompletion(
                    text=text,
                    provider=provider_name,
                    model=model,
                    tokens=estimated_tokens,
                    cost=estimated_cost,
                    latency_ms=latency_ms,
                    confidence=0.9,  # placeholder until we have a real confidence heuristic
                )
            except asyncio.TimeoutError as exc:
                last_retryable_error = exc
                if attempt == self._retry_policy.max_attempts:
                    raise

                delay = self._compute_delay_s(attempt)
                self._logger.warning(
                    "LLM timeout provider=%s attempt=%s/%s delay_s=%.2f timeout_s=%.2f",
                    provider_name,
                    attempt,
                    self._retry_policy.max_attempts,
                    delay,
                    self._timeout_s,
                )
                await asyncio.sleep(delay)
            except LLMProviderError as exc:
                last_retryable_error = exc
                if not self._should_retry(exc):
                    raise

                if attempt == self._retry_policy.max_attempts:
                    raise

                delay = self._compute_delay_s(attempt)
                self._logger.warning(
                    "Retrying LLM provider=%s attempt=%s/%s delay_s=%.2f status=%s",
                    provider_name,
                    attempt,
                    self._retry_policy.max_attempts,
                    delay,
                    exc.status_code,
                )
                await asyncio.sleep(delay)
            except BaseException as exc:  # noqa: BLE001
                last_retryable_error = exc
                # Unknown errors: treat as retryable, but cap attempts.
                if attempt == self._retry_policy.max_attempts:
                    raise

                delay = self._compute_delay_s(attempt)
                self._logger.warning(
                    "Retrying LLM provider=%s attempt=%s/%s delay_s=%.2f err=%s",
                    provider_name,
                    attempt,
                    self._retry_policy.max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        # Should not happen (loop either returns or raises).
        if last_retryable_error:
            raise last_retryable_error
        raise RuntimeError("LLM retries exhausted")

    def _should_retry(self, exc: LLMProviderError) -> bool:
        if exc.status_code is None:
            # For provider-level errors without a status code, retry as a safe default.
            return True
        return exc.status_code in set(self._retry_policy.retry_status_codes)

    def _compute_delay_s(self, attempt: int) -> float:
        # attempt=1 => ~base_delay_s
        exp = min(attempt - 1, 6)
        delay = min(self._retry_policy.base_delay_s * (2**exp), self._retry_policy.max_delay_s)
        delay += random.uniform(0, self._retry_policy.jitter_s)
        return delay

