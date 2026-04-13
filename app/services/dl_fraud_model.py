from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

INPUT_DIM = 16

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _stable_hash01(key: str) -> float:
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 10001) / 10000.0


def build_fraud_features(
    *,
    claim_amount: float,
    structured: Mapping[str, Any] | None,
    embedding: Sequence[float] | None,
) -> list[float]:
    """Hand-crafted lightweight features (no external tables)."""
    s = structured or {}
    try:
        amt = float(claim_amount)
    except (TypeError, ValueError):
        amt = 0.0
    try:
        lim = float(s.get("policy_limit") or 0.0)
    except (TypeError, ValueError):
        lim = 0.0
    ratio = amt / lim if lim > 0 else 0.0
    ratio = min(ratio, 3.0) / 3.0

    product = str(s.get("product_code") or "")
    currency = str(s.get("currency") or "")
    desc_len = len(str(s.get("description") or ""))
    desc_norm = min(1.0, desc_len / 5000.0)

    feats: list[float] = [
        _clamp01(math.log1p(max(amt, 0.0)) / math.log1p(50_000.0)),
        _clamp01(ratio),
        _stable_hash01(product) if product else 0.0,
        _stable_hash01(currency) if currency else 0.25,
        desc_norm,
        _clamp01((amt % 97) / 97.0),
        1.0 if str(s.get("incident_date") or "") else 0.0,
        1.0 if str(s.get("policyholder_id") or "") else 0.0,
    ]

    emb_slice = [0.0] * 8
    if embedding:
        n = len(embedding)
        if n > 0:
            for i in range(8):
                idx = (i * n) // 8
                try:
                    emb_slice[i] = float(embedding[idx])
                except (TypeError, ValueError, IndexError):
                    emb_slice[i] = 0.0
    feats.extend(emb_slice)

    while len(feats) < INPUT_DIM:
        feats.append(0.0)
    return feats[:INPUT_DIM]


class _TorchFraudNet:
    def __init__(self) -> None:
        assert torch is not None and nn is not None
        torch.manual_seed(42)
        self._net = nn.Sequential(
            nn.Linear(INPUT_DIM, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self._net.eval()

    def forward(self, features: list[float]) -> float:
        assert torch is not None
        with torch.inference_mode():
            x = torch.tensor([features], dtype=torch.float32)
            y = self._net(x).item()
            return float(_clamp01(y))


class _LogisticFallback:
    """Deterministic logistic-style scorer (no numpy/torch)."""

    def __init__(self) -> None:
        rng = random.Random(42)
        self._weights = [rng.uniform(-0.35, 0.35) for _ in range(INPUT_DIM)]
        self._bias = rng.uniform(-0.05, 0.05)

    def forward(self, features: list[float]) -> float:
        z = self._bias + sum(w * f for w, f in zip(self._weights, features))
        z = max(-30.0, min(30.0, z))
        return float(_clamp01(1.0 / (1.0 + math.exp(-z))))


class DeepFraudModel:
    """Lightweight fraud probability head: small PyTorch MLP when available, else logistic fallback."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._torch_net: _TorchFraudNet | None = None
        self._fallback: _LogisticFallback | None = None
        self._backend = "disabled"
        if not enabled:
            return
        if _TORCH_AVAILABLE:
            self._torch_net = _TorchFraudNet()
            self._backend = "torch"
        else:
            self._fallback = _LogisticFallback()
            self._backend = "logistic"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        return self._backend

    def predict(
        self,
        *,
        claim_amount: float,
        structured: Mapping[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> float | None:
        if not self._enabled:
            return None
        features = build_fraud_features(
            claim_amount=claim_amount,
            structured=structured,
            embedding=embedding,
        )
        if self._torch_net is not None:
            prob = self._torch_net.forward(features)
        elif self._fallback is not None:
            prob = self._fallback.forward(features)
        else:
            return None
        logger.debug(
            "dl_fraud_predict",
            extra={"backend": self._backend, "fraud_probability": prob},
        )
        return prob

    async def predict_async(
        self,
        *,
        claim_amount: float,
        structured: Mapping[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> float | None:
        if not self._enabled:
            return None
        return await asyncio.to_thread(
            self.predict,
            claim_amount=claim_amount,
            structured=structured,
            embedding=embedding,
        )


__all__ = ["DeepFraudModel", "build_fraud_features"]
