from __future__ import annotations

import threading
from typing import Any


class InMemoryMetrics:
    """Lightweight process-local counters (no external infrastructure).

    These reset when the API process restarts. They are *not* derived from Chroma.

    - total_claims_processed: each successful ``InsurFlowOrchestrator.process_claim`` run
      (``POST /claims`` or ``POST /claim``), this process only.
    - reviewed_claims_count: each successful ``POST /claims/{id}/review``, this process only.
      Can exceed triage count if you only review data that existed before this process started.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_claims_processed = 0
        self._hitl_triggered_count = 0
        self._reviewed_claims_count = 0

    def record_claim_processed(self, *, hitl_triggered: bool) -> None:
        with self._lock:
            self._total_claims_processed += 1
            if hitl_triggered:
                self._hitl_triggered_count += 1

    def record_review(self) -> None:
        with self._lock:
            self._reviewed_claims_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_claims_processed": self._total_claims_processed,
                "hitl_triggered_count": self._hitl_triggered_count,
                "reviewed_claims_count": self._reviewed_claims_count,
            }


metrics = InMemoryMetrics()
