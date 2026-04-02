from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import Depends

from app.agents.decision_agent import DecisionAgent
from app.agents.fraud_agent import FraudAgent
from app.agents.policy_agent import PolicyAgent
from app.core.dependencies import get_embedding_service, get_hitl_service, get_llm_service, get_vector_store
from app.models.schemas import InferenceRequest, InferenceResponse
from app.services.embedding_service import EmbeddingService
from app.services.hitl_service import HitlService
from app.services.llm_service import LLMService
from app.services.metrics import metrics
from app.services.vector_store import (
    SimilarHit,
    VectorStore,
    compute_calibrated_confidence,
    format_similar_hits_for_context,
    majority_review_from_similar_hits,
)

logger = logging.getLogger(__name__)


def _explanation_storage_value(explanation: Any) -> str:
    if isinstance(explanation, dict):
        return json.dumps(explanation, ensure_ascii=False)
    text = str(explanation or "").strip()
    if not text:
        return json.dumps(
            {"summary": "No explanation.", "key_factors": ["Missing explanation."], "similar_case_reference": ""},
            ensure_ascii=False,
        )
    return text


class InsurFlowOrchestrator:
    """Micro-insurance claim pipeline: parallel fraud + policy checks, then a decision."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        hitl_service: HitlService,
    ) -> None:
        self._llm_service = llm_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._hitl_service = hitl_service
        self._fraud_agent = FraudAgent(
            llm_service=llm_service,
        )
        self._policy_agent = PolicyAgent()
        self._decision_agent = DecisionAgent()

    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        provider_override: str | None = None
        if request.task:
            task_l = request.task.strip().lower()
            if task_l == "cheap":
                provider_override = "ollama"
            elif task_l == "complex":
                provider_override = "openai"

        completion = await self._llm_service.generate(
            prompt=request.prompt,
            context=request.context,
            model=request.model,
            provider=provider_override,
        )
        return InferenceResponse(
            text=completion.text,
            provider=completion.provider,
            model=completion.model,
            tokens=completion.tokens,
            cost=completion.cost,
            latency=completion.latency_ms,
            confidence=completion.confidence,
        )

    async def process_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Retrieval → fraud + policy (parallel) → decision → optional memory upsert."""
        workflow_start = time.perf_counter()
        logger.info(
            "orchestrator_claim_start",
            extra={"claim_keys": list(claim.keys())},
        )

        claim_id = str(claim.get("claim_id") or "").strip() or "unknown"
        claim_description = str(claim.get("description") or "").strip()
        if not claim_description:
            claim_description = json.dumps(claim, ensure_ascii=False, default=str)

        similar_context = ""
        similar_hits: list[SimilarHit] = []
        embedding_for_store: list[float] | None = None
        embedding_status = "fail"

        try:
            embedding_for_store = await self._embedding_service.embed(claim_description)
            if not embedding_for_store:
                embedding_status = "fail"
            else:
                embedding_status = "success"
                similar_hits = self._vector_store.query_similar_hits(
                    query_embedding=embedding_for_store,
                    exclude_claim_id=claim_id,
                    n_results=10,
                )
                similar_context = format_similar_hits_for_context(similar_hits)
                if similar_context:
                    logger.info("similar_claims_context_ready", extra={"claim_id": claim_id})
        except Exception as exc:
            embedding_status = "fail"
            logger.exception(
                "retrieval_context_failed",
                extra={"claim_id": claim_id, "error": f"{type(exc).__name__}: {exc}"},
            )

        retrieval_count = len(similar_hits)

        fraud_input = dict(claim)
        fraud_input["similar_claims_context"] = similar_context

        fraud_task = self._fraud_agent.run(fraud_input)
        policy_task = self._policy_agent.run(claim)
        fraud_out, policy_out = await asyncio.gather(fraud_task, policy_task)

        fraud_llm_failed = bool(fraud_out.get("_llm_failed"))

        if fraud_llm_failed:
            policy_valid = bool((policy_out or {}).get("policy_valid"))
            if not policy_valid:
                decision_out: dict[str, Any] = {
                    "decision": "REJECTED",
                    "confidence_score": 0.9,
                    "explanation": "Policy check failed; fraud model unavailable.",
                }
            else:
                decision_out = {
                    "decision": "INVESTIGATE",
                    "confidence_score": 0.5,
                    "explanation": "Fraud model unavailable or returned invalid output; escalate.",
                }
        else:
            decision_in: dict[str, Any] = {
                "fraud": fraud_out,
                "policy": policy_out,
                "similar_majority_review": majority_review_from_similar_hits(similar_hits),
            }
            decision_out = await self._decision_agent.run(decision_in)

        base_confidence = float(decision_out.get("confidence_score") or 0.0)
        calibrated = compute_calibrated_confidence(
            confidence=base_confidence,
            model_decision=str(decision_out.get("decision") or ""),
            similar_hits=similar_hits,
        )

        hitl = self._hitl_service.evaluate(
            decision=str(decision_out.get("decision") or ""),
            confidence=float(calibrated),
        )

        fraud_for_client = {k: v for k, v in fraud_out.items() if k != "_llm_failed"}

        # Single memory write: embedding + document + metadata (skip if embedding unusable).
        if embedding_status == "success" and embedding_for_store:
            try:
                embedding = embedding_for_store
                expl_str = _explanation_storage_value(fraud_out.get("explanation"))
                self._vector_store.store_claim(
                    claim_id=claim_id,
                    claim_description=claim_description,
                    embedding=embedding,
                    metadata={
                        "claim_id": claim_id,
                        "fraud_score": float(fraud_out.get("fraud_score") or 0.0),
                        "decision": str(decision_out.get("decision") or ""),
                        "confidence": float(decision_out.get("confidence_score") or 0.0),
                        "entities": fraud_out.get("entities") or {},
                        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                        "explanation": expl_str,
                        "review_status": "",
                        "hitl_needed": hitl.needs_hitl,
                        "hitl_reason": hitl.reason or "",
                    },
                )
                logger.info("claim_stored", extra={"claim_id": claim_id})
            except Exception:
                logger.exception("orchestrator_memory_store_failed", extra={"claim_id": claim_id})
        elif embedding_status != "success":
            logger.warning(
                "claim_store_skipped",
                extra={"claim_id": claim_id, "embedding_status": embedding_status},
            )

        metrics.record_claim_processed(hitl_triggered=hitl.needs_hitl)

        elapsed_ms = (time.perf_counter() - workflow_start) * 1000
        logger.info(
            "claim_triage_structured",
            extra={
                "claim_id": claim_id,
                "decision": decision_out.get("decision"),
                "confidence": base_confidence,
                "calibrated_confidence": calibrated,
                "hitl_needed": hitl.needs_hitl,
                "embedding_status": embedding_status,
                "retrieval_count": retrieval_count,
            },
        )
        logger.info(
            "orchestrator_claim_complete",
            extra={
                "duration_ms": round(elapsed_ms, 2),
                "decision": decision_out.get("decision"),
            },
        )

        return {
            "claim_id": claim_id,
            "decision": decision_out["decision"],
            "confidence_score": decision_out["confidence_score"],
            "calibrated_confidence": calibrated,
            "hitl_needed": hitl.needs_hitl,
            "agent_outputs": {
                "fraud": fraud_for_client,
                "policy": policy_out,
                "decision": decision_out,
            },
        }


def get_insurflow_orchestrator(
    llm_service: LLMService = Depends(get_llm_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    hitl_service: HitlService = Depends(get_hitl_service),
) -> InsurFlowOrchestrator:
    return InsurFlowOrchestrator(
        llm_service=llm_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
        hitl_service=hitl_service,
    )
