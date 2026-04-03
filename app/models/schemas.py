from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    context: Optional[str] = Field(default=None, max_length=20000)
    model: str = Field(default="default", max_length=100)
    task: Optional[str] = Field(default=None, max_length=50)


class InferenceResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens: int
    cost: float
    latency: int
    confidence: float = Field(default=0.9)


class FraudAnalysisResponse(BaseModel):
    """Legacy fraud-only shape (optional use by clients)."""

    fraud_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=2000)


class ClaimRequest(BaseModel):
    """Inbound micro-insurance claim.

    Required fields are the three used for routing and policy checks; optional fields are
    forwarded to the fraud agent as additional context (no unknown top-level keys).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "claim_id": "MIC-2026-00412",
                "claim_amount": 75.5,
                "policy_limit": 500.0,
                "currency": "USD",
                "product_code": "PHONE_CRACK",
                "incident_date": "2026-03-28",
                "policyholder_id": "ph_8k2m",
                "description": "Dropped phone; screen cracked; repair quote from authorized shop.",
            }
        },
    )

    claim_id: str = Field(..., min_length=1, max_length=200, description="Stable id for this micro-claim.")
    claim_amount: float = Field(
        ...,
        ge=0,
        description="Requested payout amount (same currency unit as policy_limit).",
    )
    policy_limit: float = Field(
        ...,
        ge=0,
        description="Maximum benefit / cap for this micro-policy at triage time.",
    )
    currency: Optional[str] = Field(
        default=None,
        max_length=10,
        description="ISO 4217 code (e.g. USD); same unit as amounts.",
    )
    product_code: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Micro-product identifier (e.g. crop, phone, health bundle).",
    )
    incident_date: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Incident date (ISO 8601 date recommended).",
    )
    policyholder_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Internal id for the insured party.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Free-text narrative for fraud / context (optional).",
    )


ClaimDecision = Literal["APPROVED", "REJECTED", "INVESTIGATE"]


class ClaimProcessResponse(BaseModel):
    """Outcome of automated micro-insurance triage (approve, reject, or escalate)."""

    claim_id: str
    decision: ClaimDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    hitl_needed: bool = Field(default=False)
    agent_outputs: Dict[str, Any]


ReviewAction = Literal["APPROVED", "REJECTED"]


class ClaimReviewRequest(BaseModel):
    action: ReviewAction
    reviewed_by: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional reviewer id or label for the learning loop.",
    )


CaseStatus = Literal["NEW", "ASSIGNED", "IN_PROGRESS", "RESOLVED"]


class CaseAssignRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"assigned_to": "investigator_1"}})

    assigned_to: str = Field(..., min_length=1, max_length=200)


class CaseStatusUpdateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"case_status": "IN_PROGRESS"}})

    case_status: CaseStatus


class CaseListItem(BaseModel):
    claim_id: str
    case_status: str
    assigned_to: str = ""
    decision: Optional[ClaimDecision] = None
    fraud_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = ""
    review_status: str = ""
    timestamp: str = ""


class CaseListResponse(BaseModel):
    cases: list[CaseListItem]


class ClaimListItem(BaseModel):
    claim_id: str
    claim_description: str = Field(default="")
    fraud_score: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: Optional[ClaimDecision] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    explanation: Optional[str] = Field(default=None, description="Fraud analyst explanation (bullets or text).")
    review_status: Optional[ReviewAction] = Field(
        default=None,
        description="Human review outcome when present (feeds learning loop).",
    )
    hitl_needed: bool = Field(default=False)
    reviewed_action: Optional[ReviewAction] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    entities: Optional[Dict[str, Any]] = Field(default=None, description="Structured hints from fraud agent.")
