from typing import Optional

from pydantic import BaseModel, Field


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
    fraud_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=2000)

