"""Pydantic v2 request/response schemas for the API.

Request field names come from the config data contract at runtime
(see src/data_ingestion/schemas.py); only response shapes are fixed here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.data_ingestion.schemas import CustomerIngestRequest


class PredictRequest(CustomerIngestRequest):
    pass


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_band: str


class ExplainResponse(BaseModel):
    customer_id: str
    churn_probability: float
    base_value: float
    top_features: list[dict]
    explanation: str


class RecommendResponse(BaseModel):
    customer_id: str
    churn_probability: float
    top_causes: list[str]
    recommended_actions: list[dict]
    confidence_score: float
    priority_score: float


class UploadResponse(BaseModel):
    customer_id: str
    product_id: str
    image_path: str
    quality_score: float
    report: dict


class PredictWithImageRequest(PredictRequest):
    """Same customer fields plus an optional base64-encoded product image."""

    image_b64: str | None = Field(default=None, description="Base64 image (data-URI ok)")


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
