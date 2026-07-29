"""
schemas/prediction_schema.py
=============================
Pydantic schemas for prediction request / response payloads.

All schemas are versioned (V1) to support future iterations without
breaking the public API surface.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Language(str, Enum):
    """Supported languages for detection."""

    ENGLISH = "en"
    KANNADA = "kn"
    ROMAN_KANNADA = "roman_kn"
    HINDI = "hi"
    AUTO = "auto"  # Model auto-detects the language


class HateSpeechLabel(str, Enum):
    """
    Classification outcomes — matches the model's trained id2label mapping.

        0 → non_hate   (benign text)
        1 → hate        (hate speech detected)
    """

    NON_HATE = "non_hate"
    HATE = "hate"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Schema for a single-text hate speech detection request.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5_000,
        description="The text to analyse for hate speech.",
        examples=["This is a sample sentence to classify."],
    )
    language: Language = Field(
        default=Language.AUTO,
        description=(
            "Language code for the input text. "
            "Use 'auto' to let the model detect the language automatically."
        ),
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Custom confidence threshold for the hate/offensive class. "
            "Overrides the server default when provided."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only.")
        return v


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch hate speech detection (multiple texts at once).
    """

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to classify (1–32 items).",
    )
    language: Language = Field(
        default=Language.AUTO,
        description="Language code applied to all texts in the batch.",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom confidence threshold (applies to every item).",
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("texts list must contain at least one item.")
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"texts[{i}] must not be blank.")
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """
    Per-text classification result.
    """

    label: HateSpeechLabel = Field(
        ...,
        description="Predicted class label.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted label (0–1).",
    )
    scores: Dict[str, float] = Field(
        ...,
        description="Full probability distribution across all labels (label → probability).",
    )
    language_detected: Optional[str] = Field(
        default=None,
        description="ISO language code detected by the model (if auto-detect enabled).",
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to process this text in milliseconds.",
    )


class PredictionResponse(BaseModel):
    """
    API response for a single prediction request.
    """

    request_id: str = Field(..., description="Unique identifier for this request.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the response.")
    input_text: str = Field(..., description="Sanitised version of the submitted text.")
    result: PredictionResult = Field(..., description="Classification result.")


class BatchPredictionResponse(BaseModel):
    """
    API response for a batch prediction request.
    """

    request_id: str = Field(..., description="Unique identifier for this batch request.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the response.")
    total_items: int = Field(..., description="Number of texts processed.")
    results: List[PredictionResult] = Field(..., description="Results for each input text.")
    total_processing_time_ms: float = Field(
        ...,
        description="Total wall-clock time for the entire batch in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Health schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.
    """

    status: str = Field(..., description="'healthy' or 'degraded'.")
    version: str = Field(..., description="API version string.")
    environment: str = Field(..., description="Deployment environment.")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded in memory.")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the health check.")
