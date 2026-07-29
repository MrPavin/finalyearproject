"""
routes/prediction.py
=====================
API v1 routes for hate speech prediction.

Endpoints:
    POST /api/v1/predict        — Single text prediction
    POST /api/v1/predict/batch  — Batch text prediction
    GET  /api/v1/model/info     — Model metadata
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from schemas.prediction_schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HateSpeechLabel,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from services.model_service import ModelNotLoadedError, ModelService, model_service
from utils.helper import (
    Timer,
    build_error_response,
    generate_request_id,
    sanitize_text,
    utcnow_iso,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_model_service() -> ModelService:
    """FastAPI dependency that returns the module-level ModelService instance."""
    return model_service


# ---------------------------------------------------------------------------
# Helper: map raw service output → PredictionResult
# ---------------------------------------------------------------------------

def _build_prediction_result(raw: Dict[str, Any]) -> PredictionResult:
    """
    Convert the dictionary returned by ModelService._infer() into a
    typed PredictionResult schema.

    The scores dict uses plain string keys (e.g. "non_hate", "hate")
    because PredictionResult.scores is Dict[str, float].
    """
    return PredictionResult(
        label=HateSpeechLabel(raw["label"]),
        confidence=raw["confidence"],
        scores=raw["scores"],          # already Dict[str, float] from _infer
        language_detected=raw.get("language_detected"),
        processing_time_ms=raw["processing_time_ms"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Single text hate speech detection",
    description=(
        "Submit a single piece of text for hate speech classification. "
        "Returns a label (hate / offensive / normal), confidence score, "
        "and full probability distribution."
    ),
    responses={
        503: {"description": "Model not loaded"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def predict_single(
    payload: PredictionRequest,
    request: Request,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """
    **Single-text prediction endpoint.**

    - Sanitises and validates the input text.
    - Delegates to the `ModelService` for inference.
    - Returns a structured `PredictionResponse`.

    > ⚠️ Returns HTTP 503 when the model is not yet loaded.
    """
    request_id = generate_request_id()
    clean_text = sanitize_text(payload.text)

    logger.info(
        "Prediction request | id=%s lang=%s text_len=%d",
        request_id,
        payload.language.value,
        len(clean_text),
    )

    try:
        with Timer() as t:
            raw_result = await service.predict(
                text=clean_text,
                threshold=payload.threshold,
            )

        logger.info(
            "Prediction complete | id=%s label=%s confidence=%.4f elapsed_ms=%.2f",
            request_id,
            raw_result["label"],
            raw_result["confidence"],
            t.elapsed_ms,
        )

        return PredictionResponse(
            request_id=request_id,
            timestamp=utcnow_iso(),
            input_text=clean_text,
            result=_build_prediction_result(raw_result),
        )

    except ModelNotLoadedError as exc:
        logger.warning("Model not loaded | id=%s error=%s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later.",
        ) from exc

    except NotImplementedError as exc:
        logger.warning("Prediction not implemented | id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Prediction is not yet implemented.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during prediction | id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch hate speech detection",
    description=(
        "Submit up to 32 texts at once for hate speech classification. "
        "All texts share the same language setting and threshold."
    ),
    responses={
        503: {"description": "Model not loaded"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    """
    **Batch prediction endpoint.**

    - Accepts 1–32 texts per request.
    - Processes them in a single batched forward pass.
    - Returns a result for each input text preserving input order.
    """
    request_id = generate_request_id()
    clean_texts = [sanitize_text(t) for t in payload.texts]

    logger.info(
        "Batch prediction request | id=%s lang=%s items=%d",
        request_id,
        payload.language.value,
        len(clean_texts),
    )

    try:
        with Timer() as total_timer:
            raw_results = await service.predict_batch(
                texts=clean_texts,
                threshold=payload.threshold,
            )

        logger.info(
            "Batch prediction complete | id=%s items=%d total_ms=%.2f",
            request_id,
            len(raw_results),
            total_timer.elapsed_ms,
        )

        return BatchPredictionResponse(
            request_id=request_id,
            timestamp=utcnow_iso(),
            total_items=len(raw_results),
            results=[_build_prediction_result(r) for r in raw_results],
            total_processing_time_ms=total_timer.elapsed_ms,
        )

    except ModelNotLoadedError as exc:
        logger.warning("Model not loaded | id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later.",
        ) from exc

    except NotImplementedError as exc:
        logger.warning("Batch prediction not implemented | id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Batch prediction is not yet implemented.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during batch prediction | id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from exc


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@router.get(
    "/model/info",
    tags=["Model"],
    summary="Model metadata",
    description="Returns metadata about the currently configured XLM-RoBERTa model.",
)
async def model_info(
    service: ModelService = Depends(get_model_service),
) -> Dict[str, Any]:
    """Return static metadata about the model configuration."""
    from config import get_settings
    cfg = get_settings()

    return {
        "model_name": cfg.model_name,
        "model_dir": cfg.model_dir,
        "max_sequence_length": cfg.max_sequence_length,
        "prediction_threshold": cfg.prediction_threshold,
        "is_loaded": service.is_loaded,
        "supported_languages": ["en", "kn", "roman_kn", "hi", "auto"],
        "labels": [label.value for label in HateSpeechLabel],
    }
