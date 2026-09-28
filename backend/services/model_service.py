"""
services/model_service.py
==========================
Model service layer for Context-Aware Multilingual Hate Speech Detection.

Supports:
- Hugging Face Transformers (AutoModelForSequenceClassification)
- Dynamic compute device selection (CUDA GPU, Apple Silicon MPS, or CPU)
- Single and batched non-blocking asynchronous inference (via asyncio.to_thread)
- Configurable decision thresholds and standardized response formatting
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import get_settings

# ---------------------------------------------------------------------------
# Module-level logger and settings
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ModelNotLoadedError(Exception):
    pass

class ModelLoadError(Exception):
    pass

# ---------------------------------------------------------------------------
# ModelService
# ---------------------------------------------------------------------------

class ModelService:
    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._is_loaded: bool = False
        self._model_path: Path = settings.model_path
        self._default_threshold: float = settings.prediction_threshold

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_ensemble(self) -> bool:
        return False

    @property
    def has_shap(self) -> bool:
        return False

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise ModelNotLoadedError("Tokenizer is not available.")
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise ModelNotLoadedError("Device is not set. Call load() first.")
        return self._device

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelNotLoadedError("Model is not available. Call load() first.")
        return self._model

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def _validate_model_directory(self) -> None:
        if not self._model_path.exists():
            raise ModelLoadError(f"Model directory not found: '{self._model_path}'")

    async def load(self) -> None:
        logger.info("=" * 60)
        logger.info("Initialising model loading sequence...")
        logger.info("Model directory : %s", self._model_path)

        self._validate_model_directory()
        self._device = self._resolve_device()
        logger.info("Compute device  : %s", self._device)

        # Tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), local_files_only=True, use_fast=True
            )
        except Exception as exc:
            raise ModelLoadError(f"Tokenizer loading failed: {exc}") from exc

        # Transformer Model
        logger.info("Loading Sequence Classification Model...")
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self._model_path), local_files_only=True
            )
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            raise ModelLoadError(f"Transformer loading failed: {exc}") from exc

        self._is_loaded = True
        logger.info("Model stack ready.")

    async def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._device = None
        self._is_loaded = False

    async def predict(self, text: str, threshold: Optional[float] = None) -> Dict[str, Any]:
        if not self._is_loaded:
            raise ModelNotLoadedError()
        results = await self._run_inference([text], threshold=threshold)
        return results[0]

    async def predict_batch(self, texts: List[str], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self._is_loaded:
            raise ModelNotLoadedError()
        return await self._run_inference(texts, threshold=threshold)

    def _infer(self, texts: List[str], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        thresh = threshold if threshold is not None else self._default_threshold

        encoded = self._tokenizer(
            texts, padding=True, truncation=True, 
            max_length=settings.max_sequence_length, return_tensors="pt"
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)

        per_item_ms = ((time.perf_counter() - t0) * 1000) / len(texts)
        results = []
        
        id2label = getattr(self._model.config, "id2label", {0: "non_hate", 1: "hate"})

        for i, probs in enumerate(probabilities):
            probs_list = probs.cpu().tolist()
            
            # Assuming id 1 is 'HATE' based on id2label typical layout
            hate_idx = 1
            non_hate_idx = 0
            if "HATE" in id2label.values():
                for k, v in id2label.items():
                    if v == "HATE": hate_idx = k
                    if v == "NON-HATE": non_hate_idx = k

            hate_prob = float(probs_list[hate_idx])
            non_hate_prob = float(probs_list[non_hate_idx])
            
            is_hate = hate_prob >= thresh
            pred_label = "hate" if is_hate else "non_hate"
            confidence = hate_prob if is_hate else non_hate_prob

            results.append({
                "label": pred_label,
                "confidence": round(confidence, 6),
                "scores": {
                    "non_hate": round(non_hate_prob, 6),
                    "hate": round(hate_prob, 6),
                },
                "explanation": None,
                "language_detected": None,
                "processing_time_ms": round(per_item_ms, 3),
            })
        return results

    async def _run_inference(self, texts: List[str], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._infer, texts, threshold)

model_service = ModelService()
