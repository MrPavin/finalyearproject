"""
services/model_service.py
==========================
Model service layer for the XLM-RoBERTa hate speech detection model.

Responsibilities:
- Detect and select the compute device (CUDA GPU or CPU).
- Load the tokenizer once at application startup via AutoTokenizer.
- Load the model once at application startup via AutoModelForSequenceClassification.
- Set the model to evaluation mode and move it to the selected device.
- Expose the tokenizer, model, and device as typed properties.
- Provide a clean unload path for graceful shutdown.
- Run single and batch inference with softmax probability decoding.
"""

import asyncio
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
    """
    Raised when a prediction or tokenization call is attempted before
    the model and tokenizer have been successfully loaded.
    """


class ModelLoadError(Exception):
    """
    Raised when the model or tokenizer fails to load from disk.
    Wraps the underlying exception with additional context.
    """


# ---------------------------------------------------------------------------
# ModelService
# ---------------------------------------------------------------------------

class ModelService:
    """
    Singleton service that owns the lifecycle of the XLM-RoBERTa model.

    Designed to be instantiated **once** at module level and shared across
    the entire FastAPI application via dependency injection or direct import.

    Attributes:
        _tokenizer:   Loaded HuggingFace tokenizer (None until load() succeeds).
        _model:       Loaded classification model (None until load() succeeds).
        _device:      torch.device selected at load time (cuda or cpu).
        _is_loaded:   True only after both tokenizer and model are ready.
        _model_path:  Absolute path to the model directory on disk.

    Usage::

        # In FastAPI lifespan:
        await model_service.load()

        # In a route:
        if model_service.is_loaded:
            tok = model_service.tokenizer
            mdl = model_service.model
            dev = model_service.device
    """

    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._is_loaded: bool = False
        self._model_path: Path = settings.model_path

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True when both tokenizer and model are ready for inference."""
        return self._is_loaded

    @property
    def tokenizer(self) -> Any:
        """
        The loaded AutoTokenizer instance.

        Raises:
            ModelNotLoadedError: If accessed before a successful load().
        """
        if self._tokenizer is None:
            raise ModelNotLoadedError(
                "Tokenizer is not available. Call load() first."
            )
        return self._tokenizer

    @property
    def model(self) -> Any:
        """
        The loaded AutoModelForSequenceClassification instance.

        Raises:
            ModelNotLoadedError: If accessed before a successful load().
        """
        if self._model is None:
            raise ModelNotLoadedError(
                "Model is not available. Call load() first."
            )
        return self._model

    @property
    def device(self) -> torch.device:
        """
        The torch.device the model is running on (cuda or cpu).

        Raises:
            ModelNotLoadedError: If accessed before a successful load().
        """
        if self._device is None:
            raise ModelNotLoadedError(
                "Device is not set. Call load() first."
            )
        return self._device

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        """
        Detect the best available compute device.

        Priority order:
            1. CUDA (NVIDIA GPU) — if torch.cuda.is_available()
            2. MPS  (Apple Silicon) — if torch.backends.mps.is_available()
            3. CPU  — fallback

        Returns:
            torch.device instance pointing at the selected device.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(
                "GPU detected — using CUDA | device=%s | VRAM=%.2f GB",
                gpu_name,
                vram_gb,
            )
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon GPU detected — using MPS.")
        else:
            device = torch.device("cpu")
            logger.info(
                "No GPU detected — running on CPU. "
                "Inference will be slower than on GPU."
            )
        return device

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """
        Load the tokenizer and model from *settings.model_path*.

        Steps:
            1. Validate that the model directory exists and contains the
               required files.
            2. Detect the compute device (CUDA / MPS / CPU).
            3. Load the tokenizer with AutoTokenizer.from_pretrained().
            4. Load the model with AutoModelForSequenceClassification.from_pretrained().
            5. Move the model to the selected device.
            6. Set the model to eval mode (disables dropout / batch-norm).
            7. Set self._is_loaded = True on success.

        Raises:
            ModelLoadError: If any step above fails. The server continues
                            to run but the /health endpoint will report
                            model_loaded=False and predictions will return 503.
        """
        logger.info("=" * 60)
        logger.info("Initialising model loading sequence...")
        logger.info("Model directory : %s", self._model_path)

        # ------------------------------------------------------------------
        # Step 1 — Validate model directory
        # ------------------------------------------------------------------
        self._validate_model_directory()

        # ------------------------------------------------------------------
        # Step 2 — Device selection
        # ------------------------------------------------------------------
        logger.info("Resolving compute device...")
        self._device = self._resolve_device()
        logger.info("Compute device  : %s", self._device)

        # ------------------------------------------------------------------
        # Step 3 — Load tokenizer
        # ------------------------------------------------------------------
        logger.info("Loading tokenizer from disk...")
        t0 = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path),
                local_files_only=True,   # Never download; use local weights only
                use_fast=True,           # Use Rust-backed fast tokenizer
            )
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            logger.info(
                "Tokenizer loaded successfully | vocab_size=%d | elapsed=%.0f ms",
                self._tokenizer.vocab_size,
                elapsed_ms,
            )
        except Exception as exc:
            logger.exception(
                "Failed to load tokenizer from '%s'.", self._model_path
            )
            raise ModelLoadError(
                f"Tokenizer loading failed: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 4 — Load model
        # ------------------------------------------------------------------
        logger.info("Loading model weights from disk (this may take a moment)...")
        t0 = time.perf_counter()
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self._model_path),
                local_files_only=True,        # Never download; use local weights only
                ignore_mismatched_sizes=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            num_params = sum(p.numel() for p in self._model.parameters())
            logger.info(
                "Model weights loaded | params=%s | elapsed=%.0f ms",
                f"{num_params:,}",
                elapsed_ms,
            )
        except Exception as exc:
            # Tokenizer loaded fine; reset it so state is consistent
            self._tokenizer = None
            logger.exception(
                "Failed to load model weights from '%s'.", self._model_path
            )
            raise ModelLoadError(
                f"Model weight loading failed: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 5 — Move model to device
        # ------------------------------------------------------------------
        logger.info("Moving model to device: %s ...", self._device)
        try:
            self._model = self._model.to(self._device)
            logger.info("Model moved to %s successfully.", self._device)
        except Exception as exc:
            self._tokenizer = None
            self._model = None
            logger.exception("Failed to move model to device '%s'.", self._device)
            raise ModelLoadError(
                f"Failed to move model to device {self._device}: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 6 — Evaluation mode
        # ------------------------------------------------------------------
        self._model.eval()
        logger.info("Model set to evaluation mode (gradients disabled).")

        # ------------------------------------------------------------------
        # Step 7 — Log label mapping and mark as ready
        # ------------------------------------------------------------------
        if hasattr(self._model, "config"):
            id2label = getattr(self._model.config, "id2label", {})
            num_labels = getattr(self._model.config, "num_labels", "unknown")
            logger.info("num_labels : %s", num_labels)
            if id2label:
                logger.info(
                    "Label mapping : %s",
                    {str(k): v for k, v in id2label.items()},
                )
            else:
                logger.warning(
                    "No id2label mapping found in model config. "
                    "Ensure config.json contains 'id2label'."
                )

        self._is_loaded = True
        logger.info("=" * 60)
        logger.info(
            "Model ready for inference | device=%s | vocab_size=%d",
            self._device,
            self._tokenizer.vocab_size,
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Lifecycle: unload
    # ------------------------------------------------------------------

    async def unload(self) -> None:
        """
        Release model and tokenizer resources.

        Called during application shutdown (FastAPI lifespan context) to
        free GPU VRAM and system RAM. Sets is_loaded to False so any
        in-flight requests get a clean 503.
        """
        logger.info("Releasing model resources...")

        self._tokenizer = None
        self._model = None

        # Free CUDA cache if it was in use
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

        self._device = None
        self._is_loaded = False
        logger.info("Model and tokenizer unloaded successfully.")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_model_directory(self) -> None:
        """
        Ensure the model directory exists and contains the minimum set of
        required files for XLM-RoBERTa to load correctly.

        Required files:
            - config.json
            - tokenizer.json           (fast tokenizer — sufficient without sentencepiece)
            - model.safetensors        OR pytorch_model.bin

        Raises:
            ModelLoadError: If the directory is missing or incomplete.
        """
        if not self._model_path.exists():
            raise ModelLoadError(
                f"Model directory not found: '{self._model_path}'. "
                "Please place the trained model weights in the configured "
                f"MODEL_DIR (currently: '{settings.model_dir}')."
            )

        if not self._model_path.is_dir():
            raise ModelLoadError(
                f"Model path is not a directory: '{self._model_path}'."
            )

        # Each inner list = at least one of these files must exist
        required_any = [
            ["model.safetensors", "pytorch_model.bin"],  # model weights
            ["config.json"],                              # model architecture config
            ["tokenizer.json"],                           # fast tokenizer (XLM-RoBERTa)
        ]

        missing: list[str] = []
        for alternatives in required_any:
            if not any((self._model_path / f).exists() for f in alternatives):
                missing.append(" or ".join(alternatives))

        if missing:
            raise ModelLoadError(
                f"Model directory '{self._model_path}' is missing required files: "
                + ", ".join(f"[{m}]" for m in missing)
                + ". Ensure all model artifacts are present before starting."
            )

        # Log all files found for diagnostics
        found_files = [f.name for f in self._model_path.iterdir() if f.is_file()]
        logger.info(
            "Model directory validated | files_found=%s",
            found_files,
        )

    # ------------------------------------------------------------------
    # Inference stubs (to be implemented in predict.py)
    # ------------------------------------------------------------------

    async def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a single pre-processed text string.

        Args:
            text:      Cleaned input text ready for tokenization.
            threshold: Unused for now (binary model uses argmax). Reserved
                       for future threshold-based override.

        Returns:
            Dictionary with keys:
                label              (str)   — predicted class name (e.g. 'hate')
                confidence         (float) — softmax probability of predicted class
                scores             (dict)  — {label: probability} for all classes
                language_detected  (None)  — reserved for future language detection
                processing_time_ms (float) — wall-clock inference time

        Raises:
            ModelNotLoadedError: If the model has not been loaded.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                "The ML model is not loaded. "
                "Ensure model weights exist at the configured MODEL_DIR."
            )

        results = await self._run_inference([text])
        return results[0]

    async def predict_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of pre-processed text strings.

        Args:
            texts:     List of cleaned input strings (1–32 items).
            threshold: Unused for now. Reserved for future threshold override.

        Returns:
            List of result dicts in the same order as *texts*.

        Raises:
            ModelNotLoadedError: If the model has not been loaded.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                "The ML model is not loaded. "
                "Ensure model weights exist at the configured MODEL_DIR."
            )

        return await self._run_inference(texts)

    # ------------------------------------------------------------------
    # Core inference (synchronous, called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _infer(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Synchronous inference method.

        Runs inside a thread pool (via asyncio.to_thread) so the async
        event loop is never blocked during the torch forward pass.

        Steps:
            1. Tokenize all texts in one batch.
            2. Move tensors to the model's device.
            3. Forward pass inside torch.no_grad().
            4. Apply softmax to convert raw logits → probabilities.
            5. Select the highest-probability label via argmax.
            6. Build and return a result dict for each input.

        Args:
            texts: List of cleaned input strings.

        Returns:
            List of result dicts (one per input text).
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1: Tokenise
        # ------------------------------------------------------------------
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length,
            return_tensors="pt",
        )

        # ------------------------------------------------------------------
        # Step 2: Move tensors to device
        # ------------------------------------------------------------------
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # ------------------------------------------------------------------
        # Step 3: Forward pass — no gradient tracking needed at inference
        # ------------------------------------------------------------------
        with torch.no_grad():
            outputs = self._model(**encoded)

        # ------------------------------------------------------------------
        # Step 4: Softmax → probabilities  [batch_size, num_labels]
        # ------------------------------------------------------------------
        probabilities = F.softmax(outputs.logits, dim=-1)   # shape: (N, num_labels)

        total_elapsed_ms = (time.perf_counter() - t0) * 1_000
        per_item_ms = total_elapsed_ms / len(texts)

        # ------------------------------------------------------------------
        # Step 5 & 6: Build result for each item
        # ------------------------------------------------------------------
        id2label: Dict[int, str] = self._model.config.id2label
        results: List[Dict[str, Any]] = []

        for i, probs in enumerate(probabilities):
            probs_list = probs.cpu().tolist()          # [p0, p1, ...]
            pred_idx   = int(probs.argmax().item())    # index of highest prob
            pred_label = id2label[pred_idx]            # e.g. 'hate'
            confidence = probs_list[pred_idx]

            # Build full label → probability mapping
            scores = {
                id2label[idx]: round(prob, 6)
                for idx, prob in enumerate(probs_list)
            }

            results.append({
                "label":              pred_label,
                "confidence":         round(confidence, 6),
                "scores":             scores,
                "language_detected":  None,        # reserved for future use
                "processing_time_ms": round(per_item_ms, 3),
            })

            logger.debug(
                "[%d/%d] label=%s confidence=%.4f",
                i + 1, len(texts), pred_label, confidence,
            )

        return results

    async def _run_inference(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Async wrapper: runs the synchronous *_infer* method in a thread pool
        so the FastAPI event loop is not blocked during the torch forward pass.
        """
        return await asyncio.to_thread(self._infer, texts)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Single shared instance. Import this object in routes and tests.
#: Do NOT instantiate ModelService elsewhere.
model_service = ModelService()
