"""
app.py
======
FastAPI application factory.

This module:
- Creates and configures the FastAPI application instance.
- Sets up structured logging.
- Registers global middleware (CORS, request ID).
- Registers exception handlers.
- Mounts all API routers under the versioned prefix.
- Defines the /health endpoint.

Import `create_app()` to get a fully configured application, or
run this module directly to start the development server.
"""

import logging
import logging.handlers
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from routes.prediction import router as prediction_router
from schemas.prediction_schema import HealthResponse  # noqa: F401 — kept for re-export / test imports
from services.model_service import ModelLoadError, model_service
from utils.helper import utcnow_iso

settings = get_settings()


# ===========================================================================
# Logging setup
# ===========================================================================

def _configure_logging() -> None:
    """
    Configure the root logger with:
    - A rotating file handler writing to logs/app.log.
    - A coloured stream handler for console output.
    """
    log_dir: Path = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Suppress overly verbose third-party loggers in production
    for noisy in ("httpx", "httpcore", "transformers", "torch", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger(__name__)


# ===========================================================================
# Lifespan
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Startup:  Load the XLM-RoBERTa model into memory and print status.
    Shutdown: Unload model and release resources.
    """
    logger.info("=== Starting up %s v%s ===", settings.app_name, settings.app_version)
    logger.info("Environment : %s", settings.environment)
    logger.info("Debug mode  : %s", settings.debug)

    # ------------------------------------------------------------------
    # Load model — catch ModelLoadError so the server still starts
    # even when weights are absent (health endpoint will report degraded).
    # ------------------------------------------------------------------
    try:
        await model_service.load()

        # Console banner — ASCII-safe for all terminal encodings
        print("\n[OK] XLM-RoBERTa model loaded successfully.")

        # Report which device the model is running on
        device_type = model_service.device.type          # 'cuda', 'mps', or 'cpu'
        if device_type == "cuda":
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            print(f"     Running on CUDA  ({gpu_name})")
        elif device_type == "mps":
            print("     Running on MPS   (Apple Silicon)")
        else:
            print("     Running on CPU")

        print()   # blank line for readability
        sys.stdout.flush()

    except ModelLoadError as exc:
        logger.error(
            "Model failed to load — API will run in DEGRADED mode. "
            "Prediction endpoints will return HTTP 503 until the model is available. "
            "Reason: %s",
            exc,
        )
        print(f"\n[ERROR] Model failed to load: {exc}")
        print("        API is running in DEGRADED mode — predictions unavailable.")
        print()
        sys.stdout.flush()

    yield  # Application runs here

    # Shutdown cleanup
    logger.info("=== Shutting down %s ===", settings.app_name)
    await model_service.unload()


# ===========================================================================
# Application factory
# ===========================================================================

def create_app() -> FastAPI:
    """
    Create and fully configure the FastAPI application.

    Returns:
        Configured FastAPI instance ready to be served by an ASGI server.
    """

    # ------------------------------------------------------------------
    # Core application
    # ------------------------------------------------------------------
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        swagger_ui_parameters={
            "syntaxHighlight.theme": "obsidian",
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
        },
    )

    # ------------------------------------------------------------------
    # CORS middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Request ID & timing middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next):  # type: ignore[return]
        """Attach a unique request ID and measure response time."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "%s %s | status=%d | %.2f ms | id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "HTTP %d | %s %s | detail=%s | id=%s",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
            request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "request_id": request_id,
                "timestamp": utcnow_iso(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception(
            "Unhandled exception | %s %s | id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error. Please try again later.",
                "request_id": request_id,
                "timestamp": utcnow_iso(),
            },
        )

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(prediction_router, prefix=settings.api_v1_prefix)

    # ------------------------------------------------------------------
    # Health check endpoint
    # ------------------------------------------------------------------
    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check",
        description=(
            "Returns the operational status of the API and whether the "
            "XLM-RoBERTa model is loaded in memory."
        ),
        responses={
            200: {
                "description": "Service is healthy",
                "content": {
                    "application/json": {
                        "example": {"status": "healthy", "model": "loaded"}
                    }
                },
            }
        },
    )
    async def health_check() -> JSONResponse:
        """
        Lightweight health probe.

        Returns:
            200 with ``{"status": "healthy", "model": "loaded"}`` when the
            model is ready, or ``{"status": "degraded", "model": "unavailable"}``
            when the model failed to load.
        """
        if model_service.is_loaded:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "healthy", "model": "loaded"},
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "degraded", "model": "unavailable"},
        )

    # ------------------------------------------------------------------
    # Root redirect info
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.app_name}",
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/health",
            }
        )

    logger.info("Application factory complete — routers registered.")
    return app


# ===========================================================================
# Application instance (used by ASGI server)
# ===========================================================================
app = create_app()
