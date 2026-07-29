"""
utils/helper.py
===============
Shared utility functions used across the application.
"""

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Request
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def transliterate_to_roman(text: str) -> str:
    """
    Check if the text contains native Kannada characters.
    If it does, transliterate it to Roman (Latin) script using ITRANS,
    and convert to lowercase.
    This is required because the ML model was fine-tuned on Romanized
    (Kanglish) hate speech data rather than native Unicode text.
    """
    # Simple heuristic: Kannada Unicode block is U+0C80 to U+0CFF
    if any(0x0C80 <= ord(char) <= 0x0CFF for char in text):
        # Transliterate to ITRANS and lowercase it
        text = transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
        text = text.lower()
    return text

def sanitize_text(text: str) -> str:
    """
    Remove leading / trailing whitespace, collapse internal runs of
    whitespace, and transliterate native Indic scripts to Romanized
    forms that the model understands.

    Args:
        text: Raw input string from the client.

    Returns:
        Cleaned string ready for further processing.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = transliterate_to_roman(text)
    return text


def truncate_text(text: str, max_length: int = 512) -> str:
    """
    Truncate *text* to *max_length* characters, appending an ellipsis if
    truncation occurred.

    Args:
        text:       Input string.
        max_length: Maximum allowed character count (default 512).

    Returns:
        Original string if within limit, otherwise a truncated version.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def generate_request_id() -> str:
    """
    Generate a universally unique request identifier.

    Returns:
        A UUID4 string (e.g. "3d6f0f1a-4b2e-4a7c-9c3b-1a2b3c4d5e6f").
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utcnow_iso() -> str:
    """
    Return the current UTC time as an ISO-8601 string with timezone info.

    Returns:
        Example: "2025-01-15T10:30:00+00:00"
    """
    return datetime.now(tz=timezone.utc).isoformat()


class Timer:
    """
    Simple context-manager timer for measuring elapsed wall-clock time.

    Usage::

        with Timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1_000


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def extract_client_info(request: Request) -> Dict[str, Any]:
    """
    Extract useful client metadata from a FastAPI Request object.

    Args:
        request: The incoming FastAPI/Starlette request.

    Returns:
        Dictionary with client IP, user-agent, and request method/URL.
    """
    client_host = request.client.host if request.client else "unknown"
    return {
        "client_ip": client_host,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "method": request.method,
        "url": str(request.url),
    }


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def build_success_response(
    data: Any,
    message: str = "Success",
    request_id: str | None = None,
) -> Dict[str, Any]:
    """
    Wrap *data* in a standardised success envelope.

    Args:
        data:       The payload to return to the client.
        message:    A human-readable status message.
        request_id: Optional correlation ID for tracing.

    Returns:
        Standardised response dictionary.
    """
    return {
        "success": True,
        "message": message,
        "request_id": request_id or generate_request_id(),
        "timestamp": utcnow_iso(),
        "data": data,
    }


def build_error_response(
    error: str,
    detail: str | None = None,
    request_id: str | None = None,
) -> Dict[str, Any]:
    """
    Wrap *error* in a standardised error envelope.

    Args:
        error:      Short error category / title.
        detail:     Optional longer explanation.
        request_id: Optional correlation ID for tracing.

    Returns:
        Standardised error response dictionary.
    """
    return {
        "success": False,
        "error": error,
        "detail": detail,
        "request_id": request_id or generate_request_id(),
        "timestamp": utcnow_iso(),
    }
