"""
predict.py
==========
Top-level prediction helper used by predict.py CLI and tests.

This thin wrapper delegates to the ModelService so that callers do
not need to import service internals directly.

Usage (CLI / script)::

    python predict.py --text "Sample text to classify"

Usage (programmatic)::

    from predict import run_prediction
    result = await run_prediction("Some text here")
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from services.model_service import model_service
from utils.helper import sanitize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public async helper
# ---------------------------------------------------------------------------

async def run_prediction(
    text: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper around ModelService.predict().

    Loads the model on first call if not already loaded.

    Args:
        text:      Raw input text to classify.
        threshold: Optional confidence threshold override.

    Returns:
        Prediction result dictionary from ModelService.

    Raises:
        NotImplementedError: Until model inference is implemented.
    """
    if not model_service.is_loaded:
        await model_service.load()

    clean = sanitize_text(text)
    return await model_service.predict(text=clean, threshold=threshold)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hate speech prediction from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --text "I hate this!"
  python predict.py --text "Hello world" --threshold 0.7
        """,
    )
    parser.add_argument(
        "--text",
        required=True,
        help="The text to classify.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold (0.0–1.0). Defaults to server setting.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    result = await run_prediction(text=args.text, threshold=args.threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
