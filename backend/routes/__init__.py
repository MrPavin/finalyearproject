"""
routes/__init__.py
==================
Public re-exports for the routes package.
"""

from routes.prediction import router as prediction_router

__all__ = ["prediction_router"]
