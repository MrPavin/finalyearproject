"""
config.py
=========
Centralised application configuration using Pydantic Settings.
All values are read from the .env file or environment variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from the .env file.

    Using Pydantic's BaseSettings ensures:
    - Automatic type coercion
    - Environment variable overrides
    - Clear documentation of all configurable values
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "Hate Speech Detection API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Context-Aware Multilingual Hate Speech Detection Using XLM-RoBERTa"
    )
    debug: bool = False
    environment: str = "development"

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_dir: str = "models/xlm_roberta_hate_model"
    model_name: str = "xlm-roberta-base"
    max_sequence_length: int = 512
    prediction_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "app.log"
    log_max_bytes: int = 10_485_760  # 10 MB
    log_backup_count: int = 5

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"
    rate_limit_per_minute: int = 60

    # ------------------------------------------------------------------
    # Derived helpers (not loaded from env)
    # ------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """Absolute path to the model directory."""
        return Path(self.model_dir).resolve()

    @property
    def log_path(self) -> Path:
        """Absolute path to the log directory."""
        return Path(self.log_dir).resolve()

    @property
    def is_production(self) -> bool:
        """True when running in production mode."""
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton instance of Settings.

    Using @lru_cache ensures the .env file is parsed only once per
    application lifetime, avoiding repeated disk I/O.

    Usage:
        from config import get_settings
        settings = get_settings()
    """
    return Settings()
