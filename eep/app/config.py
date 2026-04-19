"""
EEP Configuration
==================
Settings loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Service URLs
    IEP1_URL: str = "http://iep1:8001"
    IEP2_URL: str = "http://iep2:8002"

    # Rate limiting
    RATE_LIMIT: str = "5/minute"

    # Payload constraints
    MAX_AUDIO_SIZE_MB: float = 5.0
    ALLOWED_AUDIO_TYPES: list[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/ogg",
        "audio/flac",
        "application/octet-stream",  # Fallback for untyped uploads
    ]

    # Signal QA thresholds
    SILENCE_RMS_THRESHOLD: float = 0.001   # Below = "Dead Sensor"
    CLIPPING_PEAK_THRESHOLD: float = 0.99  # Above = "Broken Mic"

    # CORS — never use ["*"] in production (CSRF risk).
    # Set OMNI_CORS_ORIGINS as a comma-separated list in the environment, e.g.:
    #   OMNI_CORS_ORIGINS='["https://dashboard.omni-sense.io","https://ops.omni-sense.io"]'
    # The default allows local dev only.
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",   # web UI dev server
        "http://localhost:8501",   # Streamlit ops console
    ]

    # IEP3 — Dispatch & Active Learning
    IEP3_URL: str = "http://iep3:8003"
    IEP3_TIMEOUT: float = 5.0
    DISPATCH_CONFIDENCE_THRESHOLD: float = 0.90

    # IEP4 — Deep CNN Classifier (parallel with IEP2)
    IEP4_URL: str = "http://iep4:8004"
    IEP4_TIMEOUT: float = 15.0

    # Timeouts (seconds)
    IEP1_TIMEOUT: float = 30.0
    IEP2_TIMEOUT: float = 10.0

    class Config:
        env_prefix = "OMNI_"
        env_file = ".env"


settings = Settings()
