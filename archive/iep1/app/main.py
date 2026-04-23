"""
IEP 1 — Vibration Feature Extractor
======================================
FastAPI service that converts raw WAV audio (piezoelectric accelerometer
recordings) into 208-dimensional vibration feature vectors for downstream
fault classification.

Feature set (208 dims): MFCC mean/std, MFCC delta mean/std, spectral
centroid/bandwidth/rolloff/contrast, ZCR, RMS, chroma.
No TensorFlow dependency — feature extraction is pure librosa.
"""

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.audio_processor import preprocess_audio
from app.feature_extractor import N_FEATURES, feature_extractor
from app.schemas import EmbeddingResponse, HealthResponse

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep1")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
INFERENCE_DURATION = Histogram(
    "iep1_inference_duration_seconds",
    "Time spent on vibration feature extraction (excluding I/O)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREPROCESSING_DURATION = Histogram(
    "iep1_preprocessing_duration_seconds",
    "Time spent on audio preprocessing",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

REQUESTS_TOTAL = Counter(
    "iep1_requests_total",
    "Total feature extraction requests",
    ["status"],
)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Omni-Sense IEP1 — Vibration Feature Extractor",
    description=(
        f"Converts piezoelectric accelerometer WAV recordings into "
        f"{N_FEATURES}-d vibration feature vectors (MFCC + spectral)."
    ),
    version="0.2.0",
)

Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup():
    """Initialize feature extractor on startup."""
    logger.info("Initializing vibration feature extractor...")
    feature_extractor.load()
    logger.info(f"Feature extractor ready ({N_FEATURES}-d output).")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=feature_extractor.is_loaded,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def extract_features(audio: UploadFile = File(...)):
    """
    Extract a vibration feature vector from an uploaded audio file.

    Accepts: WAV, OGG, FLAC audio files
    Returns: 208-dimensional float array
    """
    if not feature_extractor.is_loaded:
        raise HTTPException(status_code=503, detail="Feature extractor not ready")

    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        if len(audio_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file too large (max 5MB)")
    except HTTPException:
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {str(e)}")

    try:
        with PREPROCESSING_DURATION.time():
            waveform = preprocess_audio(audio_bytes)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=f"Audio preprocessing failed: {str(e)}")

    try:
        with INFERENCE_DURATION.time():
            features = feature_extractor.extract_features(waveform)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        logger.error(f"Feature extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    REQUESTS_TOTAL.labels(status="success").inc()

    return EmbeddingResponse(
        embedding=features.tolist(),
        embedding_dim=N_FEATURES,
        duration_samples=len(waveform),
        sample_rate=16000,
    )
