"""
IEP 1 — YAMNet Embedding Service
==================================
FastAPI wrapper around the YAMNet model from TensorFlow Hub.
Accepts raw WAV audio bytes and returns a 1024-dimensional
mean-pooled embedding vector.
"""

import io
import time
import logging

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Histogram, Counter
from prometheus_fastapi_instrumentator import Instrumentator

from app.yamnet_service import YAMNetService
from app.audio_processor import preprocess_audio
from app.schemas import EmbeddingResponse, HealthResponse

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep1")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
INFERENCE_DURATION = Histogram(
    "iep1_inference_duration_seconds",
    "Time spent on YAMNet inference (excluding I/O)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREPROCESSING_DURATION = Histogram(
    "iep1_preprocessing_duration_seconds",
    "Time spent on audio preprocessing",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

REQUESTS_TOTAL = Counter(
    "iep1_requests_total",
    "Total embedding requests",
    ["status"],
)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Omni-Sense IEP1 — YAMNet Embedding Service",
    description="Extracts 1024-d acoustic embeddings from audio using YAMNet.",
    version="0.1.0",
)

# Auto-instrument standard HTTP metrics
Instrumentator().instrument(app).expose(app)

# Singleton model service
yamnet_service = YAMNetService()


@app.on_event("startup")
async def startup():
    """Load YAMNet model on startup."""
    logger.info("Loading YAMNet model...")
    yamnet_service.load()
    logger.info("YAMNet model loaded successfully.")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=yamnet_service.is_loaded,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def extract_embedding(audio: UploadFile = File(...)):
    """
    Extract a 1024-d YAMNet embedding from an uploaded audio file.

    Accepts: WAV, OGG, FLAC audio files
    Returns: 1024-dimensional float array (mean-pooled across time frames)
    """
    if not yamnet_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read audio bytes
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        if len(audio_bytes) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=413, detail="Audio file too large (max 5MB)")
    except HTTPException:
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {str(e)}")

    # Preprocess
    try:
        with PREPROCESSING_DURATION.time():
            waveform = preprocess_audio(audio_bytes)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=f"Audio preprocessing failed: {str(e)}")

    # Inference
    try:
        with INFERENCE_DURATION.time():
            embedding = yamnet_service.extract_embedding(waveform)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    REQUESTS_TOTAL.labels(status="success").inc()

    return EmbeddingResponse(
        embedding=embedding.tolist(),
        embedding_dim=len(embedding),
        duration_samples=len(waveform),
        sample_rate=16000,
    )
