"""
Calibrate Route
=================
POST /api/v1/calibrate — Dynamic OOD threshold calibration.

Accepts a multi-part audio upload of ambient recording(s),
extracts 39-d physics features locally (the old IEP1 microservice is
decommissioned), then calls IEP2 to adjust the Isolation Forest threshold.
"""

import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.middleware.rate_limiter import limiter
from app.services.orchestrator import (
    OrchestratorError,
    call_iep2_calibrate,
    extract_features_local,
)

logger = logging.getLogger("eep.calibrate")
router = APIRouter()


@router.post("/calibrate")
@limiter.limit("2/minute")
async def calibrate(
    request: Request,
    audio: UploadFile = File(
        ...,
        description="10-second ambient audio recording for calibration (WAV/OGG)",
    ),
):
    """
    Calibrate the system to a new acoustic environment.

    Upload a 10-second ambient recording from the deployment site.
    The system will:
    1. Split the recording into 5-second overlapping windows
    2. Extract 39-d physics features for each window (locally in EEP)
    3. Compute the ambient acoustic profile
    4. Adjust the OOD detection threshold accordingly
    """
    # ── Read audio ──
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    max_bytes = int(settings.MAX_AUDIO_SIZE_MB * 1024 * 1024)
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    # ── Decode ──
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot decode audio: {str(e)}",
        )

    # ── Split into overlapping 5-second windows ──
    # Window size and hop in samples (at original SR for now)
    duration_s = len(audio_data) / sr
    if duration_s < 3.0:
        raise HTTPException(
            status_code=400,
            detail=f"Recording too short ({duration_s:.1f}s). Need at least 3 seconds.",
        )

    window_s = 5.0
    hop_s = 2.5  # 50% overlap
    window_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)

    windows = []
    start = 0
    while start + window_samples <= len(audio_data):
        window = audio_data[start:start + window_samples]
        windows.append(window)
        start += hop_samples

    # If we have leftover audio that's at least 3 seconds, pad and include it
    if start < len(audio_data):
        remaining = audio_data[start:]
        if len(remaining) >= int(3.0 * sr):
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[:len(remaining)] = remaining
            windows.append(padded)

    if len(windows) == 0:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any valid windows from the recording.",
        )

    logger.info(f"Calibration: {len(windows)} windows from {duration_s:.1f}s recording")

    # ── Extract 39-d features for each window (locally, no network hop) ──
    embeddings = []
    for i, window in enumerate(windows):
        buffer = io.BytesIO()
        sf.write(buffer, window, sr, format="WAV")
        window_bytes = buffer.getvalue()

        try:
            embedding = await extract_features_local(window_bytes)
            embeddings.append(embedding)
        except OrchestratorError as e:
            logger.warning(f"Local feature extraction failed for window {i}: {e}")
            continue

    if len(embeddings) < 1:
        raise HTTPException(
            status_code=502,
            detail="Failed to extract any embeddings from ambient recording.",
        )

    # ── Call IEP2 to calibrate ──
    try:
        result = await call_iep2_calibrate(embeddings)
    except OrchestratorError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": "Calibration Failed",
                "message": str(e),
                **e.detail,
            },
        )

    return {
        **result,
        "windows_extracted": len(windows),
        "embeddings_used": len(embeddings),
        "recording_duration_s": round(duration_s, 1),
    }
