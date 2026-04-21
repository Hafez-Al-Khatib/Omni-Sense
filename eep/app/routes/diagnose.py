"""
Diagnose Route
================
POST /api/v1/diagnose — The primary diagnostic endpoint.

Flow:
    1. Validate audio payload (size, format)
    2. Signal QA checks (dead sensor, clipping)
    3. Amplitude-threshold baseline (industry 80dB trigger)
    4. Extract 39-d physics features locally (replaces decommissioned IEP1)
    5. Fan-out in parallel to IEP2 (XGBoost+RF, classical) and IEP4 (CNN)
    6. Weighted ensemble of IEP2 and IEP4 probabilities (OOD short-circuits)
    7. Fire-and-forget dispatch to IEP3 if high-confidence fault
    8. Return diagnosis or safety (OOD) exception
"""

import asyncio
import io
import json
import logging
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from prometheus_client import Counter

from app.config import settings

BASELINE_DECISIONS = Counter(
    "eep_baseline_decisions_total",
    "Amplitude-threshold baseline classifier decisions",
    ["decision"],  # "leak_detected" | "no_leak"
)
from app.middleware.rate_limiter import limiter
from app.services.baseline import run_baseline
from app.services.orchestrator import (
    OrchestratorError,
    call_iep2_diagnose,
    call_iep3_notify,
    call_iep4_classify,
    ensemble_iep2_iep4,
    extract_features_local,
)
# Back-compat import alias (kept for tests that patch `call_iep1_embed`).
from app.services.orchestrator import extract_features_local as call_iep1_embed  # noqa: F401
from app.services.signal_qa import check_signal_quality

logger = logging.getLogger("eep.diagnose")
router = APIRouter()


@router.post("/diagnose")
@limiter.limit(settings.RATE_LIMIT)
async def diagnose(
    request: Request,
    audio: UploadFile = File(..., description="5-second audio file (WAV/OGG)"),
    metadata: str = Form(
        default='{"pipe_material": "PVC", "pressure_bar": 3.0}',
        description="JSON metadata: pipe_material, pressure_bar",
    ),
):
    """
    Diagnose infrastructure health from an audio sample.

    Upload a 5-second audio recording along with metadata about the
    recording environment. The system will:
    1. Validate the audio signal quality
    2. Extract acoustic features via YAMNet
    3. Check for Out-of-Distribution environments
    4. Classify leak probability
    """
    start_time = time.time()

    # ── Parse metadata ──
    try:
        meta = json.loads(metadata)
        pipe_material = meta.get("pipe_material", "PVC")
        pressure_bar = float(meta.get("pressure_bar", 3.0))
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metadata JSON: {str(e)}",
        )

    # Validate pipe_material
    if pipe_material not in ("PVC", "Steel", "Cast_Iron"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipe_material: {pipe_material}. Must be PVC, Steel, or Cast_Iron.",
        )

    # ── Size guard: reject oversized uploads BEFORE reading into memory ──
    # Check the Content-Length header first so we can 413 well-behaved clients
    # before any bytes land in RAM.  Then cap the streaming read to
    # (max_bytes + 1) so chunked / headerless uploads are also bounded.
    max_bytes = int(settings.MAX_AUDIO_SIZE_MB * 1024 * 1024)
    content_length_hdr = request.headers.get("content-length")
    if content_length_hdr is not None:
        try:
            cl = int(content_length_hdr)
            if cl > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Content-Length {cl} exceeds maximum allowed "
                        f"{max_bytes} bytes ({settings.MAX_AUDIO_SIZE_MB} MB)."
                    ),
                )
        except ValueError:
            pass  # malformed header — let the read-cap below catch it

    # Read at most (max_bytes + 1) bytes so we can detect oversize uploads
    # that omit Content-Length (e.g. chunked transfer) without loading the
    # entire file into memory first.
    audio_bytes = await audio.read(max_bytes + 1)

    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio file too large: exceeds {settings.MAX_AUDIO_SIZE_MB} MB limit."
            ),
        )

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # ── Decode and Signal QA ──
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot decode audio: {str(e)}. Supported formats: WAV, OGG, FLAC.",
        )

    quality = check_signal_quality(
        audio_data,
        silence_threshold=settings.SILENCE_RMS_THRESHOLD,
        clipping_threshold=settings.CLIPPING_PEAK_THRESHOLD,
    )
    baseline = run_baseline(audio_data)
    BASELINE_DECISIONS.labels(decision=baseline["baseline_decision"]).inc()

    if not quality["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Signal Quality Check Failed",
                "hardware_status": quality["hardware_status"],
                "message": quality["error"],
                "rms": quality["rms"],
                "peak": quality["peak"],
                "clipping_ratio": quality["clipping_ratio"],
            },
        )

    # ── Extract 39-d physics features locally (replaces decommissioned IEP1) ──
    try:
        embedding = await extract_features_local(audio_bytes)
    except OrchestratorError as e:
        logger.error(f"Local feature extraction error: {e}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": "Feature Extraction Failed",
                "message": str(e),
                "service": "EEP DSP pipeline",
                **e.detail,
            },
        )

    # ── Parallel fan-out: IEP2 (classical ML) + IEP4 (CNN) ──
    # asyncio.gather runs both calls concurrently.
    # IEP4 errors are suppressed (returns None) — the pipeline never fails
    # due to IEP4 being unavailable or not yet trained.
    iep2_task = call_iep2_diagnose(embedding, pipe_material, pressure_bar)
    iep4_task = call_iep4_classify(audio_bytes)

    try:
        iep2_result, iep4_result = await asyncio.gather(
            iep2_task,
            iep4_task,
            return_exceptions=False,   # IEP2 errors still propagate
        )
    except OrchestratorError as e:
        logger.error(f"IEP2 orchestration error: {e}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": "Diagnosis Failed",
                "message": str(e),
                "service": "IEP2 (Diagnostic Engine)",
                **e.detail,
            },
        )

    # ── Ensemble IEP2 + IEP4 ──
    result = ensemble_iep2_iep4(iep2_result, iep4_result)

    elapsed_ms = (time.time() - start_time) * 1000

    # ── OOD Response ──
    if result.get("is_ood"):
        return JSONResponse(
            status_code=422,
            content={
                **result,
                "hardware_status": quality["hardware_status"],
                "signal_quality": quality,
                "baseline_decision": baseline["baseline_decision"],
                "baseline_rms": baseline["baseline_rms"],
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

    # ── Fire-and-forget dispatch to IEP3 when confidence is high ──
    label = result.get("label", "")
    confidence = result.get("confidence", 0.0)
    if label not in ("No_Leak", "Normal_Operation") and confidence >= settings.DISPATCH_CONFIDENCE_THRESHOLD:
        asyncio.create_task(
            call_iep3_notify(
                label=label,
                confidence=confidence,
                probabilities=result.get("probabilities", {}),
                anomaly_score=result.get("anomaly_score", 0.0),
                pipe_material=pipe_material,
                pressure_bar=pressure_bar,
                scada_mismatch=result.get("scada_mismatch", False),
            )
        )

    # ── Success Response ──
    return {
        **result,
        "hardware_status": quality["hardware_status"],
        "signal_quality": quality,
        "baseline_decision": baseline["baseline_decision"],
        "baseline_rms": baseline["baseline_rms"],
        "elapsed_ms": round(elapsed_ms, 1),
    }
