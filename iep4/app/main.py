"""
IEP 4 — Deep CNN Classifier with Autoencoder OOD Detection
===========================================================
FastAPI service that classifies 5-second accelerometer WAV recordings
using a small 2-D CNN trained on log-linear STFT spectrograms.

Two-stage pipeline
------------------
Stage 1: CNN Autoencoder OOD check (Taiwan Water Corp design, 99.07% acc).
          Trained on Normal_Operation samples only.  High reconstruction error
          → acoustic environment not seen during training → OOD.
          If OOD: classification still proceeds, but the response flags is_ood=True
          so the EEP can discount IEP4's vote and trigger a support ticket.

Stage 2: 2-D CNN spectrogram classifier (SA Water 92.44% acc).
          Log-magnitude linear-frequency STFT → Leak / No_Leak.

Architecture distinction from IEP1 + IEP2:
  IEP1 extracts hand-crafted physics features (kurtosis, wavelet, envelope).
  IEP2 classifies those features with XGBoost + Random Forest.
  IEP4 learns its own filters end-to-end from raw spectrograms.

Endpoints:
  POST /classify   — classify a WAV file
  GET  /health     — health + model status
  GET  /metrics    — Prometheus metrics

Graceful degradation:
  CNN not trained           → /classify returns HTTP 503
  Autoencoder not trained   → is_ood always False, ood_reconstruction_error=None
  Both degrade independently; IEP2 result is always available from EEP.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import cnn_classifier
from app.schemas import CNNResponse, HealthResponse
from app.audio import preprocess_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep4")

# ─── Prometheus ───────────────────────────────────────────────────────────────

CNN_DURATION = Histogram(
    "iep4_cnn_inference_duration_seconds",
    "End-to-end CNN inference time (spectrogram + forward pass)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
CNN_REQUESTS = Counter(
    "iep4_requests_total",
    "Total CNN classification requests",
    ["status"],
)
CNN_CONFIDENCE = Histogram(
    "iep4_cnn_prediction_confidence",
    "CNN prediction confidence distribution",
    buckets=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
)
AUTOENCODER_RECON_ERROR = Histogram(
    "iep4_autoencoder_reconstruction_error",
    "CNN Autoencoder reconstruction error (OOD proxy)",
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
)
OOD_DETECTIONS = Counter(
    "iep4_ood_detections_total",
    "Number of frames flagged as out-of-distribution by the CNN autoencoder",
)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Omni-Sense IEP4 — Deep CNN Classifier",
    description=(
        "Two-stage pipeline: "
        "(1) CNN Autoencoder OOD detection (Taiwan Water Corp design), "
        "(2) 2-D CNN spectrogram classifier (SA Water design). "
        "Returns 503 until CNN weights are available."
    ),
    version="0.2.0",
)

Instrumentator().instrument(app).expose(app)

# ─── Autoencoder (lazy-loaded to avoid torch import at startup) ───────────────

_autoencoder = None


def _get_autoencoder():
    """Return the AutoencoderOODDetector singleton, or None if not loadable."""
    global _autoencoder
    if _autoencoder is not None:
        return _autoencoder

    # Model lives in iep2/models/ (shared between IEP2 and IEP4)
    from pathlib import Path
    onnx_p = Path("../iep2/models/autoencoder_ood.onnx")
    pt_p   = Path("../iep2/models/autoencoder_ood.pt")
    thr_p  = Path("../iep2/models/autoencoder_threshold.npy")

    # Also check local models/ directory
    local_onnx = Path("models/autoencoder_ood.onnx")
    local_pt   = Path("models/autoencoder_ood.pt")
    local_thr  = Path("models/autoencoder_threshold.npy")

    if local_onnx.exists() or local_pt.exists():
        onnx_p, pt_p, thr_p = local_onnx, local_pt, local_thr

    if not (onnx_p.exists() or pt_p.exists()):
        logger.info(
            "Autoencoder model not found — OOD detection disabled. "
            "Run scripts/train_autoencoder.py to enable."
        )
        return None

    try:
        import sys, os
        # Ensure iep2/app is importable
        proj_root = str(Path(__file__).parent.parent.parent)
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

        from iep2.app.autoencoder_ood_detector import AutoencoderOODDetector
        ae = AutoencoderOODDetector()
        ae.load(onnx_path=onnx_p if onnx_p.exists() else None,
                pt_path=pt_p if pt_p.exists() else None,
                threshold_path=thr_p if thr_p.exists() else None)
        _autoencoder = ae
        logger.info("Autoencoder OOD detector loaded (backend=%s)", ae._backend)
        return _autoencoder
    except Exception as exc:
        logger.warning("Autoencoder load failed: %s — OOD disabled", exc)
        return None


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Loading CNN classifier...")
    try:
        cnn_classifier.load()
        if cnn_classifier.is_loaded:
            logger.info("CNN ready (backend=%s)", cnn_classifier._backend)
        else:
            logger.warning(
                "CNN model not yet trained — service will return 503 on /classify. "
                "Run: python scripts/train_cnn.py --clips-dir data/synthesized "
                "--output-dir iep4/models --epochs 150 --binary"
            )
    except Exception as exc:
        logger.error("CNN load error: %s", exc)

    # Try to load autoencoder (non-fatal)
    _get_autoencoder()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    ae = _get_autoencoder()
    return HealthResponse(
        status="healthy",
        model_loaded=cnn_classifier.is_loaded,
        backend=cnn_classifier._backend,
        autoencoder_loaded=(ae is not None and ae.is_loaded),
    )


# ─── Classify Raw (called by Omni orchestrator) ───────────────────────────────

@app.post("/classify_raw")
async def classify_raw(body: dict):
    """
    Classify raw PCM float32 bytes sent as base64 from the Omni orchestrator.

    Request body:
        {"pcm_b64": "<base64 of float32 numpy array>", "sr": 16000}

    Response:
        {"p_leak": 0.72, "label": "Leak", "confidence": 0.72, "is_ood": false}

    This endpoint exists so the Omni EEP orchestrator can call IEP4 directly
    without WAV encoding overhead.  The orchestrator applies a 120 ms timeout
    and falls back to the CNN stub on any failure or 503.
    """
    if not cnn_classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="CNN model not yet trained. Run scripts/train_cnn.py.",
        )

    try:
        import base64 as _b64
        import numpy as _np
        pcm_bytes = _b64.b64decode(body["pcm_b64"])
        pcm = _np.frombuffer(pcm_bytes, dtype=_np.float32).copy()
        sr = int(body.get("sr", 16000))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PCM decode failed: {exc}")

    if len(pcm) < 1600:
        raise HTTPException(status_code=422, detail="PCM too short (need ≥ 0.1 s)")

    # Pad/trim to 5 s (80 000 samples) matching the CNN training window
    target = sr * 5
    if len(pcm) > target:
        start = (len(pcm) - target) // 2
        pcm = pcm[start: start + target]
    elif len(pcm) < target:
        import numpy as _np2
        pcm = _np2.pad(pcm, (0, target - len(pcm)))

    # Autoencoder OOD check (best-effort, non-fatal)
    is_ood = False
    ae = _get_autoencoder()
    if ae is not None and ae.is_loaded:
        try:
            is_ood, _ = ae.is_anomalous_wav(pcm)
        except Exception:
            pass

    # CNN classification
    try:
        with CNN_DURATION.time():
            result = cnn_classifier.predict(pcm)
    except Exception as exc:
        CNN_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"CNN inference failed: {exc}")

    CNN_REQUESTS.labels(status="success").inc()
    CNN_CONFIDENCE.observe(result["confidence"])

    probs = result.get("probabilities", {})
    p_leak = float(probs.get("Leak", 1.0 - probs.get("No_Leak", 0.5)))

    return {
        "p_leak": float(p_leak),
        "label": result["label"],
        "confidence": result["confidence"],
        "is_ood": is_ood,
        "backend": result["backend"],
    }


# ─── Classify (WAV file upload) ────────────────────────────────────────────────

@app.post("/classify", response_model=CNNResponse)
async def classify(audio: UploadFile = File(...)):
    """
    Two-stage pipeline: CNN Autoencoder OOD check → spectrogram CNN classification.

    Returns HTTP 503 if the CNN model has not been trained yet.
    The autoencoder OOD check is best-effort: if it is not trained, is_ood=False.
    """
    if not cnn_classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "CNN model not yet trained. "
                "Run scripts/train_cnn.py to train IEP4. "
                "IEP2 (XGBoost + RF) is still operational."
            ),
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        CNN_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 5 * 1024 * 1024:
        CNN_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=413, detail="Audio too large (max 5 MB)")

    try:
        waveform = preprocess_audio(audio_bytes)
    except Exception as exc:
        CNN_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=422, detail=f"Audio decode failed: {exc}")

    # ── Stage 1: Autoencoder OOD check ───────────────────────────────────
    recon_error: Optional[float] = None
    is_ood: bool                 = False
    ood_threshold: Optional[float] = None

    ae = _get_autoencoder()
    if ae is not None and ae.is_loaded:
        try:
            is_ood_flag, err = ae.is_anomalous_wav(waveform)
            recon_error   = err
            is_ood        = is_ood_flag
            ood_threshold = ae._threshold
            AUTOENCODER_RECON_ERROR.observe(err)
            if is_ood:
                OOD_DETECTIONS.inc()
                logger.warning(
                    "Autoencoder OOD: recon_error=%.5f > threshold=%.5f",
                    err, ae._threshold,
                )
        except Exception as exc:
            logger.debug("Autoencoder inference error: %s", exc)

    # ── Stage 2: CNN Classification ───────────────────────────────────────
    try:
        with CNN_DURATION.time():
            result = cnn_classifier.predict(waveform)
    except Exception as exc:
        CNN_REQUESTS.labels(status="error").inc()
        logger.error("CNN inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"CNN inference failed: {exc}")

    CNN_REQUESTS.labels(status="success").inc()
    CNN_CONFIDENCE.observe(result["confidence"])

    return CNNResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        backend=result["backend"],
        model_loaded=True,
        ood_reconstruction_error=recon_error,
        is_ood=is_ood,
        ood_threshold=ood_threshold,
    )
