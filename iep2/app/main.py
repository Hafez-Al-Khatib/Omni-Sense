"""
IEP 2 — Diagnostic & Safety Engine
=====================================
Two-stage inference pipeline:
  Stage 1: Autoencoder (OOD detection / safety watchdog)
  Stage 2: XGBoost (leak classification with calibrated probabilities)

Accepts 39-d physics features (DSP) + metadata from EEP, returns diagnosis.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    HAS_SLOWAPI = True
except ImportError:  # pragma: no cover
    # Fallback when slowapi is not installed (e.g., minimal Docker images)
    class _NoOpLimiter:
        def __init__(self, *args, **kwargs):
            pass
        def limit(self, *args, **kwargs):
            return lambda f: f
    Limiter = _NoOpLimiter
    RateLimitExceeded = Exception
    get_remote_address = lambda: "127.0.0.1"
    _rate_limit_exceeded_handler = None
    HAS_SLOWAPI = False

# Ensure omni root is in path for config imports
_PROJ_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from app.calibration import CalibrationManager
from app.classifier import LeakClassifier
from app.drift_monitor import drift_monitor
from app.ood_detector import OODDetector
from app.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
)
from omni.common.config import OOD_IF_THRESHOLD

# ─── OOD threshold override ───────────────────────────────────────────────────
# OMNI_OOD_THRESHOLD env var overrides the compiled default.
# calibration_mgr.threshold overrides both after /calibrate is called.
def _get_ood_threshold() -> float:
    env_val = os.getenv("OMNI_OOD_THRESHOLD")
    if env_val is not None:
        return float(env_val)
    if calibration_mgr.is_calibrated and calibration_mgr.threshold is not None:
        return calibration_mgr.threshold
    return OOD_IF_THRESHOLD

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep2")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
OOD_ANOMALY_SCORE = Gauge(
    "iep2_ood_anomaly_score",
    "Latest Isolation Forest anomaly score (OOD proxy)",
)
# ... [rest of metrics omitted for brevity in thought, but included in implementation] ...
PREDICTION_CONFIDENCE = Histogram(
    "xgboost_prediction_confidence",
    "XGBoost prediction confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

IEP2_INFERENCE_DURATION = Histogram(
    "iep2_inference_duration_seconds",
    "Total IEP2 inference time (OOD + classification)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

OOD_REJECTIONS = Counter(
    "iep2_ood_rejections_total",
    "Number of requests rejected as Out-of-Distribution",
)

SCADA_MISMATCHES = Counter(
    "iep2_scada_mismatches_total",
    "High-confidence predictions inconsistent with the provided pressure metadata",
)

# ─── SCADA Consistency ────────────────────────────────────────────────────────
# ... [rest of SCADA logic unchanged] ...
_PRESSURE_PHYSICS: dict[str, tuple[float, float]] = {
    "No_Leak":               (0.3, 20.0),
    "Orifice_Leak":          (0.5,  8.0),
    "Gasket_Leak":           (0.5,  8.0),
    "Longitudinal_Crack":    (0.5,  5.5),
    "Circumferential_Crack": (0.5,  5.5),
    "Normal_Operation":      (0.0, 20.0),
}
_SCADA_CONFIDENCE_GATE = 0.85


def _check_scada_consistency(
    label: str, confidence: float, pressure_bar: float
) -> tuple[bool, str | None]:
    if confidence < _SCADA_CONFIDENCE_GATE:
        return False, None

    p_min, p_max = _PRESSURE_PHYSICS.get(label, (0.0, 20.0))

    if pressure_bar < p_min:
        return True, (
            f"Pressure {pressure_bar:.1f} bar is below the physical minimum "
            f"({p_min} bar) for '{label}'."
        )

    if pressure_bar > p_max:
        return True, (
            f"Pressure {pressure_bar:.1f} bar is unexpectedly stable for "
            f"'{label}' at {confidence:.0%} confidence."
        )

    return False, None

# ─── App ──────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Omni-Sense IEP2 — Diagnostic & Safety Engine",
    description="OOD-aware acoustic classification: Isolation Forest + XGBoost.",
    version="0.2.0",
)

app.state.limiter = limiter
if HAS_SLOWAPI:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Instrumentator().instrument(app).expose(app)

# Service singletons
ood_detector = OODDetector()
classifier = LeakClassifier()
calibration_mgr = CalibrationManager()


_CENTROID_PATH = Path("models/centroid.npy")


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    try:
        logger.info("Loading Isolation Forest OOD model...")
        ood_detector.load()
        logger.info("Loading Leak Classifier ensemble...")
        classifier.load()
        logger.info("All models loaded.")
    except Exception as exc:
        logger.critical(f"Startup failed: could not load model artifacts — {exc}")
        # In a container environment, raising here will cause the container to restart/fail
        raise

    if _CENTROID_PATH.exists():
        centroid = np.load(_CENTROID_PATH)
        drift_monitor.set_reference_centroid(centroid)
        logger.info(f"Drift monitor: reference centroid loaded from {_CENTROID_PATH}")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        ood_model_loaded=ood_detector.is_loaded,
        classifier_loaded=classifier.is_loaded,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
@limiter.limit(os.getenv("OMNI_IEP_RATE_LIMIT", "100/minute"))
async def diagnose(request: Request, request_body: DiagnoseRequest):
    if not ood_detector.is_loaded or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    embedding = np.array(request_body.embedding, dtype=np.float32)

    with IEP2_INFERENCE_DURATION.time():
        # ── Stage 1: OOD Detection (Isolation Forest) ──
        # Important: The Isolation Forest is trained on vibration features only (39-d),
        # while the classifier uses features + metadata (41-d).
        vibration_features = embedding[:39]
        anomaly_score = ood_detector.score(vibration_features)
        OOD_ANOMALY_SCORE.set(float(anomaly_score))
        
        # Use calibrated threshold if available, then env override, then default.
        # Higher = more normal; lower (more negative) = more anomalous.
        threshold = _get_ood_threshold()
        is_ood = anomaly_score < threshold

        if is_ood:
            OOD_REJECTIONS.inc()
            logger.warning(
                f"OOD rejection: score={anomaly_score:.4f}, threshold={threshold}"
            )
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Out-of-Distribution Acoustic Environment",
                    "anomaly_score": float(anomaly_score),
                    "threshold": float(threshold),
                },
            )

        # ── Stage 2: Classification (The Specialist) ──
        prediction = classifier.predict(
            embedding=embedding,
            pipe_material=request_body.pipe_material,
            pressure_bar=request_body.pressure_bar,
        )

        PREDICTION_CONFIDENCE.observe(prediction["confidence"])
        drift_monitor.observe(embedding, prediction["confidence"])

        scada_mismatch, scada_detail = _check_scada_consistency(
            label=prediction["label"],
            confidence=prediction["confidence"],
            pressure_bar=request_body.pressure_bar,
        )
        if scada_mismatch:
            SCADA_MISMATCHES.inc()

    return DiagnoseResponse(
        label=prediction["label"],
        confidence=prediction["confidence"],
        probabilities=prediction["probabilities"],
        anomaly_score=float(anomaly_score),
        is_in_distribution=True,
        scada_mismatch=scada_mismatch,
        scada_mismatch_detail=scada_detail,
    )



@app.post("/calibrate", response_model=CalibrateResponse)
@limiter.limit(os.getenv("OMNI_IEP_RATE_LIMIT", "100/minute"))
async def calibrate(request: Request, request_body: CalibrateRequest):
    """
    Dynamically calibrate OOD detection thresholds.

    Accepts a set of ambient embeddings and adjusts the
    CNN Autoencoder threshold to accommodate the new environment.
    """
    if not ood_detector.is_loaded:
        raise HTTPException(status_code=503, detail="OOD model not loaded yet")

    embeddings = np.array(request_body.ambient_embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Expected a 2-D array of shape (N, n_features), got {embeddings.shape}",
        )

    # Calculate scores for ambient embeddings
    scores = [ood_detector.score(emb) for emb in embeddings]
    new_threshold = calibration_mgr.calibrate(scores)

    logger.info(
        f"Calibrated: {len(embeddings)} samples, "
        f"new threshold={new_threshold:.4f}"
    )

    return CalibrateResponse(
        message="Calibration successful",
        num_samples=len(embeddings),
        new_threshold=float(new_threshold),
        ambient_score_mean=float(np.mean(scores)),
        ambient_score_std=float(np.std(scores)),
    )
