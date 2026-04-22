"""
IEP 2 — Diagnostic & Safety Engine
=====================================
Two-stage inference pipeline:
  Stage 1: Autoencoder (OOD detection / safety watchdog)
  Stage 2: XGBoost (leak classification with calibrated probabilities)

Accepts 1024-d YAMNet embeddings + metadata, returns diagnosis.
"""

import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.calibration import CalibrationManager
from app.classifier import LeakClassifier
from app.drift_monitor import drift_monitor
from app.autoencoder_ood_detector import AutoencoderOODDetector
from app.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iep2")

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
OOD_SCORE = Gauge(
    "ood_isolation_forest_score",
    "Latest Isolation Forest anomaly score (input drift proxy)",
)

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

# Physics-based pressure window (bar) per fault class.
# A confident prediction outside this window signals a potential
# acoustic spoof or sensor/metadata error.
_PRESSURE_PHYSICS: dict[str, tuple[float, float]] = {
    "No_Leak":               (0.3, 20.0),
    "Orifice_Leak":          (0.5,  8.0),
    "Gasket_Leak":           (0.5,  8.0),
    "Longitudinal_Crack":    (0.5,  5.5),  # cracks cause rapid pressure drop
    "Circumferential_Crack": (0.5,  5.5),  # catastrophic; very high stable P = suspicious
    "Normal_Operation":      (0.0, 20.0),
}
_SCADA_CONFIDENCE_GATE = 0.85  # only flag when model is this confident


def _check_scada_consistency(
    label: str, confidence: float, pressure_bar: float
) -> tuple[bool, str | None]:
    """
    Compare the classified fault type against the operator-supplied pressure.

    Returns (mismatch: bool, detail: str | None).
    A mismatch at high confidence indicates either a sensor/metadata error
    or active acoustic spoofing (playing a leak recording near the sensor
    while pipe pressure is normal/high).
    """
    if confidence < _SCADA_CONFIDENCE_GATE:
        return False, None

    p_min, p_max = _PRESSURE_PHYSICS.get(label, (0.0, 20.0))

    if pressure_bar < p_min:
        return True, (
            f"Pressure {pressure_bar:.1f} bar is below the physical minimum "
            f"({p_min} bar) for '{label}'. Possible dead pipe, closed valve, "
            f"or sensor detachment."
        )

    if pressure_bar > p_max:
        return True, (
            f"Pressure {pressure_bar:.1f} bar is unexpectedly stable for "
            f"'{label}' at {confidence:.0%} confidence (max expected: {p_max} bar). "
            f"Real {label.replace('_', ' ').lower()} events cause measurable pressure "
            f"drop. Possible acoustic spoofing or metadata error."
        )

    return False, None

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Omni-Sense IEP2 — Diagnostic & Safety Engine",
    description="OOD-aware acoustic classification: Isolation Forest + XGBoost.",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)

# Service singletons
ood_detector = AutoencoderOODDetector()
classifier = LeakClassifier()
calibration_mgr = CalibrationManager()


_CENTROID_PATH = Path("models/centroid.npy")


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    logger.info("Loading Autoencoder OOD model...")
    ood_detector.load()
    logger.info("Loading XGBoost model...")
    classifier.load()
    logger.info("All models loaded.")

    # Load reference centroid for drift monitoring (written by train_models.py)
    if _CENTROID_PATH.exists():
        centroid = np.load(_CENTROID_PATH)
        drift_monitor.set_reference_centroid(centroid)
        logger.info(f"Drift monitor: reference centroid loaded from {_CENTROID_PATH}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="healthy",
        ood_model_loaded=ood_detector.is_loaded,
        classifier_loaded=classifier.is_loaded,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Two-stage diagnostic pipeline.

    Stage 1: OOD check via Autoencoder
    Stage 2: Leak classification via XGBoost (if in-distribution)
    """
    if not ood_detector.is_loaded or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    embedding = np.array(request.embedding, dtype=np.float32)


    with IEP2_INFERENCE_DURATION.time():
        # ── Stage 1: OOD Detection (The Watchdog) ──
        threshold = calibration_mgr.get_threshold()
        anomaly_score = ood_detector.score(embedding)
        is_ood = ood_detector.is_anomalous(embedding, threshold_override=threshold)

        OOD_SCORE.set(float(anomaly_score))

        if is_ood:
            OOD_REJECTIONS.inc()
            logger.warning(
                f"OOD rejection: score={anomaly_score:.4f}, threshold={threshold}"
            )
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Out-of-Distribution Acoustic Environment",
                    "detail": (
                        "The acoustic signature does not match any known environment. "
                        "This may indicate a novel noise source or sensor malfunction. "
                        "Use POST /calibrate to adapt to a new environment."
                    ),
                    "anomaly_score": float(anomaly_score),
                    "threshold": float(threshold),
                },
            )

        # ── Stage 2: Classification (The Specialist) ──
        prediction = classifier.predict(
            embedding=embedding,
            pipe_material=request.pipe_material,
            pressure_bar=request.pressure_bar,
        )

        PREDICTION_CONFIDENCE.observe(prediction["confidence"])
        drift_monitor.observe(embedding, prediction["confidence"])

        scada_mismatch, scada_detail = _check_scada_consistency(
            label=prediction["label"],
            confidence=prediction["confidence"],
            pressure_bar=request.pressure_bar,
        )
        if scada_mismatch:
            SCADA_MISMATCHES.inc()
            logger.warning(
                f"SCADA mismatch: label={prediction['label']} "
                f"confidence={prediction['confidence']:.2f} "
                f"pressure={request.pressure_bar} bar — {scada_detail}"
            )

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
async def calibrate(request: CalibrateRequest):
    """
    Dynamically calibrate OOD detection thresholds.

    Accepts a set of ambient embeddings and adjusts the
    Isolation Forest threshold to accommodate the new environment.
    """
    if not ood_detector.is_loaded:
        raise HTTPException(status_code=503, detail="OOD model not loaded yet")

    embeddings = np.array(request.ambient_embeddings, dtype=np.float32)

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
