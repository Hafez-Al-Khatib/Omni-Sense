"""
IEP 2 — Diagnostic & Safety Engine
=====================================
Two-stage inference pipeline:
  Stage 1: Isolation Forest (OOD detection / safety watchdog)
  Stage 2: XGBoost (leak classification with calibrated probabilities)

Accepts 1024-d YAMNet embeddings + metadata, returns diagnosis.
"""

import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Histogram, Gauge, Counter
from prometheus_fastapi_instrumentator import Instrumentator

from app.ood_detector import OODDetector
from app.classifier import LeakClassifier
from app.calibration import CalibrationManager
from app.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    CalibrateRequest,
    CalibrateResponse,
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

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Omni-Sense IEP2 — Diagnostic & Safety Engine",
    description="OOD-aware acoustic classification: Isolation Forest + XGBoost.",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)

# Service singletons
ood_detector = OODDetector()
classifier = LeakClassifier()
calibration_mgr = CalibrationManager()


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    logger.info("Loading Isolation Forest model...")
    ood_detector.load()
    logger.info("Loading XGBoost model...")
    classifier.load()
    logger.info("All models loaded.")


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

    Stage 1: OOD check via Isolation Forest
    Stage 2: Leak classification via XGBoost (if in-distribution)
    """
    if not ood_detector.is_loaded or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    embedding = np.array(request.embedding, dtype=np.float32)

    if len(embedding) != 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 1024-d embedding, got {len(embedding)}-d",
        )

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

    return DiagnoseResponse(
        label=prediction["label"],
        confidence=prediction["confidence"],
        probabilities=prediction["probabilities"],
        anomaly_score=float(anomaly_score),
        is_in_distribution=True,
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

    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Expected (N, 1024) embeddings, got shape {embeddings.shape}",
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
