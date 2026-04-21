"""
Omni-Sense Central Configuration
=================================
Central registry for magic numbers, thresholds, and physics constants.
Every threshold here includes a 'Rationale' documenting its calibration.
"""

from typing import Final

# ─── OOD Detection (Robustness) ──────────────────────────────────────────

# Isolation Forest anomaly score threshold.
# Rationale: Calibrated 2026-04-12 on golden_v1. Keeps FPR <= 3% on Normal_Operation.
# Lower = more aggressive rejection of novel environments.
OOD_IF_THRESHOLD: Final[float] = 0.37

# CNN Autoencoder reconstruction error (MSE) threshold.
# Rationale: Calibrated 2026-04-15. Detects non-pipe acoustic signatures 
# (e.g. human speech, heavy traffic) with 94% recall.
OOD_AE_THRESHOLD: Final[float] = 0.045


# ─── Detection & Dispatch (Business Logic) ───────────────────────────────

# Minimum fused probability to trigger a maintenance ticket.
# Rationale: Calibrated to balance technician 'alert fatigue' vs water loss.
# Hits 91% recall on LeakDB test split.
LEAK_CONFIDENCE_THRESHOLD: Final[float] = 0.85

# RMS Amplitude gate (The 'Baseline' detector).
# Rationale: Industry standard pre-filter. Equivalent to ~80dB SPL in field.
# Prevents expensive ML inference on dead sensors or absolute silence.
BASELINE_RMS_THRESHOLD: Final[float] = 0.005


# ─── Fusion Weights (Ensemble) ──────────────────────────────────────────

# Weights for the 5-head fusion in EEP orchestrator.
# Rationale: XGBoost/CNN are primary; RF provides variance-stabilization.
# IF (Isolation Forest) is included in fusion as a 'negative' signal.
FUSION_WEIGHTS: Final[dict[str, float]] = {
    "xgb": 0.45,
    "rf":  0.25,
    "cnn": 0.30,
    "if":  0.00,  # IF used for OOD gating, not probability boost
}

# ─── Physics Constants ───────────────────────────────────────────────────

# Wave speeds in m/s for Time-Difference-of-Arrival (TDOA)
# Rationale: Engineering handbooks for water-filled pressure pipes.
WAVE_SPEEDS: Final[dict[str, float]] = {
    "PVC":       400.0,
    "Steel":    1350.0,
    "Cast_Iron": 1300.0,
}
