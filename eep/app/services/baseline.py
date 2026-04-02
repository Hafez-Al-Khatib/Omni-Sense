"""
Amplitude-Threshold Baseline Comparator
=========================================
Implements the industry-standard pre-AI baseline for leak detection:
RMS amplitude thresholding (equivalent to a fixed dB trigger level).

This is included in every diagnose response so the demo can show
side-by-side how the baseline fails on ambient noise (trucks, generators)
while the AI pipeline correctly rejects or classifies.

Threshold rationale (Phase 1 engineering tradeoff):
  A typical field deployment uses ~80 dB SPL as a trigger.
  For normalized floating-point audio (peak = 1.0 ≈ 0 dBFS), an
  aggressive field threshold corresponds to RMS ≈ 0.05 — sensitive
  enough to catch genuine leaks but also triggered by passing trucks
  and generator harmonics in Lebanese urban environments.
"""

import numpy as np

# RMS level that a standard amplitude-threshold system would flag as "leak".
# Set deliberately sensitive to demonstrate the false-positive problem.
AMPLITUDE_THRESHOLD_RMS = 0.05


def run_baseline(audio: np.ndarray) -> dict:
    """
    Apply amplitude-threshold leak detection to a raw audio array.

    Args:
        audio: 1D float32 numpy array of normalized samples.

    Returns:
        dict with:
            baseline_decision  : "leak_detected" | "no_leak_detected"
            baseline_rms       : RMS amplitude of the signal
            baseline_threshold : The RMS threshold used
            baseline_method    : Human-readable method name
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    detected = rms > AMPLITUDE_THRESHOLD_RMS

    return {
        "baseline_decision":  "leak_detected" if detected else "no_leak_detected",
        "baseline_rms":       round(rms, 6),
        "baseline_threshold": AMPLITUDE_THRESHOLD_RMS,
        "baseline_method":    "RMS Amplitude Threshold (industry baseline)",
    }
