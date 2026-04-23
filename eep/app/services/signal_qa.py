"""
Signal Quality Assurance
=========================
Pre-inference checks on raw audio to reject obviously bad inputs:
  - Dead Sensor  : absolute silence (RMS below threshold)
  - Broken Sensor: pure clipping (peak at max amplitude)

hardware_status values
----------------------
``OK``                 — signal is valid and within expected bounds.
``SENSOR_MALFUNCTION`` — RMS is flat/near-zero; sensor is not transmitting
                         vibration data.  Central Service should alert the
                         operator that the specific hardware unit is offline.
``SIGNAL_DEGRADED``    — excessive clipping ratio; ADC saturation or loose
                         mounting.  Data may be partial — treat with caution.

Consumers must check ``hardware_status`` independently of ``is_valid`` so the
ops console can display *why* a sensor is not contributing to detections,
rather than silently returning a false-negative "No Leak".
"""
from __future__ import annotations

import numpy as np

# Recognised hardware status values (matches the omni common schema)
HW_OK                = "OK"
HW_SENSOR_MALFUNCTION = "SENSOR_MALFUNCTION"
HW_SIGNAL_DEGRADED   = "SIGNAL_DEGRADED"


def check_signal_quality(
    audio: np.ndarray,
    silence_threshold: float = 0.001,
    clipping_threshold: float = 0.99,
) -> dict:
    """Validate audio signal quality before inference.

    Args:
        audio: 1D float32 numpy array (raw audio samples, normalised to ±1).
        silence_threshold: RMS below this → ``SENSOR_MALFUNCTION``.
        clipping_threshold: Fraction of clipped samples above this →
            ``SIGNAL_DEGRADED`` when > 30 %.

    Returns:
        dict with keys:
            is_valid        (bool)
            hardware_status (str)  — "OK" | "SENSOR_MALFUNCTION" | "SIGNAL_DEGRADED"
            rms             (float)
            peak            (float)
            clipping_ratio  (float)
            error           (str | None) — human-readable reason if not valid
    """
    if len(audio) == 0:
        return {
            "is_valid": False,
            "hardware_status": HW_SENSOR_MALFUNCTION,
            "rms": 0.0,
            "peak": 0.0,
            "clipping_ratio": 0.0,
            "error": "Empty audio signal — sensor produced no data.",
        }

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Count samples at or near maximum amplitude
    clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
    clipping_ratio = float(clipped_samples / len(audio))

    # ── Dead Sensor / Sensor Malfunction ──────────────────────────────────────
    if rms < silence_threshold:
        return {
            "is_valid": False,
            "hardware_status": HW_SENSOR_MALFUNCTION,
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "error": (
                f"Sensor malfunction: RMS={rms:.6f} is below the silence threshold "
                f"{silence_threshold}. The sensor is not capturing vibration data."
            ),
        }

    # ── Signal Degraded (clipping / ADC saturation) ───────────────────────────
    # If more than 30 % of samples are clipped, the signal is unreliable.
    if clipping_ratio > 0.30:
        return {
            "is_valid": False,
            "hardware_status": HW_SIGNAL_DEGRADED,
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "error": (
                f"Signal degraded: {clipping_ratio:.1%} of samples are clipped "
                f"(threshold: 30 %). ADC saturation or loose sensor mounting."
            ),
        }

    return {
        "is_valid": True,
        "hardware_status": HW_OK,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "error": None,
    }
