"""
Signal Quality Assurance
=========================
Pre-inference checks on raw audio to reject obviously bad inputs:
  - Dead Sensor: absolute silence (RMS below threshold)
  - Broken Mic: pure clipping (peak at max amplitude)
"""

import numpy as np


def check_signal_quality(
    audio: np.ndarray,
    silence_threshold: float = 0.001,
    clipping_threshold: float = 0.99,
) -> dict:
    """
    Validate audio signal quality before inference.

    Args:
        audio: 1D float32 numpy array (raw audio samples)
        silence_threshold: RMS below this = dead sensor
        clipping_threshold: Percentage of clipped samples above this = broken mic

    Returns:
        dict with keys:
            - is_valid (bool)
            - rms (float)
            - peak (float)
            - clipping_ratio (float)
            - error (str | None)
    """
    if len(audio) == 0:
        return {
            "is_valid": False,
            "rms": 0.0,
            "peak": 0.0,
            "clipping_ratio": 0.0,
            "error": "Empty audio signal",
        }

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Count samples at or near maximum amplitude
    clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
    clipping_ratio = float(clipped_samples / len(audio))

    # ── Dead Sensor Check ──
    if rms < silence_threshold:
        return {
            "is_valid": False,
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "error": (
                f"Dead Sensor detected: RMS={rms:.6f} is below threshold "
                f"{silence_threshold}. The microphone may not be capturing audio."
            ),
        }

    # ── Broken Mic Check ──
    # If more than 30% of samples are clipped, the mic is likely broken
    if clipping_ratio > 0.30:
        return {
            "is_valid": False,
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
            "error": (
                f"Broken Microphone detected: {clipping_ratio:.1%} of samples are "
                f"clipped (threshold: 30%). The audio is saturated/distorted."
            ),
        }

    return {
        "is_valid": True,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "error": None,
    }
