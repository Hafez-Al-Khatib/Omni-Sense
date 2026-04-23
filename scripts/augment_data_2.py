import warnings

import numpy as np
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
TARGET_SR = 16_000
WINDOW_S = 5.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_S)

# Butterworth LPF parameters per pipe material.
# Rationale: denser/stiffer materials transmit higher frequencies;
# PVC absorbs high-frequency energy more than steel.
PIPE_PROFILES: dict[str, dict] = {
    "PVC" : {"cutoff_hz": 2000, "order": 4},
    "Steel" : {"cutoff_hz": 6000, "order": 4},
    "Cast_Iron" : {"cutoff_hz": 3500, "order": 3},
}

# Condition label found in filename -> simulated operating pressure.
# Based on the LeakDB experimental setup documentation.
CONDITION_PRESSURE: dict[str, float] = {
    "0.18_LPS": 2.0,   # low-flow, lower line pressure
    "0.47_LPS": 4.5,   # higher-flow, higher line pressure
    "ND":        3.0,   # normal discharge (steady-state)
    "Transient": 6.0,   # pressure surge / water hammer
    "Unknown":   3.0,
}

# Normalised fault-class names (replace hyphens, keep underscores).
# "No-leak" folder → "No_Leak" label to match Python identifier conventions.
FAULT_CLASS_NORMALISE: dict[str, str] = {
    "No-leak": "No_Leak",
}

# Audio Utilities

def apply_pipe_lpf(audio: np.ndarray, material: str) -> np.ndarray:
    """Low pass filter to simulate pipe material's effect on vibration propagation."""
    cfg = PIPE_PROFILES[material]
    nyquist = TARGET_SR / 2.0
    norm_cutoff = min(cfg["cutoff_hz"] / nyquist, 0.99)
    sos = butter(cfg["order"], norm_cutoff, btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)

def peak_normalise(audio: np.ndarray, headroom: float = 0.99) -> np.ndarray:
    """Normalise to peak amplitude without hard-clipping."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return (audio * (headroom / peak)).astype(np.float32)

def add_awgn(audio: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at amplitude sigma (models ADC noise floor)."""
    noise = rng.normal(0.0, sigma, size=len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

