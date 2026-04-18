"""TDOA (Time Difference of Arrival) pipe-graph leak localization.

Background
----------
Water pipe leaks generate structure-borne acoustic stress waves that propagate
along the pipe at a speed determined by the pipe material and diameter:

    Pipe material   Wave speed (m/s)
    ─────────────── ────────────────
    Cast iron       1200 – 1600
    Steel           3000 – 5100
    PVC             300  – 500
    HDPE/PE         200  – 400

Given two sensors A and B mounted on the same pipe, separated by a known
pipe-path distance L (metres), the cross-correlation of their simultaneous
vibration signals yields the propagation delay Δt (seconds):

    Δt = t_B - t_A          (positive if wave arrives at A first)

The leak position x measured from sensor A along the pipe is then:

    x = (L - v · Δt) / 2

where v is the wave speed in the pipe material.

Constraints:
    0 ≤ x ≤ L    — if outside, the leak is not between A and B

When three or more sensors observe the same event, we solve two TDOA
equations (A–B and A–C) and take the mean, optionally weighted by peak
cross-correlation magnitude.

Why cross-correlation?
----------------------
The raw vibration frames from both sensors are contaminated by background
pump noise and are not directly comparable.  Cross-correlation in the
frequency domain (via FFT) finds the time lag that maximises their
similarity — this is the standard technique used by every commercial
correlator (Gutermann, Primayer, Orca).  We use GCC-PHAT (Phase Transform)
weighting which suppresses harmonic pump noise and sharpens the delay peak.

References
----------
- Hunaidi, O. & Chu, W. (1999). Acoustical characteristics of leak signals
  in plastic water distribution pipes.  Applied Acoustics, 58(3), 235–254.
- Gao, Y. et al. (2004). The development of a new acoustic technique for
  detecting leaks in buried plastic water supply pipes.  Mech. Syst. Signal
  Process., 18(6), 1329–1344.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("tdoa")

# ─── Wave speed table (m/s) ──────────────────────────────────────────────────

WAVE_SPEED_MS: dict[str, float] = {
    "Cast_Iron": 1_400.0,
    "Steel":     4_500.0,
    "PVC":         400.0,
    "HDPE":        350.0,
    "PE":          350.0,
    # Fallback for unknown materials — conservative midpoint
    "Unknown":   1_200.0,
}

_DEFAULT_MATERIAL = "Cast_Iron"


def wave_speed(pipe_material: str) -> float:
    """Return acoustic wave propagation speed for the given pipe material (m/s)."""
    return WAVE_SPEED_MS.get(pipe_material, WAVE_SPEED_MS[_DEFAULT_MATERIAL])


# ─── GCC-PHAT cross-correlation ───────────────────────────────────────────────

def gcc_phat(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    sr: int,
    max_delay_s: float = 0.1,
) -> tuple[float, float]:
    """GCC-PHAT cross-correlation estimator.

    Returns
    -------
    delay_s : float
        Time delay of sig_b relative to sig_a in seconds.
        Positive  → wave arrived at A before B  (leak is closer to A).
        Negative  → wave arrived at B before A  (leak is closer to B).
    peak_val : float
        Normalised peak magnitude [0, 1].  Use as confidence weight.

    Algorithm
    ---------
    1. Zero-pad both signals to the next power of two for efficiency.
    2. Compute FFTs.
    3. Cross-power spectrum: G = X_A * conj(X_B).
    4. PHAT weighting:       G_phat = G / |G|   (makes it robust to coloured noise).
    5. IFFT → GCC-PHAT function; peak lag = delay estimate.
    """
    n = len(sig_a)
    if len(sig_b) != n:
        # Align to shorter signal
        n = min(n, len(sig_b))
        sig_a, sig_b = sig_a[:n], sig_b[:n]

    # Zero-pad to next power of 2 for fast FFT
    nfft = 1 << (n - 1).bit_length()

    Fa = np.fft.rfft(sig_a, n=nfft)
    Fb = np.fft.rfft(sig_b, n=nfft)

    # Fb * conj(Fa) gives IFFT peak at +lag when B is delayed relative to A,
    # i.e. delay_s > 0  ⟺  wave arrived at A first — matching the docstring.
    G = Fb * np.conj(Fa)

    # PHAT weighting — normalise by magnitude, preserving only phase
    denom = np.abs(G) + 1e-10
    G_phat = G / denom

    gcc = np.fft.irfft(G_phat, n=nfft)

    # Limit search to ± max_delay_s
    max_lag = int(max_delay_s * sr)
    max_lag = min(max_lag, nfft // 2 - 1)

    # gcc is circular: positive lags are at start, negative lags at end
    # Build a lag range [-max_lag, +max_lag]
    lags    = np.concatenate([np.arange(0, max_lag + 1),
                               np.arange(nfft - max_lag, nfft)])
    values  = gcc[lags]

    peak_idx = int(np.argmax(np.abs(values)))
    raw_lag  = lags[peak_idx]

    # Convert circular index to signed lag
    if raw_lag > nfft // 2:
        raw_lag -= nfft
    delay_s = raw_lag / sr

    # Normalise peak to [0, 1]
    peak_val = float(np.abs(values[peak_idx]) / (np.max(np.abs(gcc)) + 1e-10))

    return delay_s, peak_val


# ─── Pipe segment descriptor ──────────────────────────────────────────────────

@dataclass
class PipeSegment:
    """A straight pipe segment between two sensors.

    Attributes
    ----------
    segment_id   : str     Segment identifier (matches PIPE_NETWORK entry).
    sensor_a_id  : str     Omni sensor ID at one end.
    sensor_b_id  : str     Omni sensor ID at the other end.
    pipe_length_m: float   Pipe path distance between the two sensors (m).
    pipe_material: str     Material name (key into WAVE_SPEED_MS).
    lat_a, lon_a : float   Geographic position of sensor A.
    lat_b, lon_b : float   Geographic position of sensor B.
    """
    segment_id:    str
    sensor_a_id:   str
    sensor_b_id:   str
    pipe_length_m: float
    pipe_material: str   = _DEFAULT_MATERIAL
    lat_a:         float = 0.0
    lon_a:         float = 0.0
    lat_b:         float = 0.0
    lon_b:         float = 0.0

    def position_from_a(self, delay_s: float) -> Optional[float]:
        """Return leak distance from sensor A (metres) given the TDOA delay.

        Returns None if the estimate falls outside [0, pipe_length_m].
        """
        v = wave_speed(self.pipe_material)
        x = (self.pipe_length_m - v * delay_s) / 2.0
        if not (0.0 <= x <= self.pipe_length_m):
            return None
        return x

    def to_latlon(self, dist_from_a: float) -> tuple[float, float]:
        """Linear interpolation from sensor A toward sensor B along the pipe.

        For real deployments, replace with PostGIS ST_LineInterpolatePoint.
        """
        ratio = dist_from_a / max(self.pipe_length_m, 1.0)
        lat   = self.lat_a + ratio * (self.lat_b - self.lat_a)
        lon   = self.lon_a + ratio * (self.lon_b - self.lon_a)
        return lat, lon


# ─── TDOA result ──────────────────────────────────────────────────────────────

@dataclass
class TDOAResult:
    """Result of a TDOA localization for one sensor pair."""
    segment_id:       str
    sensor_a_id:      str
    sensor_b_id:      str
    delay_s:          float          # Δt: sig_b arrival - sig_a arrival
    peak_correlation: float          # GCC-PHAT peak magnitude [0, 1]
    dist_from_a_m:    Optional[float]  # leak position from sensor A (None if out-of-range)
    lat:              Optional[float] = None
    lon:              Optional[float] = None
    pipe_material:    str  = _DEFAULT_MATERIAL
    wave_speed_ms:    float = 1_400.0
    uncertainty_m:    float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.dist_from_a_m is not None

    @property
    def confidence(self) -> float:
        """Heuristic confidence [0, 1] based on GCC-PHAT peak and delay validity."""
        if not self.is_valid:
            return 0.0
        return float(np.clip(self.peak_correlation, 0.0, 1.0))


# ─── Core localization function ───────────────────────────────────────────────

def localize(
    pcm_a: np.ndarray,
    pcm_b: np.ndarray,
    sr: int,
    segment: PipeSegment,
) -> TDOAResult:
    """Estimate leak position on a pipe segment from two simultaneous recordings.

    Parameters
    ----------
    pcm_a, pcm_b : np.ndarray
        Simultaneously captured vibration frames from sensor A and B.
        Must be float32, same length, same sample rate.
    sr : int
        Sample rate in Hz (e.g. 3200 for ADXL345 at full ODR).
    segment : PipeSegment
        Pipe geometry descriptor.

    Returns
    -------
    TDOAResult
        Contains delay estimate, GCC-PHAT confidence, and lat/lon if valid.
    """
    v = wave_speed(segment.pipe_material)

    # Max possible delay = pipe_length / wave_speed (a tiny margin added)
    max_delay = segment.pipe_length_m / v * 1.2

    delay_s, peak = gcc_phat(pcm_a, pcm_b, sr=sr, max_delay_s=max_delay)

    dist_from_a = segment.position_from_a(delay_s)

    lat = lon = None
    if dist_from_a is not None:
        lat, lon = segment.to_latlon(dist_from_a)

    # Uncertainty grows with sample period (half a sample precision)
    sample_period = 1.0 / sr
    uncertainty_m = (v * sample_period / 2.0)   # propagated from ±0.5 sample timing error

    result = TDOAResult(
        segment_id=segment.segment_id,
        sensor_a_id=segment.sensor_a_id,
        sensor_b_id=segment.sensor_b_id,
        delay_s=delay_s,
        peak_correlation=peak,
        dist_from_a_m=dist_from_a,
        lat=lat,
        lon=lon,
        pipe_material=segment.pipe_material,
        wave_speed_ms=v,
        uncertainty_m=uncertainty_m,
    )

    log.debug(
        "TDOA %s A=%s B=%s: delay=%.4fs x_from_A=%.1fm unc=%.2fm peak=%.3f v=%.0fm/s",
        segment.segment_id,
        segment.sensor_a_id,
        segment.sensor_b_id,
        delay_s,
        dist_from_a if dist_from_a is not None else float("nan"),
        uncertainty_m,
        peak,
        v,
    )

    return result


def fuse_results(results: list[TDOAResult]) -> Optional[TDOAResult]:
    """Combine multiple TDOA results (from ≥ 2 sensor pairs) into one estimate.

    Uses GCC-PHAT peak magnitude as the fusion weight.
    Returns None if no valid result exists.
    """
    valid = [r for r in results if r.is_valid and r.lat is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    weights = [r.confidence for r in valid]
    w_sum   = sum(weights)
    if w_sum < 1e-9:
        return valid[0]

    lat = sum(r.lat * w for r, w in zip(valid, weights)) / w_sum  # type: ignore[operator]
    lon = sum(r.lon * w for r, w in zip(valid, weights)) / w_sum  # type: ignore[operator]
    unc = max(r.uncertainty_m for r in valid)   # conservative

    # Borrow attributes from best result
    best = max(valid, key=lambda r: r.confidence)
    best.lat = lat
    best.lon = lon
    best.uncertainty_m = unc
    return best
