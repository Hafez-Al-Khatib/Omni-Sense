"""
Physics-Informed Vibration Feature Extractor
==============================================
Replaces speech-domain MFCC/chroma features with structural-vibration
diagnostic features appropriate for piezoelectric accelerometer data.

WHY NOT MFCCs:
  MFCCs were designed for speech and music (airborne, microphone recordings).
  Our sensor is a piezoelectric accelerometer measuring structure-borne
  vibration.  The physics are fundamentally different:
    - Accelerometer data has no pitch / formant structure
    - Chroma features (pitch classes) are meaningless for pipe vibration
    - Kurtosis and envelope analysis are the gold standard for mechanical
      fault detection (ISO 13373, bearing diagnostics literature)

Feature vector (N_FEATURES = 100 dimensions):
  Group 1 — Time-domain fault indicators    12
  Group 2 — Hilbert envelope statistics      8
  Group 3 — Octave-band energy              7
  Group 4 — Octave-band kurtosis            7
  Group 5 — Global spectral statistics      7
  Group 6 — Autocorrelation                 8
  Group 7 — Teager-Kaiser energy            4
  Group 8 — Short-time frame statistics    18
  Group 9 — Sub-band temporal modulation    7
  Group 10 — Band-pass signal statistics   16 (4 bands × 4 stats)
  ─────────────────────────────────────────────
  Total                                    94
  (padding zeros to 100)                    6
  ─────────────────────────────────────────────
  N_FEATURES                              100

References:
  - ISO 13373-1: Vibration condition monitoring
  - Randall & Antoni, "Rolling element bearing diagnostics", Mech Sys & Signal Proc 2011
  - Yan et al., "Kurtosis-based wavelet filter", Mech Sys & Signal Proc 2013
"""

import logging

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

logger = logging.getLogger("iep1.features")

# ─── Constants ────────────────────────────────────────────────────────────────

N_FEATURES = 100
_SR = 16000                      # sample rate (Hz)
_FRAME_LEN = int(_SR * 0.25)     # 250 ms frames
_N_FRAMES = 20                   # number of non-overlapping frames in 5 s
_OCTAVE_BANDS_HZ = [             # 7 octave bands (upper edge, Hz)
    125, 250, 500, 1000, 2000, 4000, 8000
]
_AUTOCORR_LAGS = [               # meaningful diagnostic lags (samples)
    10, 50, 100, 200, 500, 1000, 2000, 5000
]
_BANDPASS_RANGES_HZ = [          # 4 diagnostic frequency bands
    (0, 500),
    (500, 2000),
    (2000, 4000),
    (4000, 7900),                 # keep below Nyquist
]


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _safe_kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (excess); returns 0 if degenerate."""
    if len(x) < 4 or np.std(x) < 1e-10:
        return 0.0
    return float(scipy_stats.kurtosis(x, fisher=True, bias=False))


def _safe_skewness(x: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10:
        return 0.0
    return float(scipy_stats.skew(x, bias=False))


def _shannon_entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    """Normalised Shannon entropy of a non-negative array."""
    x = np.abs(x) + eps
    p = x / x.sum()
    return float(-np.sum(p * np.log(p + eps)))


def _bandpass(y: np.ndarray, lo: float, hi: float, fs: int = _SR) -> np.ndarray:
    """2nd-order Butterworth band-pass filter; returns zeros on failure."""
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 1.0 - 1e-4)
    if lo_n >= hi_n:
        return np.zeros_like(y)
    try:
        b, a = scipy_signal.butter(2, [lo_n, hi_n], btype="band")
        return scipy_signal.filtfilt(b, a, y).astype(np.float32)
    except Exception:
        return np.zeros_like(y)


def _stft_power(y: np.ndarray, n_fft: int = 1024, hop: int = 256
                ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (power_spectrum [freq×time], freqs [Hz])."""
    freqs, _, Zxx = scipy_signal.stft(y, fs=_SR, nperseg=n_fft, noverlap=n_fft - hop)
    power = np.abs(Zxx) ** 2
    return power, freqs


# ─── Feature groups ───────────────────────────────────────────────────────────

def _group1_time_domain(y: np.ndarray) -> np.ndarray:
    """
    12 classical vibration fault indicators.
    Kurtosis and crest factor are THE primary ISO 13373 indicators.
    """
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    mean_abs = float(np.mean(np.abs(y))) + 1e-12

    kurtosis = _safe_kurtosis(y)
    crest_factor = peak / (rms + 1e-12)
    skewness = _safe_skewness(y)
    shape_factor = rms / mean_abs                          # ~ 1.11 for sine
    impulse_factor = peak / mean_abs
    variance = float(np.var(y))
    peak_to_peak = float(np.ptp(y))
    zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)
    log_energy = float(np.log(np.sum(y ** 2) + 1e-12))

    dy = np.diff(y)
    diff_rms = float(np.sqrt(np.mean(dy ** 2)))
    diff_kurtosis = _safe_kurtosis(dy)

    return np.array([
        rms, kurtosis, crest_factor, skewness,
        shape_factor, impulse_factor, variance, peak_to_peak,
        zcr, log_energy, diff_rms, diff_kurtosis,
    ], dtype=np.float32)   # 12 features


def _group2_hilbert_envelope(y: np.ndarray) -> np.ndarray:
    """
    8 Hilbert envelope features.
    Envelope analysis is standard practice for bearing / crack diagnostics.
    A crack modulates energy rhythmically; turbulent leaks are more random.
    """
    analytic = scipy_signal.hilbert(y)
    env = np.abs(analytic).astype(np.float32)

    env_rms = float(np.sqrt(np.mean(env ** 2)))
    env_mean = float(np.mean(env))
    env_kurtosis = _safe_kurtosis(env)
    env_crest_factor = float(np.max(env)) / (env_rms + 1e-12)
    env_skewness = _safe_skewness(env)
    env_peak = float(np.max(env))
    env_entropy = _shannon_entropy(env)
    env_zcr = float(np.mean(np.abs(np.diff(np.sign(env - env_mean)))) / 2.0)

    return np.array([
        env_rms, env_mean, env_kurtosis, env_crest_factor,
        env_skewness, env_peak, env_entropy, env_zcr,
    ], dtype=np.float32)   # 8 features


def _group3_octave_band_energy(y: np.ndarray) -> np.ndarray:
    """
    7 normalised octave-band energies.
    Each band has different propagation characteristics through pipe material;
    leak turbulence typically energises higher bands (500 Hz–2 kHz).
    """
    total_energy = np.sum(y ** 2) + 1e-12
    energies = []
    lo = 0.0
    for hi in _OCTAVE_BANDS_HZ:
        band = _bandpass(y, lo + 1.0, hi)
        energies.append(float(np.sum(band ** 2) / total_energy))
        lo = hi
    return np.array(energies, dtype=np.float32)              # 7 features


def _group4_octave_band_kurtosis(y: np.ndarray) -> np.ndarray:
    """
    7 band-limited kurtoses — tells us WHICH frequency band carries the fault.
    A burst-pipe fault shows elevated kurtosis in 500 Hz–2 kHz range.
    """
    kurtoses = []
    lo = 0.0
    for hi in _OCTAVE_BANDS_HZ:
        band = _bandpass(y, lo + 1.0, hi)
        kurtoses.append(_safe_kurtosis(band))
        lo = hi
    return np.array(kurtoses, dtype=np.float32)              # 7 features


def _group5_spectral_statistics(y: np.ndarray) -> np.ndarray:
    """7 global spectral statistics from the STFT power spectrum."""
    power, freqs = _stft_power(y)
    mean_power = power.mean(axis=1) + 1e-12        # (freq,) averaged over time

    total = mean_power.sum()
    cumsum = np.cumsum(mean_power)
    centroid = float(np.sum(freqs * mean_power) / total)
    bandwidth = float(np.sqrt(np.sum(mean_power * (freqs - centroid) ** 2) / total))
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    entropy = _shannon_entropy(mean_power)
    flatness = float(
        np.exp(np.mean(np.log(mean_power + 1e-12))) / (np.mean(mean_power) + 1e-12)
    )
    dominant_freq = float(freqs[np.argmax(mean_power)])
    spectral_kurtosis = _safe_kurtosis(mean_power)

    return np.array([
        centroid, bandwidth, rolloff, entropy,
        flatness, dominant_freq, spectral_kurtosis,
    ], dtype=np.float32)                                     # 7 features


def _group6_autocorrelation(y: np.ndarray) -> np.ndarray:
    """
    8 normalised autocorrelation coefficients at diagnostic lags.
    Periodic pump signals → high autocorr at the pump period.
    Turbulent leak noise → near-zero autocorr at all lags.
    """
    y_norm = y / (np.std(y) + 1e-12)
    n = len(y_norm)
    coeffs = []
    for lag in _AUTOCORR_LAGS:
        if lag >= n:
            coeffs.append(0.0)
        else:
            coeffs.append(float(np.mean(y_norm[:n - lag] * y_norm[lag:])))
    return np.array(coeffs, dtype=np.float32)                # 8 features


def _group7_teager_kaiser(y: np.ndarray) -> np.ndarray:
    """
    4 Teager-Kaiser energy operator statistics.
    TK energy: Ψ[x(n)] = x(n)² − x(n−1)·x(n+1)
    More sensitive to instantaneous frequency/amplitude changes than RMS;
    highlights the nonlinear energy bursts associated with leak turbulence.
    """
    tk = y[1:-1] ** 2 - y[:-2] * y[2:]
    tk_rms = float(np.sqrt(np.mean(tk ** 2)))
    tk_mean = float(np.mean(np.abs(tk)))
    tk_kurtosis = _safe_kurtosis(tk)
    tk_crest = float(np.max(np.abs(tk))) / (tk_rms + 1e-12)
    return np.array([tk_rms, tk_mean, tk_kurtosis, tk_crest], dtype=np.float32)  # 4


def _group8_frame_statistics(y: np.ndarray) -> np.ndarray:
    """
    18 short-time frame-level statistics.
    Splits the 5 s signal into 20 non-overlapping 250 ms frames and
    captures temporal dynamics: how stable is the vibration?
    Steady leaks → low temporal modulation; impulsive events → high modulation.
    """
    frames = np.array_split(y, _N_FRAMES)
    rms_per = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    kurt_per = np.array([_safe_kurtosis(f) for f in frames])
    zcr_per = np.array([
        float(np.mean(np.abs(np.diff(np.sign(f)))) / 2.0) for f in frames
    ])

    # Spectral centroid per frame
    power, freqs = _stft_power(y)
    n_fft, n_time = power.shape
    frames_per_bin = max(1, n_time // _N_FRAMES)
    centroids_per = []
    for i in range(_N_FRAMES):
        s = i * frames_per_bin
        e = min((i + 1) * frames_per_bin, n_time)
        p = power[:, s:e].mean(axis=1) + 1e-12
        centroids_per.append(float(np.sum(freqs * p) / p.sum()))
    centroids_per = np.array(centroids_per)

    # Spectral flux (frame-to-frame spectral change)
    flux_per = np.array([
        float(np.sqrt(np.sum((power[:, i + 1] - power[:, i]) ** 2)))
        for i in range(n_time - 1)
    ] + [0.0])

    rms_nonzero = rms_per.mean() + 1e-12
    temporal_mod_idx = float(rms_per.std() / rms_nonzero)
    energy_entropy = _shannon_entropy(rms_per ** 2)
    onset_density = float(np.mean(rms_per > rms_per.mean()))
    peak_to_rms_temporal = float(rms_per.max() / rms_nonzero)
    stationarity_idx = float(1.0 / (1.0 + temporal_mod_idx))

    flux_arr = np.array([flux_per.mean(), flux_per.std()])

    return np.array([
        rms_per.mean(), rms_per.std(), rms_per.max(), _shannon_entropy(rms_per),  # 4
        kurt_per.mean(), kurt_per.std(), kurt_per.max(),                          # 3
        centroids_per.mean(), centroids_per.std(),                                # 2
        zcr_per.mean(), zcr_per.std(),                                            # 2
        flux_arr[0], flux_arr[1],                                                 # 2
        temporal_mod_idx,                                                         # 1
        energy_entropy,                                                           # 1
        onset_density,                                                            # 1
        peak_to_rms_temporal,                                                     # 1
        stationarity_idx,                                                         # 1
    ], dtype=np.float32)                                      # 18 features


def _group9_subband_temporal_modulation(y: np.ndarray) -> np.ndarray:
    """
    7 sub-band temporal modulation indices (one per octave band).
    Splits each band-filtered signal into frames and measures how much
    the energy fluctuates over time.  Leak turbulence is highly variable;
    normal steady-state flow is relatively stationary.
    """
    mods = []
    lo = 0.0
    for hi in _OCTAVE_BANDS_HZ:
        band = _bandpass(y, lo + 1.0, hi)
        frames = np.array_split(band, _N_FRAMES)
        rms_frames = np.array([np.sqrt(np.mean(f ** 2)) + 1e-12 for f in frames])
        mods.append(float(rms_frames.std() / rms_frames.mean()))
        lo = hi
    return np.array(mods, dtype=np.float32)                  # 7 features


def _group10_bandpass_statistics(y: np.ndarray) -> np.ndarray:
    """
    16 band-pass signal statistics (4 diagnostic bands × 4 stats).
    Each band captures a different physical mechanism:
      0–500 Hz   : low-frequency pipe resonance, structural modes
      500–2000 Hz: primary leak turbulence band (ISO 13373 key band)
      2–4 kHz    : high-frequency cavitation and crack signatures
      4–8 kHz    : surface micro-crack and orifice whistle harmonics
    """
    feats = []
    for lo, hi in _BANDPASS_RANGES_HZ:
        band = _bandpass(y, max(lo, 1.0), hi)
        rms = float(np.sqrt(np.mean(band ** 2)))
        kurt = _safe_kurtosis(band)
        crest = float(np.max(np.abs(band))) / (rms + 1e-12)
        skew = _safe_skewness(band)
        feats.extend([rms, kurt, crest, skew])
    return np.array(feats, dtype=np.float32)                  # 16 features


# ─── Main extractor class ─────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Stateless physics-informed vibration feature extractor.
    No model file needed — all features are computed analytically.

    Interface-compatible with the old YAMNetService so call sites
    only need to update imports.
    """

    @property
    def is_loaded(self) -> bool:
        return True   # always ready; no model weights to load

    def load(self) -> None:
        """No-op — kept for API compatibility with startup hook."""
        pass

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract N_FEATURES-d physics-informed feature vector.

        Args:
            waveform: 1D float32 array at 16 kHz (80 000 samples for 5 s)

        Returns:
            float32 array of shape (N_FEATURES,)
        """
        y = waveform.astype(np.float32)

        # Normalise to [-1, 1] to make features scale-invariant
        peak = np.max(np.abs(y))
        if peak > 1e-10:
            y = y / peak

        parts = [
            _group1_time_domain(y),           # 12
            _group2_hilbert_envelope(y),       #  8
            _group3_octave_band_energy(y),     #  7
            _group4_octave_band_kurtosis(y),   #  7
            _group5_spectral_statistics(y),    #  7
            _group6_autocorrelation(y),        #  8
            _group7_teager_kaiser(y),          #  4
            _group8_frame_statistics(y),       # 18
            _group9_subband_temporal_modulation(y),  # 7
            _group10_bandpass_statistics(y),   # 16
        ]

        vec = np.concatenate(parts).astype(np.float32)

        # Pad or truncate to exactly N_FEATURES (safety net)
        if len(vec) < N_FEATURES:
            vec = np.concatenate([vec, np.zeros(N_FEATURES - len(vec), dtype=np.float32)])
        else:
            vec = vec[:N_FEATURES]

        # Replace any NaN/Inf (e.g. near-silent signals) with 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec


# Singleton used by IEP1 main
feature_extractor = FeatureExtractor()
