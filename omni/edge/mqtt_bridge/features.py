"""
DSP Feature Extractor — pure NumPy, no scipy / librosa dependency.
====================================================================
Extracts the same 39-dimensional physics feature vector that
scripts/extract_dsp_features.py produces via librosa + scipy, but
reimplemented entirely with numpy so it can run inside the omni
edge platform without heavy ML dependencies.

Feature vector layout (39 dimensions)
--------------------------------------
Index   Feature
  0-1   RMS         (mean, std across frames)
  2-3   ZCR         (mean, std across frames)
  4     Kurtosis    (excess, Fisher definition: normal dist → 0)
  5     Skewness
  6     Crest factor  (peak / RMS)
  7-8   Spectral centroid  (mean, std across frames)
  9-10  Spectral rolloff   (mean, std) — frequency below which 85% of energy lies
 11-12  Spectral flatness  (mean, std) — geometric/arithmetic mean ratio
 13-38  MFCCs 0-12  (mean and std of each coefficient across frames, interleaved)
        i.e. [mfcc0_mean, mfcc0_std, mfcc1_mean, mfcc1_std, … mfcc12_mean, mfcc12_std]

Training note
-------------
Run ``scripts/train_omni_heads.py`` to train XGB + RF on these features
and export ONNX models to ``omni/models/``.  The EEP orchestrator will
automatically load them and skip the physics stubs.
"""
from __future__ import annotations

import numpy as np

# ─── Frame-level parameters ───────────────────────────────────────────────────
_FRAME_LEN  = 512
_HOP        = 256
_N_FFT      = 512
_N_MELS     = 40
_N_MFCC     = 13
_ROLL_PCT   = 0.85       # spectral rolloff percentile
_FEATURE_DIM = 39        # exported constant for training scripts


# ─── Low-level DSP primitives ─────────────────────────────────────────────────

def _frames(x: np.ndarray, frame_len: int = _FRAME_LEN, hop: int = _HOP) -> np.ndarray:
    """Split 1-D signal into overlapping frames → (n_frames, frame_len)."""
    n = len(x)
    n_frames = max(1, 1 + (n - frame_len) // hop)
    out = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        seg = x[s : s + frame_len]
        out[i, : len(seg)] = seg
    return out


def _rfft_mag(frames: np.ndarray) -> np.ndarray:
    """Magnitude spectrum for each frame → (n_frames, n_fft//2+1)."""
    return np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), n=_N_FFT))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Triangular mel filterbank → (n_mels, n_fft//2+1).

    Uses the standard HTK formula:
        mel(f) = 2595 * log10(1 + f/700)
        f(mel) = 700 * (10^(m/2595) - 1)
    """
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_min, f_max = 0.0, sr / 2.0
    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)

    mel_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points  = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m1, f_m, f_m1r = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m1, f_m):
            if f_m != f_m1:
                fbank[m - 1, k] = (k - f_m1) / (f_m - f_m1)
        for k in range(f_m, f_m1r):
            if f_m1r != f_m:
                fbank[m - 1, k] = (f_m1r - k) / (f_m1r - f_m)
    return fbank


def _dct2(x: np.ndarray) -> np.ndarray:
    """
    Type-II DCT (unnormalized) via FFT. Input shape: (..., N).

    For MFCCs we only need the first n_mfcc components, so a full DCT
    matrix multiply is fine for N=40.
    """
    N = x.shape[-1]
    n = np.arange(N, dtype=np.float64)
    k = n.reshape(-1, 1)               # (N, 1)
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * N)).astype(np.float32)  # (N, N)
    # x: (..., N)  D: (N, N) → result: (..., N)
    return x @ D.T


# ─── Frame-wise spectral features ─────────────────────────────────────────────

def _spectral_centroid(mag: np.ndarray, sr: int) -> np.ndarray:
    """(n_frames,) — weighted mean frequency in Hz."""
    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr).astype(np.float32)
    energy = mag.sum(axis=1, keepdims=True) + 1e-9
    return (mag @ freqs) / energy.squeeze()


def _spectral_rolloff(mag: np.ndarray, sr: int, roll_pct: float = _ROLL_PCT) -> np.ndarray:
    """(n_frames,) — frequency below which roll_pct of total energy lies."""
    freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr).astype(np.float32)
    cum  = np.cumsum(mag, axis=1)
    total = cum[:, -1:] * roll_pct + 1e-9
    # First bin where cumulative energy >= threshold
    idx = np.argmax(cum >= total, axis=1)
    return freqs[idx]


def _spectral_flatness(mag: np.ndarray) -> np.ndarray:
    """(n_frames,) — exp(mean(log|X|)) / mean(|X|), clamped to [0, 1]."""
    eps = 1e-9
    log_mean = np.mean(np.log(mag + eps), axis=1)
    arith    = np.mean(mag, axis=1) + eps
    return np.exp(log_mean) / arith


def _compute_mfccs(mag: np.ndarray, sr: int,
                   n_mels: int = _N_MELS,
                   n_mfcc: int = _N_MFCC) -> np.ndarray:
    """
    (n_frames, n_mfcc) MFCCs.

    Pipeline: log-power mel filterbank → DCT type-II → first n_mfcc coefficients.
    """
    fbank = _mel_filterbank(sr, _N_FFT, n_mels)            # (n_mels, freq_bins)
    mel_power = (mag ** 2) @ fbank.T + 1e-9                # (n_frames, n_mels)
    log_mel   = np.log(mel_power).astype(np.float32)
    dct_out   = _dct2(log_mel)                             # (n_frames, n_mels)
    return dct_out[:, :n_mfcc]                             # (n_frames, n_mfcc)


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_features(pcm: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """
    Extract the 39-dimensional DSP physics feature vector from raw PCM.

    Parameters
    ----------
    pcm : 1-D float32 array, mono signal at ``sr`` Hz
    sr  : sample rate (default 16 000 Hz)

    Returns
    -------
    np.ndarray, shape (39,), float32

    Raises
    ------
    ValueError if pcm is too short (< _FRAME_LEN samples)
    """
    if len(pcm) < _FRAME_LEN:
        raise ValueError(f"PCM too short: {len(pcm)} < {_FRAME_LEN}")

    pcm = pcm.astype(np.float32)

    # ── Time-domain stats ────────────────────────────────────────────────
    frames_t = _frames(pcm, _FRAME_LEN, _HOP)           # (n_frames, frame_len)

    rms_frames = np.sqrt(np.mean(frames_t ** 2, axis=1)) + 1e-9
    zcr_frames = np.mean(np.abs(np.diff(np.sign(frames_t), axis=1)) / 2, axis=1)

    rms_mean  = float(np.mean(rms_frames))
    rms_std   = float(np.std(rms_frames))
    zcr_mean  = float(np.mean(zcr_frames))
    zcr_std   = float(np.std(zcr_frames))

    # Excess kurtosis + skewness on the full signal (not frame-wise)
    mu    = float(np.mean(pcm))
    sigma = float(np.std(pcm)) + 1e-9
    kurt  = float(np.mean(((pcm - mu) / sigma) ** 4)) - 3.0
    skw   = float(np.mean(((pcm - mu) / sigma) ** 3))
    crest = float(np.max(np.abs(pcm))) / rms_mean

    # ── Spectral stats ───────────────────────────────────────────────────
    mag = _rfft_mag(frames_t)                            # (n_frames, freq_bins)

    cent    = _spectral_centroid(mag, sr)
    rolloff = _spectral_rolloff(mag, sr)
    flat    = _spectral_flatness(mag)

    cent_mean  = float(np.mean(cent))
    cent_std   = float(np.std(cent))
    roll_mean  = float(np.mean(rolloff))
    roll_std   = float(np.std(rolloff))
    flat_mean  = float(np.mean(flat))
    flat_std   = float(np.std(flat))

    # ── MFCCs ────────────────────────────────────────────────────────────
    mfccs = _compute_mfccs(mag, sr, _N_MELS, _N_MFCC)   # (n_frames, 13)
    mfcc_means = np.mean(mfccs, axis=0)                  # (13,)
    mfcc_stds  = np.std(mfccs,  axis=0)                  # (13,)

    # interleave: [m0_mean, m0_std, m1_mean, m1_std, ...]
    mfcc_feats = np.empty(26, dtype=np.float32)
    mfcc_feats[0::2] = mfcc_means
    mfcc_feats[1::2] = mfcc_stds

    # ── Assemble 39-d vector ─────────────────────────────────────────────
    feat = np.array([
        rms_mean, rms_std,
        zcr_mean, zcr_std,
        kurt, skw, crest,
        cent_mean, cent_std,
        roll_mean, roll_std,
        flat_mean, flat_std,
    ], dtype=np.float32)

    return np.concatenate([feat, mfcc_feats])


def extract_features_with_meta(
    pcm:          np.ndarray,
    sr:           int   = 16_000,
    pipe_material: str  = "PVC",
    pressure_bar:  float = 3.0,
) -> np.ndarray:
    """
    41-dimensional vector: DSP features (39) + metadata (pipe_material, pressure_bar).
    This matches the feature layout expected by the XGB/RF models from
    scripts/train_omni_heads.py.
    """
    _PIPE_MAP = {"PVC": 0.0, "Steel": 1.0, "Cast_Iron": 2.0}
    dsp = extract_features(pcm, sr)
    meta = np.array([
        _PIPE_MAP.get(pipe_material, 0.0),
        float(pressure_bar),
    ], dtype=np.float32)
    return np.concatenate([dsp, meta])
