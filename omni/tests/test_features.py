"""DSP feature extractor: shape, finiteness, and basic physics validation."""
from __future__ import annotations

import numpy as np
import pytest

from omni.eep.features import (
    _FEATURE_DIM,
    _compute_mfccs,
    _frames,
    _rfft_mag,
    _spectral_flatness,
    extract_features,
    extract_features_with_meta,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SR = 16_000

def _white_noise(n: int = 80_000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.1, n).astype(np.float32)


def _sine(freq: float = 500.0, n: int = 80_000) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(n: int = 80_000) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ── Shape & finiteness ────────────────────────────────────────────────────────

def test_feature_dim_constant():
    assert _FEATURE_DIM == 39


def test_extract_features_shape():
    feat = extract_features(_white_noise())
    assert feat.shape == (_FEATURE_DIM,)


def test_extract_features_all_finite():
    feat = extract_features(_white_noise())
    assert np.all(np.isfinite(feat)), f"Non-finite values: {feat[~np.isfinite(feat)]}"


def test_extract_features_with_meta_shape():
    feat = extract_features_with_meta(_white_noise(), pipe_material="Steel", pressure_bar=5.0)
    assert feat.shape == (41,)


def test_extract_features_with_meta_last_two():
    """Meta appended correctly: pipe_material=Steel→1.0, pressure=5.0."""
    feat = extract_features_with_meta(_white_noise(), pipe_material="Steel", pressure_bar=5.0)
    assert feat[-2] == pytest.approx(1.0, abs=1e-6)
    assert feat[-1] == pytest.approx(5.0, abs=1e-6)


def test_extract_features_unknown_pipe_material_defaults_to_zero():
    feat = extract_features_with_meta(_white_noise(), pipe_material="Titanium", pressure_bar=2.0)
    assert feat[-2] == pytest.approx(0.0, abs=1e-6)


# ── Different signal types ────────────────────────────────────────────────────

def test_white_noise_finite():
    assert np.all(np.isfinite(extract_features(_white_noise())))


def test_sine_tone_finite():
    assert np.all(np.isfinite(extract_features(_sine(500.0))))


def test_silence_finite():
    """All-zero input must not produce NaN/Inf (division guard)."""
    feat = extract_features(_silence())
    assert np.all(np.isfinite(feat))


def test_short_pcm_raises():
    with pytest.raises(ValueError, match="too short"):
        extract_features(np.zeros(100, dtype=np.float32))


# ── RMS index 0 is positive for non-silent signal ────────────────────────────

def test_rms_positive_for_noise():
    feat = extract_features(_white_noise())
    assert feat[0] > 0.0   # rms_mean at index 0


def test_rms_near_zero_for_silence():
    feat = extract_features(_silence())
    assert feat[0] < 1e-6


# ── Spectral centroid is higher for high-frequency tone ───────────────────────

def test_centroid_higher_for_hf_tone():
    feat_lf = extract_features(_sine(200.0))
    feat_hf = extract_features(_sine(4000.0))
    # centroid_mean at index 7
    assert feat_hf[7] > feat_lf[7], (
        f"HF centroid ({feat_hf[7]:.1f}) should exceed LF ({feat_lf[7]:.1f})"
    )


# ── Kurtosis is high for impulsive signals ────────────────────────────────────

def test_kurtosis_high_for_impulsive():
    """Impulsive (delta-spike) signal has high kurtosis."""
    impulse = np.zeros(80_000, dtype=np.float32)
    impulse[40_000] = 1.0
    feat = extract_features(impulse)
    # kurtosis at index 4
    assert feat[4] > 10.0, f"Expected high kurtosis, got {feat[4]:.2f}"


def test_kurtosis_near_zero_for_gaussian():
    """Gaussian noise has excess kurtosis ≈ 0."""
    rng  = np.random.default_rng(1)
    pcm  = rng.normal(0, 0.1, 80_000).astype(np.float32)
    feat = extract_features(pcm)
    assert abs(feat[4]) < 2.0, f"Expected near-zero kurtosis, got {feat[4]:.3f}"


# ── MFCCs ────────────────────────────────────────────────────────────────────

def test_mfcc_range():
    """MFCC values for white noise should be in a physically reasonable range."""
    feat  = extract_features(_white_noise())
    mfccs = feat[13:]   # indices 13-38 are MFCC mean/std
    assert mfccs.shape == (26,)
    # Absolute values should not be astronomically large
    assert np.all(np.abs(mfccs) < 1e4), f"MFCC out of range: max={np.abs(mfccs).max()}"


# ── Low-level helpers ─────────────────────────────────────────────────────────

def test_frames_shape():
    pcm    = np.ones(8000, dtype=np.float32)
    frames = _frames(pcm, frame_len=512, hop=256)
    assert frames.ndim == 2
    assert frames.shape[1] == 512


def test_rfft_mag_non_negative():
    pcm  = _white_noise(8000)
    frms = _frames(pcm)
    mag  = _rfft_mag(frms)
    assert np.all(mag >= 0.0)


def test_spectral_flatness_range():
    pcm  = _white_noise()
    frms = _frames(pcm)
    mag  = _rfft_mag(frms)
    flat = _spectral_flatness(mag)
    # Flatness is in [0, 1]: white noise ≈ high flatness
    assert np.all(flat >= 0.0)
    assert np.all(flat <= 1.01)   # allow tiny float error


def test_mfccs_shape():
    pcm   = _white_noise()
    frms  = _frames(pcm)
    mag   = _rfft_mag(frms)
    mfccs = _compute_mfccs(mag, SR, n_mels=40, n_mfcc=13)
    assert mfccs.ndim == 2
    assert mfccs.shape[1] == 13
