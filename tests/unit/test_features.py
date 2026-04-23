"""Unit tests for IEP1 DSP feature extraction (eep/app/features.py)."""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pytest

from eep.app.features import extract_features

SR = 16_000


def _sine(duration=5.0, freq=440.0):
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * freq * t)


def _impulsive(duration=5.0):
    """Spike signal — should have high kurtosis."""
    x = np.zeros(int(SR * duration), dtype=np.float32)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=200)
    x[idx] = rng.uniform(0.5, 1.0, size=200).astype(np.float32)
    return x


class TestOutputShape:
    def test_shape_is_expected(self):
        feats = extract_features(_sine())
        assert feats.shape[0] == 39  # expected feature size

    def test_dtype_is_float32(self):
        feats = extract_features(_sine())
        assert feats.dtype == np.float32


class TestNoNaN:
    def test_no_nan_sine(self):
        feats = extract_features(_sine())
        assert not np.any(np.isnan(feats))

    def test_no_nan_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR * 5).astype(np.float32)
        feats = extract_features(noise)
        assert not np.any(np.isnan(feats))

    def test_no_inf(self):
        feats = extract_features(_sine())
        assert not np.any(np.isinf(feats))


class TestKurtosis:
    def test_kurtosis_positive_for_impulsive(self):
        feats = extract_features(_impulsive())
        kurtosis = feats[4]  # assumed index
        assert kurtosis > 0

    def test_kurtosis_near_zero_for_sine(self):
        feats = extract_features(_sine())
        kurtosis = feats[4]
        assert abs(kurtosis) < 5.0


class TestEdgeCases:
    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.zeros(10, dtype=np.float32))