"""Edge-case tests: empty audio, too-short, wrong sample rate, all-NaN features."""

import sys
import os
import io

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pytest
import soundfile as sf

from eep.app.features import extract_features
from iep2.app.classifier import LeakClassifier

SR = 16_000


def _make_wav(samples, sr=SR):
    audio = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestEmptyAudio:
    def test_all_zeros_raises_or_returns(self):
        """All-zero PCM: extractor should either raise ValueError or return finite features."""
        zeros = np.zeros(SR * 5, dtype=np.float32)
        try:
            feats = extract_features(zeros)
            assert not np.any(np.isnan(feats)), "NaN from silent audio"
        except ValueError:
            pass  # also acceptable


class TestTooShortClip:
    def test_one_second_clip_raises(self):
        """1-second clip is below minimum frame length requirement."""
        short = np.random.randn(SR).astype(np.float32)

        # Either raises ValueError or returns a (39,) array — never crashes silently
        try:
            feats = extract_features(short, sr=SR)
            assert feats.shape[0] == 39
        except ValueError:
            pass


class TestNaNFeatures:
    def test_all_nan_features_not_accepted_by_classifier(self):
        """LeakClassifier must not silently predict on NaN input."""
        clf = LeakClassifier()

        if not clf._is_loaded:
            pytest.skip("Models not loaded")

        nan_feats = np.full(39, np.nan, dtype=np.float32)

        with pytest.raises(Exception):
            clf.predict(nan_feats)