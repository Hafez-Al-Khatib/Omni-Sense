"""
Unit tests for IEP1 audio preprocessing.
"""

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio_processor import preprocess_audio, TARGET_SR, TARGET_SAMPLES


class TestAudioPreprocessor:
    """Test suite for audio preprocessing."""

    def _make_wav_bytes(self, duration_s=5.0, sr=44100, channels=1):
        """Generate WAV bytes for testing."""
        samples = int(sr * duration_s)
        t = np.linspace(0, duration_s, samples, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        if channels > 1:
            audio = np.column_stack([audio] * channels)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        return buffer.getvalue()

    def test_basic_processing(self):
        """Standard 16kHz mono WAV should pass through."""
        wav_bytes = self._make_wav_bytes(duration_s=5.0, sr=16000)
        result = preprocess_audio(wav_bytes)
        assert result.shape == (TARGET_SAMPLES,)
        assert result.dtype == np.float32

    def test_resampling(self):
        """44.1kHz audio should be resampled to 16kHz."""
        wav_bytes = self._make_wav_bytes(duration_s=5.0, sr=44100)
        result = preprocess_audio(wav_bytes)
        assert result.shape == (TARGET_SAMPLES,)

    def test_stereo_to_mono(self):
        """Stereo audio should be converted to mono."""
        wav_bytes = self._make_wav_bytes(duration_s=5.0, sr=16000, channels=2)
        result = preprocess_audio(wav_bytes)
        assert result.ndim == 1
        assert result.shape == (TARGET_SAMPLES,)

    def test_short_audio_padded(self):
        """Audio shorter than 5s should be zero-padded."""
        wav_bytes = self._make_wav_bytes(duration_s=2.0, sr=16000)
        result = preprocess_audio(wav_bytes)
        assert result.shape == (TARGET_SAMPLES,)
        # Last portion should be zeros
        assert np.allclose(result[-int(16000 * 2.5):], 0, atol=0.01)

    def test_long_audio_clipped(self):
        """Audio longer than 5s should be center-clipped."""
        wav_bytes = self._make_wav_bytes(duration_s=10.0, sr=16000)
        result = preprocess_audio(wav_bytes)
        assert result.shape == (TARGET_SAMPLES,)

    def test_normalized(self):
        """Output should be normalized to [-1, 1] range."""
        wav_bytes = self._make_wav_bytes(duration_s=5.0, sr=16000)
        result = preprocess_audio(wav_bytes)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_invalid_bytes(self):
        """Invalid audio bytes should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot decode"):
            preprocess_audio(b"not-valid-audio")
