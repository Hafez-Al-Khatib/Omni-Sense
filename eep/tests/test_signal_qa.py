"""
Unit tests for EEP Signal QA.
"""

import numpy as np
import pytest

from app.services.signal_qa import check_signal_quality


class TestSignalQA:
    """Test suite for signal quality checking."""

    def test_valid_audio(self):
        """Normal audio should pass QA."""
        # Generate a sine wave (valid audio)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        result = check_signal_quality(audio)
        assert result["is_valid"] is True
        assert result["error"] is None
        assert result["rms"] > 0.001

    def test_dead_sensor_silence(self):
        """Complete silence should be flagged as dead sensor."""
        audio = np.zeros(16000, dtype=np.float32)
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert "Dead Sensor" in result["error"]

    def test_near_silence(self):
        """Very quiet audio (below RMS threshold) should fail."""
        audio = np.random.randn(16000).astype(np.float32) * 0.0001
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert "Dead Sensor" in result["error"]

    def test_broken_mic_clipping(self):
        """Heavily clipped audio should be flagged as broken mic."""
        # Create audio that's 50% at maximum amplitude
        audio = np.ones(16000, dtype=np.float32) * 0.995
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert "Broken Microphone" in result["error"]

    def test_empty_audio(self):
        """Empty array should fail."""
        audio = np.array([], dtype=np.float32)
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert "Empty" in result["error"]

    def test_mild_clipping_passes(self):
        """Mild clipping (< 30%) should still pass."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.8 * np.sin(2 * np.pi * 440 * t)
        # Clip a small portion
        audio[audio > 0.99] = 0.99
        result = check_signal_quality(audio)
        assert result["is_valid"] is True

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.01 * np.sin(2 * np.pi * 440 * t)

        # With default threshold (0.001), this should pass
        result = check_signal_quality(audio, silence_threshold=0.001)
        assert result["is_valid"] is True

        # With stricter threshold, it should fail
        result = check_signal_quality(audio, silence_threshold=0.05)
        assert result["is_valid"] is False
