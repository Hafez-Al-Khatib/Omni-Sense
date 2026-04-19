"""
Unit tests for EEP Signal QA.

hardware_status values verified here mirror the ops console display:
  "OK"                — signal accepted
  "SENSOR_MALFUNCTION"— RMS below threshold (dead / disconnected sensor)
  "SIGNAL_DEGRADED"   — excessive clipping (ADC saturation)
"""

import numpy as np
from app.services.signal_qa import (
    HW_OK,
    HW_SENSOR_MALFUNCTION,
    HW_SIGNAL_DEGRADED,
    check_signal_quality,
)


class TestSignalQA:
    """Test suite for signal quality checking."""

    def test_valid_audio(self):
        """Normal audio should pass QA with hardware_status OK."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        result = check_signal_quality(audio)
        assert result["is_valid"] is True
        assert result["hardware_status"] == HW_OK
        assert result["error"] is None
        assert result["rms"] > 0.001

    def test_dead_sensor_silence(self):
        """Complete silence → SENSOR_MALFUNCTION."""
        audio = np.zeros(16000, dtype=np.float32)
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert result["hardware_status"] == HW_SENSOR_MALFUNCTION
        assert "malfunction" in result["error"].lower() or "silence" in result["error"].lower() or "below" in result["error"].lower()

    def test_near_silence(self):
        """Very quiet audio (below RMS threshold) → SENSOR_MALFUNCTION."""
        audio = np.random.randn(16000).astype(np.float32) * 0.0001
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert result["hardware_status"] == HW_SENSOR_MALFUNCTION

    def test_broken_mic_clipping(self):
        """Heavily clipped audio → SIGNAL_DEGRADED."""
        audio = np.ones(16000, dtype=np.float32) * 0.995
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert result["hardware_status"] == HW_SIGNAL_DEGRADED
        assert "degraded" in result["error"].lower() or "clipped" in result["error"].lower()

    def test_empty_audio(self):
        """Empty array → SENSOR_MALFUNCTION (no data)."""
        audio = np.array([], dtype=np.float32)
        result = check_signal_quality(audio)
        assert result["is_valid"] is False
        assert result["hardware_status"] == HW_SENSOR_MALFUNCTION
        assert "empty" in result["error"].lower() or "no data" in result["error"].lower()

    def test_mild_clipping_passes(self):
        """Mild clipping (< 30%) should still pass with status OK."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.8 * np.sin(2 * np.pi * 440 * t)
        audio[audio > 0.99] = 0.99
        result = check_signal_quality(audio)
        assert result["is_valid"] is True
        assert result["hardware_status"] == HW_OK

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.01 * np.sin(2 * np.pi * 440 * t)

        result = check_signal_quality(audio, silence_threshold=0.001)
        assert result["is_valid"] is True
        assert result["hardware_status"] == HW_OK

        result = check_signal_quality(audio, silence_threshold=0.05)
        assert result["is_valid"] is False
        assert result["hardware_status"] == HW_SENSOR_MALFUNCTION
