"""
Integration tests: EEP → (native DSP features) → IEP2/IEP4 pipeline.

IEP1 (YAMNet) was decommissioned in 2026-04; DSP feature extraction
now happens natively inside EEP (see `omni/eep/features.py`), so the
external hop to IEP1 has been removed from this test's flow.

Requires the stack to be running via docker-compose.
Run: pytest tests/test_integration.py -v
"""

import io
import json
import os

import httpx
import numpy as np
import pytest
import soundfile as sf

EEP_URL = os.environ.get("EEP_URL", "http://localhost:8000")


def make_test_wav(duration_s=5.0, sr=16000, freq=440.0):
    """Generate a test WAV file."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def eep_client():
    return httpx.Client(base_url=EEP_URL, timeout=60.0)


class TestServiceHealth:
    """Verify all services are healthy."""

    def test_eep_health(self, eep_client):
        resp = eep_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestDiagnosePipeline:
    """End-to-end diagnose pipeline tests."""

    def test_valid_diagnosis(self, eep_client):
        """Submit valid audio and get a diagnosis."""
        wav_bytes = make_test_wav()
        metadata = json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})

        resp = eep_client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"metadata": metadata},
        )

        # Should be 200 (diagnosis) or 422 (OOD rejection)
        assert resp.status_code in (200, 422)

        data = resp.json()
        if resp.status_code == 200:
            assert "label" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "anomaly_score" in data
        else:
            # OOD rejection
            assert "anomaly_score" in data

    def test_dead_sensor_rejected(self, eep_client):
        """Silent audio should be rejected before hitting IEPs."""
        silent = np.zeros(80000, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, silent, 16000, format="WAV")

        metadata = json.dumps({"pipe_material": "Steel", "pressure_bar": 2.0})

        resp = eep_client.post(
            "/api/v1/diagnose",
            files={"audio": ("silent.wav", buffer, "audio/wav")},
            data={"metadata": metadata},
        )
        assert resp.status_code == 400

    def test_different_pipe_materials(self, eep_client):
        """Test with all valid pipe materials."""
        for material in ["PVC", "Steel", "Cast_Iron"]:
            wav_bytes = make_test_wav()
            metadata = json.dumps({"pipe_material": material, "pressure_bar": 3.0})

            resp = eep_client.post(
                "/api/v1/diagnose",
                files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
                data={"metadata": metadata},
            )
            assert resp.status_code in (200, 422)


class TestCalibrationPipeline:
    """Test the calibration flow."""

    def test_calibrate_endpoint(self, eep_client):
        """Submit ambient recording for calibration."""
        # Generate a 10-second ambient recording
        wav_bytes = make_test_wav(duration_s=10.0)

        resp = eep_client.post(
            "/api/v1/calibrate",
            files={"audio": ("ambient.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )

        assert resp.status_code in (200, 502)
        if resp.status_code == 200:
            data = resp.json()
            assert "new_threshold" in data
            assert "num_samples" in data
