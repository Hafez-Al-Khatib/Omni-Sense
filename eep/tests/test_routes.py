"""
Unit tests for EEP routes.
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_wav_bytes(duration_s=5.0, sr=16000, freq=440.0, amplitude=0.3):
    """Generate a valid WAV audio byte stream."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    return buffer.getvalue()


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "eep"


class TestDiagnoseEndpoint:
    @patch("app.routes.diagnose.call_iep2_diagnose")
    @patch("app.routes.diagnose.call_iep1_embed")
    def test_successful_diagnosis(self, mock_iep1, mock_iep2, client):
        """Full pipeline: valid audio → IEP1 → IEP2 → success."""
        mock_iep1.return_value = [0.1] * 1024
        mock_iep2.return_value = {
            "is_ood": False,
            "label": "leak",
            "confidence": 0.92,
            "probabilities": {"leak": 0.92, "background": 0.08},
            "anomaly_score": 0.15,
            "is_in_distribution": True,
        }

        wav_bytes = make_wav_bytes()
        metadata = json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})

        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"metadata": metadata},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "leak"
        assert data["confidence"] == 0.92

    def test_empty_audio_rejected(self, client):
        """Empty audio should be rejected with 400."""
        metadata = json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})
        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(b""), "audio/wav")},
            data={"metadata": metadata},
        )
        assert resp.status_code == 400

    def test_invalid_metadata(self, client):
        """Invalid JSON metadata should be rejected."""
        wav_bytes = make_wav_bytes()
        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"metadata": "not-json"},
        )
        assert resp.status_code == 400

    def test_invalid_pipe_material(self, client):
        """Invalid pipe material should be rejected."""
        wav_bytes = make_wav_bytes()
        metadata = json.dumps({"pipe_material": "Copper", "pressure_bar": 3.0})
        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
            data={"metadata": metadata},
        )
        assert resp.status_code == 400

    def test_silence_rejected(self, client):
        """Silent audio should be flagged as dead sensor."""
        silent = np.zeros(80000, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, silent, 16000, format="WAV")
        silent_wav = buffer.getvalue()

        metadata = json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})
        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(silent_wav), "audio/wav")},
            data={"metadata": metadata},
        )
        assert resp.status_code == 400
