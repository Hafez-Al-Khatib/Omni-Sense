"""Unit tests for IEP4 CNN classifier (model-free path)."""

import io

import numpy as np
import pytest
import soundfile as sf
from app.audio import preprocess_audio
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def _make_wav(duration_s: float = 5.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestAudioPreprocessing:
    def test_output_shape(self):
        wav = _make_wav()
        waveform = preprocess_audio(wav)
        assert waveform.shape == (16000 * 5,)
        assert waveform.dtype == np.float32

    def test_short_audio_padded(self):
        t = np.linspace(0, 2.0, 32000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        waveform = preprocess_audio(buf.getvalue())
        assert len(waveform) == 80000

    def test_long_audio_trimmed(self):
        t = np.linspace(0, 8.0, 128000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        waveform = preprocess_audio(buf.getvalue())
        assert len(waveform) == 80000


class TestClassifyEndpoint:
    def test_no_model_returns_503(self, client):
        """Without trained weights, /classify should return 503 gracefully."""
        wav_bytes = _make_wav()
        resp = client.post(
            "/classify",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        # 503 = model not trained yet (expected in CI without weights)
        # 200 = model loaded (expected in deployed environment)
        assert resp.status_code in (200, 503)

    def test_empty_audio_returns_400(self, client):
        resp = client.post(
            "/classify",
            files={"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert resp.status_code == 400
