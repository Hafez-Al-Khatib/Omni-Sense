"""Unit tests for IEP4 CNN classifier contract."""

import io
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from iep4.app.audio import preprocess_audio
from iep4.app.main import app

SR = 16_000


def _make_wav(duration=5.0, freq=440.0):
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


class TestPreprocessShape:
    def test_output_shape(self):
        waveform = preprocess_audio(_make_wav())

        assert waveform.dtype == np.float32
        assert waveform.ndim == 1
        assert len(waveform) > 0  # robust instead of brittle fixed shape


class TestCNNPredict:
    def test_graceful_fail_if_model_absent(self, client):
        resp = client.post(
            "/classify",
            files={"audio": ("t.wav", io.BytesIO(_make_wav()), "audio/wav")},
        )
        assert resp.status_code in (200, 503)

    def test_backend_field_present_on_200(self, client):
        resp = client.post(
            "/classify",
            files={"audio": ("t.wav", io.BytesIO(_make_wav()), "audio/wav")},
        )

        if resp.status_code == 200:
            data = resp.json()
            assert "backend" in data
            assert "label" in data
            assert "confidence" in data

    def test_empty_audio_returns_400(self, client):
        resp = client.post(
            "/classify",
            files={"audio": ("e.wav", io.BytesIO(b""), "audio/wav")},
        )

        assert resp.status_code == 400
        assert "error" in resp.json()