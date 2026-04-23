"""Integration test: POST a 5s WAV to EEP /diagnose. HTTP 200, latency < 3s."""
import io, json, os, time
import numpy as np
import httpx
import pytest
import soundfile as sf

EEP_URL = os.environ.get("EEP_URL", "http://localhost:8000")
SR = 16_000

def make_wav(duration=5.0, freq=440.0):
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return buf.getvalue()

@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=EEP_URL, timeout=10.0)

class TestEEPPipeline:
    def test_http_200_and_label_present(self, client):
        wav = make_wav()
        meta = json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})
        start = time.time()
        resp = client.post(
            "/api/v1/diagnose",
            files={"audio": ("test.wav", io.BytesIO(wav), "audio/wav")},
            data={"metadata": meta},
        )
        latency_ms = (time.time() - start) * 1000

        assert resp.status_code in (200, 422), f"Unexpected status: {resp.status_code}"
        assert latency_ms < 3000, f"Latency too high: {latency_ms:.0f}ms"

        if resp.status_code == 200:
            data = resp.json()
            assert "label" in data
            assert data["label"] in ("Leak", "No_Leak")
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0