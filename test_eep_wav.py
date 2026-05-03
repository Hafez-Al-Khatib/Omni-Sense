#!/usr/bin/env python3
"""Test EEP diagnose endpoint with a synthetic WAV file."""
import io, wave, struct, requests, json
import numpy as np

# Generate 1 second of synthetic "leak-like" noise at 16kHz
sr = 16000
t = np.linspace(0, 1, sr)
# Leak signature: 300-600 Hz band + noise
signal = 0.3 * np.sin(2*np.pi*420*t) + 0.2 * np.sin(2*np.pi*550*t)
signal += 0.05 * np.random.randn(sr)
signal = np.clip(signal, -1, 1)

# Write to WAV buffer
buf = io.BytesIO()
with wave.open(buf, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((signal * 32767).astype(np.int16).tobytes())

wav_bytes = buf.getvalue()
print(f"Generated WAV: {len(wav_bytes)} bytes, {sr} Hz, 1s")

# Test EEP
files = {"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
data = {"metadata": json.dumps({"pipe_material": "PVC", "pressure_bar": 3.0})}

try:
    resp = requests.post("http://eep:8000/api/v1/diagnose", files=files, data=data, timeout=30)
    print(f"EEP status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print(f"ERROR: {e}")
