#!/usr/bin/env python3
"""
Omni-Sense MQTT Bridge — Accelerometer → Inference → Results
=============================================================
Subscribes to sensors/+/accel, performs real-time vibration analysis,
optionally forwards to EEP, and publishes results back to MQTT.

Vibration-based classification (no audio model dependency):
  - RMS energy → baseline health indicator
  - Kurtosis → impulsive faults (cracks)
  - Spectral peak ratio → flow-induced turbulence (leaks)
  - Crest factor → periodic vs random vibration
"""

import os
import time
import json
import base64
import struct
import wave
import io
import threading
import warnings
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import paho.mqtt.client as mqtt
import requests

from features import extract_features

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── Configuration ──────────────────────────────────────────────────────────

MQTT_HOST = os.environ.get("OMNI_MQTT_HOST", "mqtt-broker")
MQTT_PORT = int(os.environ.get("OMNI_MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("OMNI_MQTT_USER", "")
MQTT_PASS = os.environ.get("OMNI_MQTT_PASSWORD", "")

EEP_URL = os.environ.get("OMNI_EEP_URL", "http://eep:8000")
DIAGNOSE_ENDPOINT = f"{EEP_URL}/api/v1/diagnose"
USE_EEP = os.environ.get("BRIDGE_USE_EEP", "false").lower() == "true"

IEP2_URL = os.environ.get("OMNI_IEP2_URL", "http://iep2:8002")
IEP2_DIAGNOSE_ENDPOINT = f"{IEP2_URL}/diagnose"

TIMESCALE_DSN = os.environ.get("TIMESCALE_DSN", "")
BRIDGE_METRICS_PORT = int(os.environ.get("BRIDGE_METRICS_PORT", "9091"))

# Buffer / flush settings
BUFFER_MAX_SIZE = 10
FLUSH_INTERVAL_S = 30.0

# Inference cadence
INFERENCE_WINDOW_S = float(os.environ.get("BRIDGE_WINDOW_S", "5.0"))
SOURCE_SR = 3200.0
TARGET_SR = 16000.0

# Topic patterns
TOPIC_ACCEL = "sensors/+/accel"

# ─── Vibration Analysis ─────────────────────────────────────────────────────

def compute_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))

def compute_kurtosis(samples: np.ndarray) -> float:
    if len(samples) < 4:
        return 3.0
    m4 = np.mean((samples - np.mean(samples))**4)
    var = np.var(samples)
    return float(m4 / (var**2 + 1e-12)) if var > 0 else 3.0

def compute_crest_factor(samples: np.ndarray) -> float:
    peak = np.max(np.abs(samples))
    rms = compute_rms(samples)
    return float(peak / (rms + 1e-12))

def compute_spectral_peak_ratio(samples: np.ndarray, sr: float) -> float:
    """Ratio of energy in 50-500 Hz band vs total (leaks create broadband)."""
    if len(samples) < 64:
        return 0.5
    fft = np.fft.rfft(samples)
    power = np.abs(fft)**2
    freqs = np.fft.rfftfreq(len(samples), 1.0/sr)
    total = np.sum(power) + 1e-12
    band_mask = (freqs >= 50) & (freqs <= 500)
    band_power = np.sum(power[band_mask])
    return float(band_power / total)

def compute_zero_crossing_rate(samples: np.ndarray) -> float:
    signs = np.sign(samples)
    signs[signs == 0] = 1
    return float(np.sum(np.abs(np.diff(signs)) > 0) / len(samples))

def classify_vibration(samples: np.ndarray, sr: float) -> dict:
    """
    Vibration-based classifier.  Returns dict with:
      verdict: HEALTHY | LEAK | CRACK | UNKNOWN
      probs:   {HEALTHY: 0.xx, LEAK: 0.xx, CRACK: 0.xx}
      confidence: float
      features: {rms, kurtosis, crest_factor, spectral_ratio, zcr}
    """
    # Normalize to g units (ADXL345 ±2g = ±32768 counts, so / 16384)
    g_samples = samples / 16384.0

    rms = compute_rms(g_samples)
    kurt = compute_kurtosis(g_samples)
    crest = compute_crest_factor(g_samples)
    spec_ratio = compute_spectral_peak_ratio(g_samples, sr)
    zcr = compute_zero_crossing_rate(g_samples)

    features = {
        "rms": round(rms, 4),
        "kurtosis": round(kurt, 2),
        "crest_factor": round(crest, 2),
        "spectral_ratio": round(spec_ratio, 3),
        "zcr": round(zcr, 4),
    }

    # Threshold-based scoring (tuned for demo; real system uses trained model)
    # Healthy: low RMS, normal kurtosis (~3), low crest
    # Leak: elevated RMS, broadband spectrum, moderate kurtosis
    # Crack: high kurtosis (>5), high crest factor, impulsive

    healthy_score = 1.0
    healthy_score *= np.exp(-rms / 0.15)                    # penalize high RMS
    healthy_score *= np.exp(-abs(kurt - 3.0) / 2.0)         # penalize non-gaussian
    healthy_score *= np.exp(-crest / 5.0)                   # penalize high crest

    leak_score = 0.1
    leak_score += 0.4 * (1 - np.exp(-rms / 0.2))           # favor moderate/high RMS
    leak_score += 0.3 * spec_ratio                          # favor broadband
    leak_score += 0.2 * zcr                                 # favor higher ZCR

    crack_score = 0.1
    crack_score += 0.5 * (1 - np.exp(-max(0, kurt - 3.5) / 3.0))  # favor high kurtosis
    crack_score += 0.3 * (1 - np.exp(-max(0, crest - 4.0) / 4.0)) # favor high crest
    crack_score += 0.2 * (1 - spec_ratio)                   # favor narrowband impulses

    # Softmax normalization
    scores = np.array([healthy_score, leak_score, crack_score])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / (np.sum(exp_scores) + 1e-12)

    labels = ["HEALTHY", "LEAK", "CRACK"]
    verdict = labels[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    return {
        "verdict": verdict,
        "probs": {k: round(float(v), 4) for k, v in zip(labels, probs)},
        "confidence": round(confidence, 4),
        "features": features,
    }

# ─── Audio conversion (optional EEP fallback) ───────────────────────────────

def resample_linear(samples: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    if len(samples) == 0:
        return samples
    src_len = len(samples)
    dst_len = int(src_len * dst_sr / src_sr)
    src_x = np.linspace(0, src_len - 1, src_len)
    dst_x = np.linspace(0, src_len - 1, dst_len)
    return np.interp(dst_x, src_x, samples)

def write_wav(samples: np.ndarray, sr: int) -> bytes:
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.95
    int_samples = np.clip(samples, -1.0, 1.0).astype(np.float32) * 32767.0
    int_samples = int_samples.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int_samples.tobytes())
    return buf.getvalue()

def submit_to_eep(sensor_id: str, wav_bytes: bytes) -> Optional[dict]:
    metadata = json.dumps({
        "pipe_material": "PVC",
        "pressure_bar": 3.0,
        "sensor_id": sensor_id,
        "source": "mqtt_bridge",
    })
    files = {"audio": ("frame.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"metadata": metadata}
    try:
        resp = requests.post(DIAGNOSE_ENDPOINT, files=files, data=data, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[bridge] EEP error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[bridge] EEP request failed: {e}")
        return None

# ─── State ──────────────────────────────────────────────────────────────────

@dataclass
class SensorBuffer:
    sensor_id: str
    site_id: str = ""
    samples: list = field(default_factory=list)
    last_frame_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

_buffers: dict[str, SensorBuffer] = {}
_buffers_lock = threading.Lock()
_mqtt_client: Optional[mqtt.Client] = None

_results_buffer: list[dict] = []
_buffer_lock = threading.Lock()
_inference_count = 0
_inference_count_lock = threading.Lock()

# ─── MQTT Handlers ──────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    print(f"[bridge] MQTT connected (rc={rc})")
    client.subscribe(TOPIC_ACCEL)
    print(f"[bridge] Subscribed to {TOPIC_ACCEL}")

def on_message(client, userdata, msg):
    try:
        handle_accel_frame(msg.topic, msg.payload)
    except Exception as e:
        print(f"[bridge] Error handling message: {e}")

def handle_accel_frame(topic: str, payload: bytes):
    parts = topic.split("/")
    if len(parts) < 3:
        return
    sensor_id = parts[1]

    # Payload: base64-encoded int16 array
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return

    n_samples = len(raw) // 2
    if n_samples == 0:
        return

    samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])
    samples_np = np.array(samples, dtype=np.float64)

    with _buffers_lock:
        if sensor_id not in _buffers:
            _buffers[sensor_id] = SensorBuffer(sensor_id=sensor_id)
        buf = _buffers[sensor_id]

    with buf.lock:
        buf.samples.append(samples_np)
        buf.last_frame_time = time.time()
        total_samples = sum(len(s) for s in buf.samples)

        if total_samples >= int(INFERENCE_WINDOW_S * SOURCE_SR):
            all_samples = np.concatenate(buf.samples)
            buf.samples = []
        else:
            return

    # Process outside lock
    process_window(sensor_id, all_samples)

def _map_iep2_label(label: str) -> str:
    """Map IEP2 diagnosis label to bridge verdict space."""
    if not label:
        return "HEALTHY"
    lower = label.lower()
    if "no_leak" in lower or "normal" in lower or "healthy" in lower:
        return "HEALTHY"
    if "crack" in lower:
        return "CRACK"
    if "leak" in lower or "orifice" in lower or "gasket" in lower:
        return "LEAK"
    return "HEALTHY"


def _normalize_probs(iep2_probs: dict) -> dict:
    """Aggregate IEP2 class probabilities into bridge probability space."""
    healthy = 0.0
    leak = 0.0
    crack = 0.0

    for key, val in iep2_probs.items():
        lower = key.lower()
        if "no_leak" in lower or "normal" in lower or "healthy" in lower:
            healthy += val
        elif "crack" in lower:
            crack += val
        elif "leak" in lower or "orifice" in lower or "gasket" in lower:
            leak += val

    total = healthy + leak + crack
    if total > 0:
        healthy /= total
        leak /= total
        crack /= total
    else:
        healthy = leak = crack = 1.0 / 3.0

    return {
        "HEALTHY": round(healthy, 4),
        "LEAK": round(leak, 4),
        "CRACK": round(crack, 4),
    }


def process_window(sensor_id: str, samples: np.ndarray):
    """Extract DSP features at 3.2 kHz and call IEP2 for ML inference."""
    global _inference_count
    t0 = time.time()
    print(f"[bridge] {sensor_id}: processing {len(samples)} samples ({len(samples)/SOURCE_SR:.2f}s)")

    # 1. Extract DSP features and call IEP2
    anomaly_score = None
    ood_threshold = None
    try:
        features = extract_features(samples, sr=int(SOURCE_SR))

        # Add metadata (pipe_material, pressure_bar) → 41-d vector
        features_with_meta = np.concatenate([
            features,
            np.array([0.0, 3.0], dtype=np.float32)  # PVC=0.0, pressure=3.0 bar
        ])

        payload = {
            "embedding": features_with_meta.tolist(),
            "pipe_material": "PVC",
            "pressure_bar": 3.0,
        }

        resp = requests.post(IEP2_DIAGNOSE_ENDPOINT, json=payload, timeout=5)
        iep2_data = resp.json()
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 422:
            # OOD rejection
            verdict = "UNKNOWN"
            confidence = 0.0
            probs = {"HEALTHY": 0.33, "LEAK": 0.33, "CRACK": 0.34}
            source = "iep2_ood"
            features_dict = {}
            anomaly_score = iep2_data.get("anomaly_score")
            ood_threshold = iep2_data.get("threshold")
        else:
            # Map IEP2 label to our verdict space
            label = iep2_data.get("label", "No_Leak")
            verdict = _map_iep2_label(label)
            confidence = iep2_data.get("confidence", 0.5)
            iep2_probs = iep2_data.get("probabilities", {})
            probs = _normalize_probs(iep2_probs)
            source = "iep2_ml"
            features_dict = {}
            anomaly_score = None
            ood_threshold = None

    except Exception as e:
        # Fallback to heuristic on IEP2 failure
        print(f"[bridge] IEP2 error: {e}, falling back to heuristic")
        vibe = classify_vibration(samples, SOURCE_SR)
        latency_ms = (time.time() - t0) * 1000
        verdict = vibe["verdict"]
        confidence = vibe["confidence"]
        probs = vibe["probs"]
        features_dict = vibe["features"]
        source = "vibration_analysis"

    # 2. Build result payload
    result = {
        "sensor_id": sensor_id,
        "ts": time.time() * 1000,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": verdict,
        "probs": probs,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "features": features_dict,
        "window_samples": len(samples),
        "source": source,
    }
    if anomaly_score is not None:
        result["anomaly_score"] = anomaly_score
    if ood_threshold is not None:
        result["ood_threshold"] = ood_threshold

    # 3. Publish result
    result_topic = f"sensors/{sensor_id}/result"
    _mqtt_client.publish(result_topic, json.dumps(result))
    print(f"[bridge] {sensor_id}: {verdict} (conf={confidence:.2f}) → {result_topic}")

    # 4. Write to JSON file for HTTP polling fallback
    try:
        os.makedirs('/srv/dashboard/data', exist_ok=True)
        with open('/srv/dashboard/data/result.json', 'w') as f:
            json.dump(result, f)
    except Exception as e:
        print(f"[bridge] Failed to write result.json: {e}")

    # 5. Buffer for DB persistence
    with _inference_count_lock:
        _inference_count += 1

    db_record = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sensor_id": sensor_id,
        "verdict": verdict,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 1),
        "features": json.dumps(features_dict),
        "source": source,
    }

    with _buffer_lock:
        _results_buffer.append(db_record)
        should_flush = len(_results_buffer) >= BUFFER_MAX_SIZE

    if should_flush:
        flush_results()

# ─── DB Persistence ─────────────────────────────────────────────────────────

def insert_results(records: list[dict]):
    if not psycopg2 or not TIMESCALE_DSN:
        return
    try:
        conn = psycopg2.connect(TIMESCALE_DSN)
        cur = conn.cursor()
        sql = """
            INSERT INTO inference_results
                (captured_at, sensor_id, verdict, confidence, latency_ms, features, source)
            VALUES %s
        """
        values = [
            (
                r["captured_at"],
                r["sensor_id"],
                r["verdict"],
                r["confidence"],
                r["latency_ms"],
                r["features"],
                r["source"],
            )
            for r in records
        ]
        execute_values(cur, sql, values)
        conn.commit()
        cur.close()
        conn.close()
        print(f"[bridge] Flushed {len(records)} records to DB")
    except Exception as e:
        print(f"[bridge] DB flush failed: {e}")


def flush_results(force: bool = False):
    with _buffer_lock:
        if not _results_buffer:
            return
        if not force and len(_results_buffer) < BUFFER_MAX_SIZE:
            return
        batch = _results_buffer.copy()
        _results_buffer.clear()

    insert_results(batch)


def flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL_S)
        flush_results(force=True)


# ─── Metrics HTTP Server ────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            with _inference_count_lock:
                count = _inference_count
            with _buffer_lock:
                buf_size = len(_results_buffer)
            payload = json.dumps({
                "inference_count": count,
                "buffer_size": buf_size,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def metrics_server():
    server = HTTPServer(("0.0.0.0", BRIDGE_METRICS_PORT), MetricsHandler)
    print(f"[bridge] Metrics server on :{BRIDGE_METRICS_PORT}")
    server.serve_forever()


# ─── Periodic flush ─────────────────────────────────────────────────────────

def inference_loop():
    while True:
        time.sleep(1.0)
        now = time.time()
        with _buffers_lock:
            for sid, buf in list(_buffers.items()):
                with buf.lock:
                    if not buf.samples:
                        continue
                    total = sum(len(s) for s in buf.samples)
                    age = now - buf.last_frame_time
                    if age > 3.0 and total > SOURCE_SR:
                        all_samples = np.concatenate(buf.samples)
                        buf.samples = []
                        threading.Thread(target=process_window, args=(sid, all_samples), daemon=True).start()

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    global _mqtt_client
    print("="*60)
    print("Omni-Sense MQTT Bridge")
    print("="*60)
    print(f"MQTT:    {MQTT_HOST}:{MQTT_PORT}")
    print(f"EEP:     {DIAGNOSE_ENDPOINT} (USE_EEP={USE_EEP})")
    print(f"IEP2:    {IEP2_DIAGNOSE_ENDPOINT}")
    print(f"Window:  {INFERENCE_WINDOW_S}s")
    print(f"DB:      {'enabled' if TIMESCALE_DSN and psycopg2 else 'disabled'}")
    print(f"Metrics: :{BRIDGE_METRICS_PORT}")
    print("="*60)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        client = mqtt.Client()
    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.on_connect = on_connect
    client.on_message = on_message

    _mqtt_client = client

    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=flush_loop, daemon=True).start()
    threading.Thread(target=metrics_server, daemon=True).start()

    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"[bridge] MQTT connection error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
