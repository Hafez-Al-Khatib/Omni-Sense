#!/usr/bin/env python3
"""Omni-Sense Raspberry Pi 5 Edge Inference Gateway

Reads vibration data from an ADXL345 accelerometer via I2C (200 Hz, honest),
or generates honest synthetic vibration data.  Accumulates 5-second windows,
resamples to 16 kHz using FFT-based scipy.signal.resample, extracts the
39-dimensional DSP feature vector, and runs IEP2 ONNX models locally:

  1. Isolation Forest OOD gate
  2. XGBoost classifier (if OOD passes)

Publishes JSON diagnosis results to MQTT.

Environment variables:
  SENSOR_ID       — sensor identity               (default: S-RPI5-001)
  SITE_ID         — site slug                     (default: beirut/hamra)
  MQTT_HOST       — broker hostname               (default: localhost)
  MQTT_PORT       — broker port                   (default: 1883)
  OMNI_MODEL_PATH — path to iep2/models           (default: ../../iep2/models)
  HARDWARE_MODE   — force I2C ADXL345 read        (default: auto-detect)
  SIMULATE_MODE   — force synthetic data          (default: auto-detect)
  VAD_THRESHOLD   — RMS gate 0.0-1.0              (default: 0.005)
  FIRMWARE_VER    — firmware string               (default: edge-fw-rpi5-v1)

Usage:
  python agent.py                          # auto-detect sensor / simulation
  HARDWARE_MODE=1 python agent.py          # force I2C hardware
  SIMULATE_MODE=1 python agent.py          # force simulation
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import signal
import struct
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt

# ─────────────────────────── Configuration ────────────────────────────────────

SENSOR_ID: str = os.environ.get("SENSOR_ID", "S-RPI5-001")
SITE_ID: str = os.environ.get("SITE_ID", "beirut/hamra")
MQTT_HOST: str = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
FIRMWARE_VER: str = os.environ.get("FIRMWARE_VER", "edge-fw-rpi5-v1")
VAD_THRESHOLD: float = float(os.environ.get("VAD_THRESHOLD", "0.005"))

# Model path: relative to this script, or override via env var
_default_model_path = (Path(__file__).resolve().parent / ".." / ".." / "iep2" / "models").resolve()
OMNI_MODEL_PATH: Path = Path(os.environ.get("OMNI_MODEL_PATH", str(_default_model_path)))

# ADXL345 I2C constants
ADXL345_ADDR: int = 0x53
ADXL345_REG_DEVID: int = 0x00
ADXL345_REG_BW_RATE: int = 0x2C
ADXL345_REG_POWER_CTL: int = 0x2D
ADXL345_REG_DATA_FORMAT: int = 0x31
ADXL345_REG_FIFO_CTL: int = 0x38
ADXL345_REG_FIFO_STATUS: int = 0x39
ADXL345_REG_DATAX0: int = 0x32

# Honest sampling parameters (NO fake upsampling)
ADXL345_ODR_HZ: int = 200           # ADXL345 output data rate (BW_RATE=0x0B)
LOCAL_SAMPLE_RATE: int = ADXL345_ODR_HZ
COLLECT_SECONDS: int = 5
LOCAL_FRAME_SAMPLES: int = LOCAL_SAMPLE_RATE * COLLECT_SECONDS  # 1 000 @ 200 Hz

# Target rate for feature extraction — models were trained on 16 kHz features.
# We use FFT-based resampling (scipy.signal.resample) which is theoretically
# correct for bandlimited signals.  It does NOT invent new frequency content.
TARGET_RATE_HZ: int = 16_000

# MQTT topics
DIAGNOSIS_TOPIC: str = f"omni/diagnosis/{SITE_ID}/{SENSOR_ID}"
ACOUSTIC_SUB_TOPIC: str = "omni/acoustic/#"

RECONNECT_MIN_S: float = 1.0
RECONNECT_MAX_S: float = 60.0

# ─────────────────────────── Logging ──────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)-5s "
        f"sensor={SENSOR_ID} "
        "%(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("edge-gateway")

# ─────────────────────────── Feature extractor (import or inline) ─────────────

# Prefer the project feature extractor; fall back to an inline copy so the
# agent can be copied to a bare Pi without the full repo.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / ".." / "omni" / "eep"))
    from features import extract_features, extract_features_with_meta  # type: ignore[import]

    log.info("features_imported source=omni.eep.features")
except Exception as _exc:
    log.warning("features_import_failed exc=%s — using inline fallback", _exc)
    extract_features_with_meta = None  # type: ignore[misc]

    # Inline minimal 39-d feature extractor (pure NumPy, matches omni/eep/features.py)
    _FRAME_LEN = 512
    _HOP = 256
    _N_FFT = 512
    _N_MELS = 40
    _N_MFCC = 13
    _ROLL_PCT = 0.85

    def _frames(x: np.ndarray, frame_len: int = _FRAME_LEN, hop: int = _HOP) -> np.ndarray:
        n = len(x)
        n_frames = max(1, 1 + (n - frame_len) // hop)
        out = np.zeros((n_frames, frame_len), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop
            seg = x[s : s + frame_len]
            out[i, : len(seg)] = seg
        return out

    def _rfft_mag(frames: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), n=_N_FFT))

    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        f_min, f_max = 0.0, sr / 2.0
        m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mel_points = np.linspace(m_min, m_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_m1, f_m, f_m1r = bin_points[m - 1], bin_points[m], bin_points[m + 1]
            for k in range(f_m1, f_m):
                if f_m != f_m1:
                    fbank[m - 1, k] = (k - f_m1) / (f_m - f_m1)
            for k in range(f_m, f_m1r):
                if f_m1r != f_m:
                    fbank[m - 1, k] = (f_m1r - k) / (f_m1r - f_m)
        return fbank

    def _dct2(x: np.ndarray) -> np.ndarray:
        N = x.shape[-1]
        n = np.arange(N, dtype=np.float64)
        k = n.reshape(-1, 1)
        D = np.cos(np.pi * k * (2 * n + 1) / (2 * N)).astype(np.float32)
        return x @ D.T

    def _spectral_centroid(mag: np.ndarray, sr: int) -> np.ndarray:
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr).astype(np.float32)
        energy = mag.sum(axis=1, keepdims=True) + 1e-9
        return (mag @ freqs) / energy.squeeze()

    def _spectral_rolloff(mag: np.ndarray, sr: int, roll_pct: float = _ROLL_PCT) -> np.ndarray:
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / sr).astype(np.float32)
        cum = np.cumsum(mag, axis=1)
        total = cum[:, -1:] * roll_pct + 1e-9
        idx = np.argmax(cum >= total, axis=1)
        return freqs[idx]

    def _spectral_flatness(mag: np.ndarray) -> np.ndarray:
        eps = 1e-9
        log_mean = np.mean(np.log(mag + eps), axis=1)
        arith = np.mean(mag, axis=1) + eps
        return np.exp(log_mean) / arith

    def _compute_mfccs(mag: np.ndarray, sr: int, n_mels: int = _N_MELS, n_mfcc: int = _N_MFCC) -> np.ndarray:
        fbank = _mel_filterbank(sr, _N_FFT, n_mels)
        mel_power = (mag ** 2) @ fbank.T + 1e-9
        log_mel = np.log(mel_power).astype(np.float32)
        dct_out = _dct2(log_mel)
        return dct_out[:, :n_mfcc]

    def extract_features(pcm: np.ndarray, sr: int = 16_000) -> np.ndarray:  # noqa: F811
        if len(pcm) < _FRAME_LEN:
            raise ValueError(f"PCM too short: {len(pcm)} < {_FRAME_LEN}")
        pcm = pcm.astype(np.float32)
        frames_t = _frames(pcm, _FRAME_LEN, _HOP)
        rms_frames = np.sqrt(np.mean(frames_t ** 2, axis=1)) + 1e-9
        zcr_frames = np.mean(np.abs(np.diff(np.sign(frames_t), axis=1)) / 2, axis=1)
        rms_mean, rms_std = float(np.mean(rms_frames)), float(np.std(rms_frames))
        zcr_mean, zcr_std = float(np.mean(zcr_frames)), float(np.std(zcr_frames))
        mu, sigma = float(np.mean(pcm)), float(np.std(pcm)) + 1e-9
        kurt = float(np.mean(((pcm - mu) / sigma) ** 4)) - 3.0
        skw = float(np.mean(((pcm - mu) / sigma) ** 3))
        crest = float(np.max(np.abs(pcm))) / rms_mean
        mag = _rfft_mag(frames_t)
        cent = _spectral_centroid(mag, sr)
        rolloff = _spectral_rolloff(mag, sr)
        flat = _spectral_flatness(mag)
        cent_mean, cent_std = float(np.mean(cent)), float(np.std(cent))
        roll_mean, roll_std = float(np.mean(rolloff)), float(np.std(rolloff))
        flat_mean, flat_std = float(np.mean(flat)), float(np.std(flat))
        mfccs = _compute_mfccs(mag, sr, _N_MELS, _N_MFCC)
        mfcc_means = np.mean(mfccs, axis=0)
        mfcc_stds = np.std(mfccs, axis=0)
        mfcc_feats = np.empty(26, dtype=np.float32)
        mfcc_feats[0::2] = mfcc_means
        mfcc_feats[1::2] = mfcc_stds
        feat = np.array([
            rms_mean, rms_std, zcr_mean, zcr_std, kurt, skw, crest,
            cent_mean, cent_std, roll_mean, roll_std, flat_mean, flat_std,
        ], dtype=np.float32)
        return np.concatenate([feat, mfcc_feats])


# ─────────────────────────── Hardware abstraction ─────────────────────────────

class ADXL345Sensor:
    """Minimal ADXL345 driver over smbus2 at 200 Hz — honest, no upsampling."""

    def __init__(self, bus_num: int = 1, address: int = ADXL345_ADDR) -> None:
        import smbus2  # type: ignore[import]
        self._bus = smbus2.SMBus(bus_num)
        self._addr = address
        self._init()

    def _init(self) -> None:
        dev_id = self._bus.read_byte_data(self._addr, ADXL345_REG_DEVID)
        if dev_id != 0xE5:
            raise RuntimeError(f"ADXL345 not found — DEVID=0x{dev_id:02X}, expected 0xE5")
        # 200 Hz ODR (BW_RATE = 0x0B)
        self._bus.write_byte_data(self._addr, ADXL345_REG_BW_RATE, 0x0B)
        # Full-resolution ±16 g
        self._bus.write_byte_data(self._addr, ADXL345_REG_DATA_FORMAT, 0x0B)
        # FIFO stream mode, watermark at 20 samples
        self._bus.write_byte_data(self._addr, ADXL345_REG_FIFO_CTL, 0x94)
        # Measurement mode
        self._bus.write_byte_data(self._addr, ADXL345_REG_POWER_CTL, 0x08)
        time.sleep(0.05)
        log.info("adxl_init odr_hz=200 range=±16g mode=stream")

    def read_fifo(self) -> list[float]:
        """Drain FIFO, return Z-axis samples normalised to [-1, 1]."""
        status = self._bus.read_byte_data(self._addr, ADXL345_REG_FIFO_STATUS)
        n_entries = status & 0x3F
        samples: list[float] = []
        for _ in range(n_entries):
            raw = self._bus.read_i2c_block_data(self._addr, ADXL345_REG_DATAX0, 6)
            z = struct.unpack_from("<h", bytes(raw), 4)[0]
            samples.append(z / 4096.0)
        return samples

    def close(self) -> None:
        try:
            self._bus.write_byte_data(self._addr, ADXL345_REG_POWER_CTL, 0x00)
            self._bus.close()
        except Exception:
            pass


class SimulatedSensor:
    """Generates honest synthetic pipe-vibration data at 3,200 Hz structure-borne rate."""

    def __init__(self) -> None:
        self._t: float = 0.0
        self._leak_mode: bool = False
        self._frame_count: int = 0
        log.warning("simulation_mode active — generating synthetic 3.2 kHz vibration data")

    def read_block(self, n_samples: int, sr: int) -> np.ndarray:
        """Return n_samples of synthetic float32 data at sr Hz."""
        dt = 1.0 / sr
        samples = np.empty(n_samples, dtype=np.float32)
        self._frame_count += 1
        # Toggle leak mode every ~60 s worth of frames
        if self._frame_count % max(1, (60 * sr) // n_samples) == 0:
            self._leak_mode = not self._leak_mode

        for i in range(n_samples):
            t = self._t + i * dt
            # Ambient pipe rumble (low freq)
            ambient = 0.002 * math.sin(2 * math.pi * 50 * t)
            ambient += 0.001 * math.sin(2 * math.pi * 120 * t)
            # Leak signature (300–600 Hz band, amplitude ~0.015)
            if self._leak_mode:
                leak = 0.015 * math.sin(2 * math.pi * 420 * t)
                leak += 0.008 * math.sin(2 * math.pi * 550 * t)
                leak += 0.005 * math.sin(2 * math.pi * 380 * t)
            else:
                leak = 0.0
            # White noise floor
            noise = np.random.normal(0.0, 0.001)
            samples[i] = max(-1.0, min(1.0, ambient + leak + noise))
        self._t += n_samples * dt
        return samples

    def read_fifo(self) -> list[float]:
        # For compatibility with the old polling loop: return ~20 samples at 200 Hz equiv
        return self.read_block(20, ADXL345_ODR_HZ).tolist()

    def close(self) -> None:
        pass


# ─────────────────────────── DSP helpers ──────────────────────────────────────

def honest_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT-based resampling using scipy.signal.resample.

    This is the theoretically correct method for bandlimited signals.
    It does NOT invent new frequency content — frequencies above orig_sr/2
    remain zero (or near-zero) after resampling.
    """
    from scipy import signal  # type: ignore[import]
    n_target = int(len(x) * target_sr / orig_sr)
    return signal.resample(x, n_target)


def compute_rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def compute_snr_db(samples: np.ndarray, noise_floor: float = 0.001) -> float:
    rms = compute_rms(samples)
    if rms <= 0 or noise_floor <= 0:
        return 0.0
    return 20.0 * math.log10(max(rms / noise_floor, 1e-9))


# ─────────────────────────── ONNX Inference Engine ────────────────────────────

class ONNXInferenceEngine:
    """Loads IEP2 ONNX models and runs OOD gate + classification."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort  # type: ignore[import]

        self._model_dir = model_dir
        self._ood_path = model_dir / "isolation_forest.onnx"
        self._clf_path = model_dir / "xgboost_classifier.onnx"

        for p in (self._ood_path, self._clf_path):
            if not p.exists():
                raise FileNotFoundError(f"Model missing: {p}")

        log.info("onnx_loading ood=%s clf=%s", self._ood_path.name, self._clf_path.name)
        t0 = time.monotonic()
        self._sess_ood = ort.InferenceSession(str(self._ood_path), providers=["CPUExecutionProvider"])
        self._sess_clf = ort.InferenceSession(str(self._clf_path), providers=["CPUExecutionProvider"])
        load_ms = (time.monotonic() - t0) * 1000
        log.info("onnx_loaded ood_inputs=%s clf_inputs=%s load_ms=%.1f",
                 [i.name for i in self._sess_ood.get_inputs()],
                 [i.name for i in self._sess_clf.get_inputs()],
                 load_ms)

        # Load label map if present
        label_map_path = model_dir / "label_map.json"
        self._label_map: dict[int, str] = {}
        if label_map_path.exists():
            with open(label_map_path) as f:
                raw = json.load(f)
            self._label_map = {int(k): v for k, v in raw.items() if k.lstrip("-").isdigit()}
            log.info("label_map_loaded %s", self._label_map)
        else:
            self._label_map = {0: "Leak", 1: "No_Leak"}

    def predict(self, features_39: np.ndarray, features_41: np.ndarray | None = None) -> dict[str, Any]:
        """Run OOD detection (39-d) then classification (41-d if available, else 39-d).

        Parameters
        ----------
        features_39 : 39-d DSP feature vector
        features_41 : 41-d vector with metadata (pipe_material, pressure_bar).
                      If None, the classifier receives the 39-d vector (may fail
                      if the model was trained on 41-d).
        """
        x39 = features_39.astype(np.float32).reshape(1, -1)
        x41 = features_41.astype(np.float32).reshape(1, -1) if features_41 is not None else x39

        # OOD gate (Isolation Forest) — trained on 39-d features
        ood_input_name = self._sess_ood.get_inputs()[0].name
        ood_out = self._sess_ood.run(None, {ood_input_name: x39})
        ood_label = int(ood_out[0].flatten()[0])
        ood_score = float(ood_out[1].flatten()[0])
        is_ood = ood_label == -1

        if is_ood:
            return {
                "label": "Unknown",
                "confidence": 0.0,
                "is_ood": True,
                "ood_score": round(ood_score, 4),
            }

        # Classification (XGBoost) — may expect 41-d with metadata
        clf_input_name = self._sess_clf.get_inputs()[0].name
        clf_out = self._sess_clf.run(None, {clf_input_name: x41})
        # XGBoost ONNX output_probability can be a list of dicts: [{0: p0, 1: p1}]
        raw_probs = clf_out[1][0]
        if isinstance(raw_probs, dict):
            probs = np.array([raw_probs[i] for i in sorted(raw_probs.keys())], dtype=np.float32)
        else:
            probs = np.array(raw_probs, dtype=np.float32)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            "label": self._label_map.get(pred_class, str(pred_class)),
            "confidence": round(confidence, 4),
            "is_ood": False,
            "ood_score": round(ood_score, 4),
            "probs": {self._label_map.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs)},
        }


# ─────────────────────────── MQTT wrapper ─────────────────────────────────────

class MQTTPublisher:
    """Thread-safe paho-mqtt wrapper with exponential back-off reconnect."""

    def __init__(self) -> None:
        self._client = mqtt.Client(
            client_id=SENSOR_ID,
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self._connected = threading.Event()
        self._shutdown = False
        self._backoff = RECONNECT_MIN_S
        self._message_queue: list[tuple[str, str]] = []
        self._lock = threading.Lock()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            log.info("mqtt_connected host=%s port=%d", MQTT_HOST, MQTT_PORT)
            self._connected.set()
            self._backoff = RECONNECT_MIN_S
            # Subscribe to ESP32 acoustic frames
            client.subscribe(ACOUSTIC_SUB_TOPIC)
            log.info("mqtt_subscribed topic=%s", ACOUSTIC_SUB_TOPIC)
        else:
            log.error("mqtt_connect_failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected.clear()
        if self._shutdown:
            return
        log.warning("mqtt_disconnected rc=%d backoff=%.1fs", rc, self._backoff)
        time.sleep(self._backoff)
        self._backoff = min(self._backoff * 2, RECONNECT_MAX_S)
        try:
            client.reconnect()
        except Exception as exc:
            log.error("mqtt_reconnect_error exc=%s", exc)

    def _on_publish(self, client, userdata, mid) -> None:
        log.debug("mqtt_published mid=%d", mid)

    def _on_message(self, client, userdata, msg) -> None:
        log.debug("mqtt_received topic=%s len=%d", msg.topic, len(msg.payload))
        # Placeholder: ESP32 acoustic frames can be processed here in future
        # by decoding the base64 PCM and feeding into the inference pipeline.

    def start(self) -> None:
        self._client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.loop_start()
        connected = self._connected.wait(timeout=15.0)
        if not connected:
            log.warning("mqtt_connect_timeout — will keep retrying in background")

    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        if not self._connected.is_set():
            log.warning("mqtt_not_connected topic=%s — dropping message", topic)
            return
        result = self._client.publish(topic, payload, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error("mqtt_publish_error rc=%d topic=%s", result.rc, topic)

    def stop(self) -> None:
        self._shutdown = True
        self._client.loop_stop()
        self._client.disconnect()
        log.info("mqtt_stopped")


# ─────────────────────────── Edge Inference Gateway ───────────────────────────

class EdgeInferenceGateway:
    def __init__(self) -> None:
        # Determine mode
        force_hw = os.environ.get("HARDWARE_MODE", "").lower() in ("1", "true", "yes")
        force_sim = os.environ.get("SIMULATE_MODE", "").lower() in ("1", "true", "yes")

        self._mode: str
        self._sensor: ADXL345Sensor | SimulatedSensor
        if force_hw:
            self._mode = "hardware"
            self._sensor = ADXL345Sensor()
        elif force_sim:
            self._mode = "simulation"
            self._sensor = SimulatedSensor()
        else:
            try:
                self._mode = "hardware"
                self._sensor = ADXL345Sensor()
                log.info("hardware=ADXL345 i2c_addr=0x%02X", ADXL345_ADDR)
            except Exception as exc:
                log.warning("adxl345_unavailable exc=%s — falling back to simulation", exc)
                self._mode = "simulation"
                self._sensor = SimulatedSensor()

        # Load inference models
        self._engine = ONNXInferenceEngine(OMNI_MODEL_PATH)
        self._mqtt = MQTTPublisher()
        self._shutdown_flag = threading.Event()
        self._buffer: list[float] = []
        self._frame_count: int = 0

        log.info(
            "gateway_init mode=%s local_sr=%d target_sr=%d collect_s=%d model_dir=%s",
            self._mode, LOCAL_SAMPLE_RATE, TARGET_RATE_HZ, COLLECT_SECONDS, OMNI_MODEL_PATH,
        )

    def start(self) -> None:
        self._mqtt.start()
        self._run_loop()

    def stop(self) -> None:
        log.info("gateway_shutdown signal received")
        self._shutdown_flag.set()
        self._sensor.close()
        self._mqtt.stop()

    def _run_loop(self) -> None:
        log.info("gateway_started sensor_id=%s site_id=%s", SENSOR_ID, SITE_ID)

        # In simulation mode we can generate blocks directly.
        # In hardware mode we poll the FIFO every ~100 ms (20 samples @ 200 Hz).
        poll_interval = 0.1 if self._mode == "hardware" else 0.05

        while not self._shutdown_flag.is_set():
            t0 = time.monotonic()

            if self._mode == "simulation":
                # Generate one block directly at target rate for efficiency
                block = self._sensor.read_block(
                    n_samples=max(1, int(TARGET_RATE_HZ * poll_interval)),
                    sr=TARGET_RATE_HZ,
                )
                self._buffer.extend(block.tolist())
            else:
                raw = self._sensor.read_fifo()
                self._buffer.extend(raw)

            # When we have 5 seconds worth at local rate, run inference
            required_local_samples = LOCAL_SAMPLE_RATE * COLLECT_SECONDS
            if len(self._buffer) >= required_local_samples:
                local_chunk = np.array(self._buffer[:required_local_samples], dtype=np.float32)
                self._buffer = self._buffer[required_local_samples:]
                self._process_window(local_chunk)

            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, poll_interval - elapsed)
            self._shutdown_flag.wait(timeout=sleep_time)

    def _process_window(self, samples: np.ndarray) -> None:
        self._frame_count += 1
        rms = compute_rms(samples)
        snr_db = compute_snr_db(samples)
        vad_confidence = min(1.0, rms / (VAD_THRESHOLD * 10))

        log.info(
            "window=%d rms=%.5f snr_db=%.1f vad=%.3f vad_pass=%s",
            self._frame_count, rms, snr_db, vad_confidence, rms > VAD_THRESHOLD,
        )

        if rms <= VAD_THRESHOLD:
            log.debug("window=%d vad_rejected rms=%.5f", self._frame_count, rms)
            return

        # Resample to target rate for feature extraction (models trained at 16 kHz)
        if self._mode == "hardware":
            resampled = honest_resample(samples, LOCAL_SAMPLE_RATE, TARGET_RATE_HZ)
        else:
            # Simulation already generates at target rate
            resampled = samples

        # Extract 39-d DSP feature vector (+ 41-d with metadata if available)
        try:
            features_39 = extract_features(resampled, sr=TARGET_RATE_HZ)
            if extract_features_with_meta is not None:
                features_41 = extract_features_with_meta(
                    resampled, sr=TARGET_RATE_HZ, pipe_material="PVC", pressure_bar=3.0
                )
            else:
                features_41 = None
        except Exception as exc:
            log.error("feature_extraction_failed exc=%s", exc)
            return

        # Run edge inference
        try:
            diagnosis = self._engine.predict(features_39, features_41)
        except Exception as exc:
            log.error("inference_failed exc=%s", exc)
            return

        payload = {
            "sensor_id": SENSOR_ID,
            "site_id": SITE_ID,
            "captured_at": datetime.now(UTC).isoformat(),
            "firmware_version": FIRMWARE_VER,
            "source_rate_hz": LOCAL_SAMPLE_RATE if self._mode == "hardware" else 3200,
            "feature_rate_hz": TARGET_RATE_HZ,
            "rms": round(rms, 5),
            "snr_db": round(snr_db, 2),
            "vad_confidence": round(vad_confidence, 4),
            "diagnosis": diagnosis,
        }

        self._mqtt.publish(DIAGNOSIS_TOPIC, json.dumps(payload))
        log.info(
            "diagnosis_published window=%d label=%s confidence=%.3f is_ood=%s",
            self._frame_count,
            diagnosis.get("label"),
            diagnosis.get("confidence", 0.0),
            diagnosis.get("is_ood"),
        )


# ─────────────────────────── Entry point ──────────────────────────────────────

def main() -> None:
    gateway = EdgeInferenceGateway()

    def _handle_signal(signum, frame):
        log.info("signal_received sig=%d", signum)
        gateway.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        gateway.start()
    except KeyboardInterrupt:
        gateway.stop()


if __name__ == "__main__":
    main()
