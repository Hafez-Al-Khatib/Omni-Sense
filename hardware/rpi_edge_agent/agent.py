#!/usr/bin/env python3
"""Omni-Sense Raspberry Pi Edge Agent — production-grade.

Reads from an ADXL345 accelerometer via I2C (smbus2), or falls back to
simulated vibration data when hardware is unavailable.  Accumulates 15 600
samples at a 16 kHz effective rate, applies VAD, encodes PCM16 and publishes
acoustic frames + periodic telemetry over MQTT with mTLS.

Environment variables (all optional, have defaults):
  SENSOR_ID       — e.g. S-HAMRA-001          (default: S-HAMRA-001)
  SITE_ID         — e.g. beirut/hamra          (default: beirut/hamra)
  MQTT_HOST       — broker hostname            (default: localhost)
  MQTT_PORT       — broker port                (default: 8883)
  MQTT_CA         — path to CA cert            (default: /etc/omni/certs/ca.crt)
  MQTT_CERT       — path to client cert        (default: /etc/omni/certs/{SENSOR_ID}.crt)
  MQTT_KEY        — path to client key         (default: /etc/omni/certs/{SENSOR_ID}.key)
  VAD_THRESHOLD   — RMS threshold 0.0-1.0      (default: 0.005)
  FIRMWARE_VER    — firmware string            (default: edge-fw-2026.04.1)
  SIMULATE        — force simulation mode      (default: auto-detect)

Usage:
  python agent.py
  SIMULATE=1 python agent.py  # forced simulation for testing
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

import paho.mqtt.client as mqtt

# ─────────────────────────── Configuration ────────────────────────────────────

SENSOR_ID: str = os.environ.get("SENSOR_ID", "S-HAMRA-001")
SITE_ID: str = os.environ.get("SITE_ID", "beirut/hamra")
MQTT_HOST: str = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT: int = int(os.environ.get("MQTT_PORT", "8883"))
FIRMWARE_VER: str = os.environ.get("FIRMWARE_VER", "edge-fw-2026.04.1")
VAD_THRESHOLD: float = float(os.environ.get("VAD_THRESHOLD", "0.005"))

# Certificate paths — fall back to /etc/omni/certs/ or /certs/
_cert_base = Path("/etc/omni/certs")
if not _cert_base.exists():
    _cert_base = Path("/certs")

MQTT_CA: str = os.environ.get("MQTT_CA", str(_cert_base / "ca.crt"))
MQTT_CERT: str = os.environ.get("MQTT_CERT", str(_cert_base / f"{SENSOR_ID}.crt"))
MQTT_KEY: str = os.environ.get("MQTT_KEY", str(_cert_base / f"{SENSOR_ID}.key"))

# ADXL345 I2C constants
ADXL345_ADDR: int = 0x53           # ALT address 0x1D if SDO=HIGH
ADXL345_REG_DEVID: int = 0x00
ADXL345_REG_BW_RATE: int = 0x2C
ADXL345_REG_POWER_CTL: int = 0x2D
ADXL345_REG_DATA_FORMAT: int = 0x31
ADXL345_REG_FIFO_CTL: int = 0x38
ADXL345_REG_FIFO_STATUS: int = 0x39
ADXL345_REG_DATAX0: int = 0x32

# Sampling: ADXL345 FIFO at 200 Hz, upsample 80× to 16 000 Hz effective
ADXL345_ODR_HZ: int = 200           # ADXL345 output data rate with BW_RATE=0x0B
UPSAMPLE_FACTOR: int = 80           # 200 × 80 = 16 000
TARGET_RATE_HZ: int = ADXL345_ODR_HZ * UPSAMPLE_FACTOR  # 16 000
FRAME_SAMPLES: int = 15_600         # 0.975 s at 16 kHz
TELEMETRY_PERIOD_FRAMES: int = 30   # publish telemetry every N acoustic frames

# MQTT topics
ACOUSTIC_TOPIC: str = f"omni/sensor/{SITE_ID}/{SENSOR_ID}/acoustic"
TELEMETRY_TOPIC: str = f"omni/sensor/{SITE_ID}/{SENSOR_ID}/telemetry"

# Reconnect back-off
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
log = logging.getLogger("edge-agent")

# ─────────────────────────── Hardware abstraction ─────────────────────────────

class ADXL345:
    """Minimal ADXL345 driver over smbus2."""

    def __init__(self, bus_num: int = 1, address: int = ADXL345_ADDR) -> None:
        import smbus2  # type: ignore[import]
        self._bus = smbus2.SMBus(bus_num)
        self._addr = address
        self._init()

    def _init(self) -> None:
        # Verify device ID
        dev_id = self._bus.read_byte_data(self._addr, ADXL345_REG_DEVID)
        if dev_id != 0xE5:
            raise RuntimeError(f"ADXL345 not found — DEVID=0x{dev_id:02X}, expected 0xE5")

        # 200 Hz output data rate (BW_RATE register: 0x0B = 200 Hz, low power off)
        self._bus.write_byte_data(self._addr, ADXL345_REG_BW_RATE, 0x0B)

        # Full-resolution ±16g, 4mg/LSB, SPI 4-wire (DATA_FORMAT)
        self._bus.write_byte_data(self._addr, ADXL345_REG_DATA_FORMAT, 0x0B)

        # FIFO stream mode, watermark at 20 samples
        self._bus.write_byte_data(self._addr, ADXL345_REG_FIFO_CTL, 0x94)

        # Measurement mode
        self._bus.write_byte_data(self._addr, ADXL345_REG_POWER_CTL, 0x08)
        time.sleep(0.05)

    def read_fifo(self) -> list[float]:
        """Drain FIFO, return Z-axis samples normalised to [-1, 1]."""
        status = self._bus.read_byte_data(self._addr, ADXL345_REG_FIFO_STATUS)
        n_entries = status & 0x3F
        samples: list[float] = []
        for _ in range(n_entries):
            raw = self._bus.read_i2c_block_data(self._addr, ADXL345_REG_DATAX0, 6)
            # Z is bytes [4:6], little-endian signed 16-bit
            z = struct.unpack_from("<h", bytes(raw), 4)[0]
            # ±16g full-res: 4 mg/LSB → normalise to ±1 (at ±16g = 4096 LSB)
            samples.append(z / 4096.0)
        return samples

    def close(self) -> None:
        try:
            self._bus.write_byte_data(self._addr, ADXL345_REG_POWER_CTL, 0x00)
            self._bus.close()
        except Exception:
            pass


class SimulatedSensor:
    """Generates realistic pipe-vibration data — used when no hardware present."""

    def __init__(self) -> None:
        self._t: float = 0.0
        self._leak_mode: bool = False
        self._frame_count: int = 0
        log.warning("Hardware unavailable — running in SIMULATION mode")

    def read_fifo(self) -> list[float]:
        # Simulate a FIFO drain of ~20 samples at 200 Hz
        n = 20
        samples: list[float] = []
        dt = 1.0 / ADXL345_ODR_HZ

        # Toggle leak mode every ~60 s to exercise VAD
        self._frame_count += 1
        if self._frame_count % (60 * ADXL345_ODR_HZ // n) == 0:
            self._leak_mode = not self._leak_mode

        for i in range(n):
            t = self._t + i * dt
            # Ambient pipe rumble
            ambient = 0.002 * math.sin(2 * math.pi * 50 * t)
            ambient += 0.001 * math.sin(2 * math.pi * 120 * t)
            # Narrow-band leak hiss (300-600 Hz, amplitude ~0.015)
            if self._leak_mode:
                leak = 0.015 * math.sin(2 * math.pi * 420 * t)
                leak += 0.008 * math.sin(2 * math.pi * 550 * t)
            else:
                leak = 0.0
            # White noise floor
            import random
            noise = random.gauss(0, 0.001)
            samples.append(max(-1.0, min(1.0, ambient + leak + noise)))
        self._t += n * dt
        return samples

    def close(self) -> None:
        pass


# ─────────────────────────── DSP helpers ──────────────────────────────────────

def linear_upsample(samples: list[float], factor: int) -> list[float]:
    """Linear interpolation upsampling by integer factor."""
    if not samples:
        return []
    out: list[float] = []
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        for k in range(factor):
            out.append(a + (b - a) * k / factor)
    out.append(samples[-1])
    return out


def compute_rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def compute_snr_db(samples: list[float], noise_floor: float = 0.001) -> float:
    rms = compute_rms(samples)
    if rms <= 0 or noise_floor <= 0:
        return 0.0
    return 20.0 * math.log10(max(rms / noise_floor, 1e-9))


def to_pcm16_b64(samples: list[float]) -> str:
    """Convert float samples [-1, 1] to PCM16 little-endian, base64-encode."""
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    pcm = struct.pack(f"<{len(clipped)}h", *(int(s * 32767) for s in clipped))
    return base64.b64encode(pcm).decode("ascii")


# ─────────────────────────── Telemetry helpers ────────────────────────────────

def _read_cpu_temp() -> float:
    """Read CPU temperature from sysfs (Raspberry Pi)."""
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(p.read_text().strip()) / 1000.0
    except Exception:
        try:
            import psutil  # type: ignore[import]
            temps = psutil.sensors_temperatures()
            for vals in temps.values():
                if vals:
                    return vals[0].current
        except Exception:
            pass
    return 40.0  # safe fallback


def _read_battery_pct() -> float:
    """Read battery percentage from power supply sysfs or psutil."""
    for ps in Path("/sys/class/power_supply").glob("BAT*"):
        cap = ps / "capacity"
        if cap.exists():
            try:
                return float(cap.read_text().strip())
            except Exception:
                pass
    # UPS HAT may expose a different path
    for ps in Path("/sys/class/power_supply").glob("*"):
        cap = ps / "capacity"
        if cap.exists():
            try:
                return float(cap.read_text().strip())
            except Exception:
                pass
    try:
        import psutil  # type: ignore[import]
        b = psutil.sensors_battery()
        if b:
            return b.percent
    except Exception:
        pass
    return 100.0  # assume mains-powered


def _read_disk_free_mb() -> float:
    try:
        import shutil
        usage = shutil.disk_usage("/")
        return usage.free / (1024 * 1024)
    except Exception:
        return 0.0


def _read_uptime_s() -> int:
    try:
        return int(Path("/proc/uptime").read_text().split()[0].split(".")[0])
    except Exception:
        return 0


def _read_rtc_drift_ms() -> int:
    """Estimate RTC drift via chronyc or hwclock comparison (best-effort)."""
    try:
        import subprocess
        result = subprocess.run(
            ["chronyc", "tracking"],
            capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.splitlines():
            if "System time" in line:
                # "System time     :  0.000000123 seconds slow of NTP time"
                parts = line.split(":")
                if len(parts) > 1:
                    val_str = parts[1].strip().split()[0]
                    return int(float(val_str) * 1000)
    except Exception:
        pass
    return 0


# ─────────────────────────── MQTT client wrapper ──────────────────────────────

class MQTTPublisher:
    """Thread-safe paho-mqtt wrapper with mTLS and exponential back-off reconnect."""

    def __init__(self) -> None:
        self._client = mqtt.Client(
            client_id=SENSOR_ID,
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self._connected = threading.Event()
        self._shutdown = False
        self._backoff = RECONNECT_MIN_S

        self._configure_tls()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

    def _configure_tls(self) -> None:
        ca = MQTT_CA
        cert = MQTT_CERT
        key = MQTT_KEY

        missing = [p for p in [ca, cert, key] if not Path(p).exists()]
        if missing:
            log.warning(
                "cert_missing paths=%s — TLS configured without client cert "
                "(server may reject)",
                missing,
            )
            self._client.tls_set(ca_certs=ca if Path(ca).exists() else None)
        else:
            log.info("tls_configured ca=%s cert=%s", ca, cert)
            import ssl
            self._client.tls_set(
                ca_certs=ca,
                certfile=cert,
                keyfile=key,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            self._client.tls_insecure_set(False)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            log.info("mqtt_connected host=%s port=%d", MQTT_HOST, MQTT_PORT)
            self._connected.set()
            self._backoff = RECONNECT_MIN_S
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
            self._client.reconnect()
        except Exception as exc:
            log.error("mqtt_reconnect_error exc=%s", exc)

    def _on_publish(self, client, userdata, mid) -> None:
        log.debug("mqtt_published mid=%d", mid)

    def start(self) -> None:
        self._client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.loop_start()
        # Wait up to 15 s for first connection
        connected = self._connected.wait(timeout=15.0)
        if not connected:
            log.warning("mqtt_connect_timeout — will keep retrying in background")

    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        if not self._connected.is_set():
            log.warning("mqtt_not_connected topic=%s — dropping frame", topic)
            return
        result = self._client.publish(topic, payload, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error("mqtt_publish_error rc=%d topic=%s", result.rc, topic)

    def stop(self) -> None:
        self._shutdown = True
        self._client.loop_stop()
        self._client.disconnect()
        log.info("mqtt_stopped")


# ─────────────────────────── Main agent loop ──────────────────────────────────

class EdgeAgent:
    def __init__(self) -> None:
        self._sensor = self._init_sensor()
        self._mqtt = MQTTPublisher()
        self._frame_count: int = 0
        self._buffer: list[float] = []
        self._shutdown_flag = threading.Event()

    # ----- lifecycle ---------------------------------------------------------

    @staticmethod
    def _init_sensor():
        force_sim = os.environ.get("SIMULATE", "").lower() in ("1", "true", "yes")
        if not force_sim:
            try:
                sensor = ADXL345()
                log.info("hardware=ADXL345 i2c_addr=0x%02X", ADXL345_ADDR)
                return sensor
            except Exception as exc:
                log.warning("adxl345_unavailable exc=%s — falling back to simulation", exc)
        return SimulatedSensor()

    def start(self) -> None:
        self._mqtt.start()
        self._run_loop()

    def stop(self) -> None:
        log.info("agent_shutdown signal received")
        self._shutdown_flag.set()
        self._sensor.close()
        self._mqtt.stop()

    # ----- sampling loop -----------------------------------------------------

    def _run_loop(self) -> None:
        log.info(
            "agent_started sensor_id=%s site_id=%s target_hz=%d frame_samples=%d",
            SENSOR_ID, SITE_ID, TARGET_RATE_HZ, FRAME_SAMPLES,
        )

        # Timing control: we poll FIFO every ~5 ms (200 Hz / ~20 samples per read)
        poll_interval = 20 / ADXL345_ODR_HZ  # 0.1 s gives ~20 raw samples

        while not self._shutdown_flag.is_set():
            t0 = time.monotonic()

            # 1. Read raw samples from hardware (or simulator)
            raw = self._sensor.read_fifo()

            # 2. Upsample to 16 kHz effective rate
            upsampled = linear_upsample(raw, UPSAMPLE_FACTOR) if raw else []

            # 3. Accumulate into frame buffer
            self._buffer.extend(upsampled)

            # 4. Emit frame(s) when we have enough samples
            while len(self._buffer) >= FRAME_SAMPLES:
                frame_samples = self._buffer[:FRAME_SAMPLES]
                self._buffer = self._buffer[FRAME_SAMPLES:]
                self._process_frame(frame_samples)

            # 5. Sleep to pace the loop
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, poll_interval - elapsed)
            self._shutdown_flag.wait(timeout=sleep_time)

    # ----- frame processing --------------------------------------------------

    def _process_frame(self, samples: list[float]) -> None:
        self._frame_count += 1
        rms = compute_rms(samples)
        snr_db = compute_snr_db(samples)
        vad_confidence = min(1.0, rms / (VAD_THRESHOLD * 10))

        log.info(
            "frame=%d rms=%.5f snr_db=%.1f vad=%.3f vad_pass=%s",
            self._frame_count, rms, snr_db, vad_confidence, rms > VAD_THRESHOLD,
        )

        # VAD gate
        if rms <= VAD_THRESHOLD:
            log.debug("frame=%d vad_rejected rms=%.5f threshold=%.5f",
                      self._frame_count, rms, VAD_THRESHOLD)
            return

        # Encode and publish acoustic frame
        pcm_b64 = to_pcm16_b64(samples)
        payload = {
            "sensor_id": SENSOR_ID,
            "site_id": SITE_ID,
            "captured_at": datetime.now(UTC).isoformat(),
            "pcm_b64": pcm_b64,
            "edge_snr_db": round(snr_db, 2),
            "edge_vad_confidence": round(vad_confidence, 4),
            "firmware_version": FIRMWARE_VER,
        }
        self._mqtt.publish(ACOUSTIC_TOPIC, json.dumps(payload))
        log.info(
            "acoustic_published frame=%d snr_db=%.1f topic=%s",
            self._frame_count, snr_db, ACOUSTIC_TOPIC,
        )

        # Periodic telemetry
        if self._frame_count % TELEMETRY_PERIOD_FRAMES == 0:
            self._publish_telemetry()

    def _publish_telemetry(self) -> None:
        telem = {
            "sensor_id": SENSOR_ID,
            "captured_at": datetime.now(UTC).isoformat(),
            "battery_pct": round(_read_battery_pct(), 1),
            "temperature_c": round(_read_cpu_temp(), 2),
            "disk_free_mb": round(_read_disk_free_mb(), 1),
            "rtc_drift_ms": _read_rtc_drift_ms(),
            "uptime_s": _read_uptime_s(),
            "firmware_version": FIRMWARE_VER,
        }
        self._mqtt.publish(TELEMETRY_TOPIC, json.dumps(telem))
        log.info(
            "telemetry_published frame=%d battery=%.1f%% temp=%.1f°C disk=%.0fMB",
            self._frame_count,
            telem["battery_pct"],
            telem["temperature_c"],
            telem["disk_free_mb"],
        )


# ─────────────────────────── Entry point ──────────────────────────────────────

def main() -> None:
    agent = EdgeAgent()

    def _handle_signal(signum, frame):
        log.info("signal_received sig=%d", signum)
        agent.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        agent.start()
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    main()
