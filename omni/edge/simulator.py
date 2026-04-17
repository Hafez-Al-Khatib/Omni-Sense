"""Edge sensor simulator.

Stands in for the C/C++ firmware on real accelerometer nodes. Generates
synthetic acoustic frames from one of three regimes (quiet, leak, pump
artifact) and publishes them to the bus at a configurable cadence.

Also emits telemetry (battery, temperature, firmware) every N frames.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import random
import struct
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import numpy as np

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AcousticFrame, TelemetrySample

log = logging.getLogger("edge")

Regime = Literal["quiet", "leak", "pump"]


def _synthesize(regime: Regime, sr: int = 16000, dur_s: float = 0.975) -> np.ndarray:
    """Cheap audio generator that mimics the spectral signature of each regime.

    This is NOT a physical model — it's just enough to drive the downstream
    ML heads with realistic-looking features for the demo.
    """
    n = int(sr * dur_s)
    t = np.linspace(0, dur_s, n, endpoint=False)
    rng = np.random.default_rng()
    if regime == "quiet":
        x = rng.normal(0, 0.002, n)
    elif regime == "leak":
        # Broadband turbulent hiss + harmonics in 500–3000 Hz band
        base = rng.normal(0, 0.015, n)
        for f in (640, 1120, 1870, 2430):
            base += 0.008 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        x = base
    else:  # pump
        # Narrowband tonal + sidebands around 120 Hz
        x = 0.03 * np.sin(2 * np.pi * 120 * t)
        x += 0.01 * np.sin(2 * np.pi * 240 * t)
        x += rng.normal(0, 0.003, n)
    # soft-clip + to int16
    x = np.tanh(x * 2.5) * 0.9
    return (x * 32767).astype(np.int16)


def _pcm_to_b64(pcm: np.ndarray) -> str:
    raw = struct.pack(f"<{len(pcm)}h", *pcm.tolist())
    return base64.b64encode(raw).decode("ascii")


async def run_sensor(
    sensor_id: str,
    site_id: str,
    lat: float,
    lon: float,
    regime_schedule: list[tuple[float, Regime]],
    cadence_s: float = 1.0,
) -> None:
    """Publish frames following the schedule `[(duration_s, regime), ...]`."""
    bus = get_bus()
    fw = "edge-fw-2026.04.1"
    battery = 100.0
    frame_no = 0
    # advertise presence once
    log.info("sensor %s online at (%.4f,%.4f)", sensor_id, lat, lon)
    for duration, regime in regime_schedule:
        n_frames = max(1, int(duration / cadence_s))
        for _ in range(n_frames):
            pcm = _synthesize(regime)
            frame = AcousticFrame(
                sensor_id=sensor_id,
                site_id=site_id,
                captured_at=datetime.now(timezone.utc),
                pcm_b64=_pcm_to_b64(pcm),
                edge_snr_db=random.uniform(6, 22) if regime != "quiet" else random.uniform(-3, 6),
                edge_vad_confidence=0.9 if regime != "quiet" else 0.1,
                firmware_version=fw,
            )
            await bus.publish(Topics.ACOUSTIC_FRAME, frame)
            frame_no += 1
            if frame_no % 5 == 0:
                battery = max(0.0, battery - random.uniform(0.01, 0.05))
                await bus.publish(
                    Topics.TELEMETRY,
                    TelemetrySample(
                        sensor_id=sensor_id,
                        captured_at=datetime.now(timezone.utc),
                        battery_pct=battery,
                        temperature_c=random.uniform(24, 38),
                        disk_free_mb=random.uniform(2000, 4000),
                        rtc_drift_ms=random.randint(-40, 40),
                        uptime_s=frame_no * int(cadence_s),
                        firmware_version=fw,
                    ),
                )
            await asyncio.sleep(cadence_s)
