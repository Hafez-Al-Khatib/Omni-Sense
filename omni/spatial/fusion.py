"""Spatial fusion + triangulation.

Buffers recent detections from correlated sensors, runs a weighted-centroid
localizer, snaps to the nearest pipe segment, and publishes a LeakHypothesis
on each new co-incident cluster.

In production this is a Kalman filter over PostGIS pipe geometry with
TDOA-based triangulation. The weighted-centroid here is a defensible
approximation that recovers the right *segment* from a small sensor mesh.
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from omni.common.bus import Topics, get_bus
from omni.common.schemas import DetectionResult, LeakHypothesis
from omni.common import store

log = logging.getLogger("spatial")

# Pipe network stub — in prod read from PostGIS
# Each segment is a polyline with an ID, plus a simple bounding box for lookups.
PIPE_NETWORK = [
    {"id": "P-HAMRA-A12", "lat": 33.8980, "lon": 35.4830, "length_m": 120},
    {"id": "P-HAMRA-A13", "lat": 33.8992, "lon": 35.4844, "length_m": 95},
    {"id": "P-VERDUN-B07", "lat": 33.8840, "lon": 35.4910, "length_m": 180},
    {"id": "P-ACHRAFIEH-C02", "lat": 33.8890, "lon": 35.5230, "length_m": 150},
]

CORRELATION_WINDOW_S = 12.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_pipe(lat: float, lon: float) -> tuple[str, float]:
    best = min(
        PIPE_NETWORK,
        key=lambda p: _haversine_m(lat, lon, p["lat"], p["lon"]),
    )
    return best["id"], _haversine_m(lat, lon, best["lat"], best["lon"])


async def _try_fuse() -> Optional[LeakHypothesis]:
    """Collect recent hot detections across sensors and publish if clustered."""
    hot = await store.twins().all_recent_leaks(min_p=0.55, horizon_s=CORRELATION_WINDOW_S)
    if len(hot) < 2:
        return None

    # Weighted centroid by fused_p_leak × (1 - uncertainty)
    weights, lats, lons, ids = [], [], [], []
    for det in hot:
        t = store.twins().twins.get(det.sensor_id)
        if not t:
            continue
        w = det.fused_p_leak * max(0.05, 1.0 - det.fused_uncertainty)
        weights.append(w)
        lats.append(t.lat)
        lons.append(t.lon)
        ids.append(det.detection_id)
    if sum(weights) == 0 or len(weights) < 2:
        return None
    lat = sum(w * x for w, x in zip(weights, lats)) / sum(weights)
    lon = sum(w * x for w, x in zip(weights, lons)) / sum(weights)

    # Uncertainty ≈ weighted stddev of sensor positions (Lebanon-scale tiny → m)
    var_lat = sum(w * (x - lat) ** 2 for w, x in zip(weights, lats)) / sum(weights)
    var_lon = sum(w * (x - lon) ** 2 for w, x in zip(weights, lons)) / sum(weights)
    # convert degrees² → meters² roughly
    uncertainty_m = max(
        5.0,
        math.sqrt(var_lat * (111_000**2) + var_lon * (111_000 * math.cos(math.radians(lat))) ** 2),
    )

    pipe_id, snap_m = _nearest_pipe(lat, lon)
    confidence = min(1.0, 0.5 + 0.15 * len(hot) - 0.01 * min(snap_m, 40))

    h = LeakHypothesis(
        contributing_detection_ids=ids,
        lat=lat,
        lon=lon,
        uncertainty_m=uncertainty_m,
        pipe_segment_id=pipe_id,
        distance_along_pipe_m=snap_m,
        estimated_flow_lps=round(0.4 + 0.8 * sum(weights) / len(weights), 2),
        confidence=confidence,
    )
    return h


_last_publish = datetime.min.replace(tzinfo=timezone.utc)
_MIN_GAP = timedelta(seconds=6)


async def on_detection(payload: dict) -> None:
    global _last_publish
    det = DetectionResult(**payload)
    await store.twins().record_detection(det)
    if not det.is_leak:
        return
    # Debounce: we don't want a storm of hypotheses for the same cluster
    now = datetime.now(timezone.utc)
    if now - _last_publish < _MIN_GAP:
        return
    h = await _try_fuse()
    if h is None:
        return
    _last_publish = now
    store.hypotheses().append(h)
    await get_bus().publish(Topics.HYPOTHESIS, h)
    log.info(
        "hypothesis %s pipe=%s conf=%.2f ±%.0fm",
        h.hypothesis_id, h.pipe_segment_id, h.confidence, h.uncertainty_m,
    )


def wire() -> None:
    get_bus().subscribe(Topics.DETECTION, on_detection)
