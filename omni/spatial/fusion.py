"""Spatial fusion — TDOA-primary, weighted-centroid fallback.

Primary path (TDOA)
--------------------
When two or more sensors on the same pipe segment both detect a leak within
the correlation window, and their raw PCM frames are still in the rolling
detection window, GCC-PHAT cross-correlation is used to compute the exact
propagation delay Δt.  The leak position is then:

    x = (L − v · Δt) / 2     metres from sensor A

where L = pipe path length between sensors (m) and v = wave speed (m/s).

This collapses the uncertainty from ≈ 20–50 m (centroid) to:

    unc ≈ v · (1/2 · sample_period)   ≈ 0.2 m at 3200 Hz for cast iron

Fallback path (weighted centroid)
-----------------------------------
Used when:
  - fewer than 2 sensors on the same segment are active, or
  - PCM frames are not available (e.g. bus-restarted and frame cache empty)

The centroid estimate uses fused_p_leak × (1 − uncertainty) weights and
snaps the result to the nearest pipe segment.

Pipe network
------------
``PIPE_SEGMENTS`` lists known sensor pairs with their pipe geometry.  In
production this is loaded from PostGIS:

    SELECT s1.sensor_id, s2.sensor_id, ST_Length(p.geom), p.material
    FROM pipes p
    JOIN sensors s1 ON ST_DWithin(p.geom, s1.location, 2)
    JOIN sensors s2 ON ST_DWithin(p.geom, s2.location, 2)
    WHERE s1.sensor_id < s2.sensor_id

For the Beirut demo, the two Hamra sensors (S-HAMRA-001 and S-HAMRA-002)
are defined with a realistic 120 m separation and cast-iron pipe material.
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

import numpy as np

from omni.common import store
from omni.common.bus import Topics, get_bus
from omni.common.schemas import DetectionResult, LeakHypothesis
from omni.spatial.tdoa import (
    PipeSegment,
    TDOAResult,
    fuse_results,
    localize,
)

log = logging.getLogger("spatial")

# ─── Pipe network (sensor pairs + geometry) ───────────────────────────────────
#
# In production: load from PostGIS at startup.
# Format: PipeSegment(segment_id, sensor_a_id, sensor_b_id, length_m, material,
#                     lat_a, lon_a, lat_b, lon_b)
#
PIPE_SEGMENTS: list[PipeSegment] = [
    PipeSegment(
        segment_id    = "P-HAMRA-A12",
        sensor_a_id   = "S-HAMRA-001",
        sensor_b_id   = "S-HAMRA-002",
        pipe_length_m = 120.0,
        pipe_material = "Cast_Iron",
        lat_a=33.8978, lon_a=35.4828,
        lat_b=33.8985, lon_b=35.4845,
    ),
    # Add additional sensor pairs here as the network grows.
    # PipeSegment("P-VERDUN-B07", "S-VERDUN-001", "S-VERDUN-002", 180.0, "PVC", ...),
]

# Quick lookup: (sensor_a, sensor_b) → PipeSegment  (both orderings)
_SEGMENT_MAP: dict[tuple[str, str], PipeSegment] = {}
for _seg in PIPE_SEGMENTS:
    _SEGMENT_MAP[(_seg.sensor_a_id, _seg.sensor_b_id)] = _seg
    _SEGMENT_MAP[(_seg.sensor_b_id, _seg.sensor_a_id)] = _seg

# Fallback pipe network for nearest-pipe snapping
PIPE_NETWORK = [
    {"id": "P-HAMRA-A12",       "lat": 33.8980, "lon": 35.4830, "length_m": 120},
    {"id": "P-HAMRA-A13",       "lat": 33.8992, "lon": 35.4844, "length_m":  95},
    {"id": "P-VERDUN-B07",      "lat": 33.8840, "lon": 35.4910, "length_m": 180},
    {"id": "P-ACHRAFIEH-C02",   "lat": 33.8890, "lon": 35.5230, "length_m": 150},
]

CORRELATION_WINDOW_S = 12.0     # max age of a detection for cluster correlation
_MIN_GAP             = timedelta(seconds=6)   # debounce: min time between hypotheses


# ─── PCM frame cache ─────────────────────────────────────────────────────────
#
# Stores the most recent raw PCM array for each sensor_id so that TDOA can
# cross-correlate them.  Frames are evicted after 2 × CORRELATION_WINDOW_S.
#
_pcm_cache: dict[str, tuple[np.ndarray, int, datetime, float]] = {}
# key: sensor_id → (pcm, sr, captured_at, drift_ms)

_PCM_TTL_S = CORRELATION_WINDOW_S * 2


def cache_pcm(sensor_id: str, pcm: np.ndarray, sr: int, captured_at: datetime) -> None:
    """Store a frame in the PCM cache for TDOA use."""
    # Try to get the latest drift from the digital twin store
    drift = 0.0
    twin = store.twins().twins.get(sensor_id)
    if twin and hasattr(twin, 'rtc_drift_ms'):
        drift = float(twin.rtc_drift_ms)
    
    _pcm_cache[sensor_id] = (pcm, sr, captured_at, drift)


def _evict_stale_pcm() -> None:
    cutoff = datetime.now(UTC) - timedelta(seconds=_PCM_TTL_S)
    stale  = [k for k, (_, _, ts, _) in _pcm_cache.items() if ts < cutoff]
    for k in stale:
        del _pcm_cache[k]


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_pipe(lat: float, lon: float) -> tuple[str, float]:
    best = min(PIPE_NETWORK,
               key=lambda p: _haversine_m(lat, lon, p["lat"], p["lon"]))
    return best["id"], _haversine_m(lat, lon, best["lat"], best["lon"])


# ─── TDOA path ────────────────────────────────────────────────────────────────

def _try_tdoa(
    hot_detections: list[DetectionResult],
) -> tuple[float, float, float, str, float] | None:
    """Attempt TDOA localization using cached PCM frames.

    Returns (lat, lon, uncertainty_m, pipe_segment_id, confidence) or None.
    """
    _evict_stale_pcm()
    active_sensors = {d.sensor_id for d in hot_detections}

    tdoa_results: list[TDOAResult] = []

    for (sid_a, sid_b), segment in _SEGMENT_MAP.items():
        # Only process each pair once (avoid duplicate)
        if sid_a > sid_b:
            continue
        if sid_a not in active_sensors or sid_b not in active_sensors:
            continue
        if sid_a not in _pcm_cache or sid_b not in _pcm_cache:
            log.debug("TDOA skipped %s-%s: PCM not cached", sid_a, sid_b)
            continue

        pcm_a, sr_a, ts_a, drift_a = _pcm_cache[sid_a]
        pcm_b, sr_b, ts_b, drift_b = _pcm_cache[sid_b]

        # Require frames captured within the correlation window
        age_diff = abs((ts_a - ts_b).total_seconds())
        if age_diff > CORRELATION_WINDOW_S:
            log.debug("TDOA skipped %s-%s: frame age diff %.1fs", sid_a, sid_b, age_diff)
            continue

        # Sample rates must match (or resample the lower one)
        sr = sr_a
        if sr_a != sr_b:
            log.warning("TDOA %s-%s: SR mismatch %d vs %d — using min", sid_a, sid_b, sr_a, sr_b)
            sr = min(sr_a, sr_b)
            n = min(len(pcm_a), len(pcm_b))
            pcm_a, pcm_b = pcm_a[:n], pcm_b[:n]

        result = localize(pcm_a.astype(np.float32), pcm_b.astype(np.float32),
                          sr=sr, segment=segment, drift_a_ms=drift_a, drift_b_ms=drift_b)
        tdoa_results.append(result)

    if not tdoa_results:
        return None

    fused = fuse_results(tdoa_results)
    if fused is None or not fused.is_valid:
        return None

    log.info(
        "TDOA localized: seg=%s x_from_A=%.1fm lat=%.5f lon=%.5f unc=%.2fm peak=%.3f",
        fused.segment_id,
        fused.dist_from_a_m,
        fused.lat,
        fused.lon,
        fused.uncertainty_m,
        fused.peak_correlation,
    )
    return (
        fused.lat,
        fused.lon,
        fused.uncertainty_m,
        fused.segment_id,
        fused.confidence,
    )


# ─── Centroid fallback ────────────────────────────────────────────────────────

async def _centroid_fallback(
    hot: list[DetectionResult],
) -> tuple[float, float, float, str, float] | None:
    """Weighted centroid — used when TDOA has insufficient data."""
    weights, lats, lons, _ids = [], [], [], []
    for det in hot:
        t = store.twins().twins.get(det.sensor_id)
        if not t:
            continue
        w = det.fused_p_leak * max(0.05, 1.0 - det.fused_uncertainty)
        weights.append(w)
        lats.append(t.lat)
        lons.append(t.lon)

    if sum(weights) < 1e-9 or len(weights) < 2:
        return None

    w_sum = sum(weights)
    lat   = sum(w * x for w, x in zip(weights, lats, strict=False)) / w_sum
    lon   = sum(w * x for w, x in zip(weights, lons, strict=False)) / w_sum

    # Weighted standard deviation → uncertainty in metres
    var_lat = sum(w * (x - lat)**2 for w, x in zip(weights, lats, strict=False)) / w_sum
    var_lon = sum(w * (x - lon)**2 for w, x in zip(weights, lons, strict=False)) / w_sum
    uncertainty_m = max(
        10.0,
        math.sqrt(var_lat * 111_000**2 +
                  var_lon * (111_000 * math.cos(math.radians(lat)))**2),
    )

    pipe_id, snap_m = _nearest_pipe(lat, lon)
    # Centroid uncertainty is much higher than TDOA; penalise confidence
    confidence = max(0.2, min(0.7, 0.4 + 0.1 * len(hot) - 0.002 * snap_m))
    log.info("Centroid fallback: lat=%.5f lon=%.5f unc=%.0fm snap_m=%.0f",
             lat, lon, uncertainty_m, snap_m)
    return lat, lon, uncertainty_m, pipe_id, confidence


# ─── Main fusion ──────────────────────────────────────────────────────────────

_last_publish = datetime.min.replace(tzinfo=UTC)


async def _try_fuse() -> LeakHypothesis | None:
    hot = await store.twins().all_recent_leaks(
        min_p=0.55, horizon_s=CORRELATION_WINDOW_S
    )
    log.debug("Spatial fusion check: hot_detections=%d", len(hot))
    if len(hot) < 2:
        return None

    ids = [d.detection_id for d in hot]

    # ── Attempt TDOA first ────────────────────────────────────────────────
    tdoa_loc = _try_tdoa(hot)
    if tdoa_loc is not None:
        lat, lon, uncertainty_m, pipe_id, confidence = tdoa_loc
        method = "tdoa"
    else:
        # ── Centroid fallback ─────────────────────────────────────────────
        centroid_loc = await _centroid_fallback(hot)
        if centroid_loc is None:
            return None
        lat, lon, uncertainty_m, pipe_id, confidence = centroid_loc
        method = "centroid"

    avg_flow = round(
        0.4 + 0.8 * sum(d.fused_p_leak for d in hot) / len(hot), 2
    )

    h = LeakHypothesis(
        contributing_detection_ids=ids,
        lat=lat,
        lon=lon,
        uncertainty_m=uncertainty_m,
        pipe_segment_id=pipe_id,
        distance_along_pipe_m=None,   # set by TDOA result if available
        estimated_flow_lps=avg_flow,
        confidence=confidence,
    )

    log.info(
        "hypothesis %s pipe=%s conf=%.2f ±%.0fm method=%s",
        h.hypothesis_id, h.pipe_segment_id, h.confidence, h.uncertainty_m, method,
    )
    return h


async def on_detection(payload: dict) -> None:
    global _last_publish
    det = DetectionResult(**payload)
    await store.twins().record_detection(det)

    if not det.is_leak:
        return

    now = datetime.now(UTC)
    if now - _last_publish < _MIN_GAP:
        return

    h = await _try_fuse()
    if h is None:
        return

    _last_publish = now
    store.hypotheses().append(h)
    await get_bus().publish(Topics.HYPOTHESIS, h)


def wire() -> None:
    get_bus().subscribe(Topics.DETECTION, on_detection)
