"""Spatial fusion: weighted centroid, pipe snapping, hypothesis suppression."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omni.common.schemas import DetectionResult
from omni.common.store import DigitalTwinStore, SensorTwin
from omni.spatial.fusion import _haversine_m, _nearest_pipe


def test_haversine_zero_same_point():
    d = _haversine_m(33.8978, 35.4828, 33.8978, 35.4828)
    assert d == pytest.approx(0.0, abs=0.001)


def test_haversine_known_distance():
    # ~1 km roughly between two Beirut points
    d = _haversine_m(33.8978, 35.4828, 33.9068, 35.4828)
    assert 900 < d < 1100, f"expected ~1000m, got {d:.0f}m"


def test_nearest_pipe_hamra():
    pipe_id, dist = _nearest_pipe(33.8980, 35.4831)
    assert "HAMRA" in pipe_id
    assert dist < 500  # within 500m


def test_nearest_pipe_achrafieh():
    pipe_id, dist = _nearest_pipe(33.8890, 35.5230)
    assert "ACHRAFIEH" in pipe_id
    assert dist < 200


@pytest.mark.asyncio
async def test_digital_twin_records_detection():
    twins = DigitalTwinStore()
    await twins.upsert_twin(
        SensorTwin(sensor_id="S-TEST", site_id="test", lat=33.89, lon=35.48)
    )
    det = DetectionResult(
        frame_id=uuid4(),
        sensor_id="S-TEST",
        site_id="test",
        captured_at=datetime.now(UTC),
        xgb_p_leak=0.85,
        rf_p_leak=0.80,
        if_anomaly_score=0.2,
        ood_score=0.3,
        fused_p_leak=0.83,
        fused_uncertainty=0.04,
        is_leak=True,
        is_ood=False,
    )
    await twins.record_detection(det)
    recent = await twins.recent_detections("S-TEST", n=5)
    assert len(recent) == 1
    assert recent[0].is_leak is True
    assert twins.twins["S-TEST"].last_p_leak == pytest.approx(0.83)


@pytest.mark.asyncio
async def test_all_recent_leaks_respects_threshold():
    twins = DigitalTwinStore()
    await twins.upsert_twin(
        SensorTwin(sensor_id="S-A", site_id="test", lat=33.89, lon=35.48)
    )
    await twins.upsert_twin(
        SensorTwin(sensor_id="S-B", site_id="test", lat=33.89, lon=35.49)
    )
    # One hot, one cold
    hot = DetectionResult(
        frame_id=uuid4(), sensor_id="S-A", site_id="test",
        captured_at=datetime.now(UTC),
        xgb_p_leak=0.9, rf_p_leak=0.9, if_anomaly_score=0.2, ood_score=0.2,
        fused_p_leak=0.85, fused_uncertainty=0.03, is_leak=True, is_ood=False,
    )
    cold = DetectionResult(
        frame_id=uuid4(), sensor_id="S-B", site_id="test",
        captured_at=datetime.now(UTC),
        xgb_p_leak=0.1, rf_p_leak=0.1, if_anomaly_score=0.05, ood_score=0.1,
        fused_p_leak=0.12, fused_uncertainty=0.01, is_leak=False, is_ood=False,
    )
    await twins.record_detection(hot)
    await twins.record_detection(cold)

    hot_list = await twins.all_recent_leaks(min_p=0.55, horizon_s=30.0)
    assert len(hot_list) == 1
    assert hot_list[0].sensor_id == "S-A"
