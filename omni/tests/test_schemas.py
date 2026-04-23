"""Schema round-trip and validation tests."""
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omni.common.schemas import (
    AcousticFrame,
    AlertState,
    DetectionResult,
    LeakHypothesis,
    WorkOrder,
    WorkOrderStatus,
)


def test_acoustic_frame_roundtrip():
    frame = AcousticFrame(
        sensor_id="S-TEST-001",
        site_id="beirut/test",
        captured_at=datetime.now(UTC),
        pcm_b64="AAAA",
        edge_snr_db=12.5,
        edge_vad_confidence=0.85,
        firmware_version="test-fw-1.0",
    )
    restored = AcousticFrame(**json.loads(frame.model_dump_json()))
    assert restored.sensor_id == frame.sensor_id
    assert restored.schema_version == "1"


def test_detection_result_bounds():
    det = DetectionResult(
        frame_id=uuid4(),
        sensor_id="S-TEST-001",
        site_id="beirut/test",
        captured_at=datetime.now(UTC),
        xgb_p_leak=0.92,
        rf_p_leak=0.88,
        if_anomaly_score=0.3,
        ood_score=0.4,
        fused_p_leak=0.90,
        fused_uncertainty=0.05,
        is_leak=True,
        is_ood=False,
    )
    assert det.is_leak is True
    assert 0 <= det.fused_p_leak <= 1


def test_detection_result_rejects_bad_probability():
    with pytest.raises(Exception):
        DetectionResult(
            frame_id=uuid4(),
            sensor_id="S",
            site_id="s",
            captured_at=datetime.now(UTC),
            xgb_p_leak=1.5,   # invalid — must be ≤ 1
            rf_p_leak=0.0,
            if_anomaly_score=0.0,
            ood_score=0.0,
            fused_p_leak=0.0,
            fused_uncertainty=0.0,
            is_leak=False,
            is_ood=False,
        )


def test_alert_state_enum():
    assert AlertState.NEW.value == "NEW"
    assert AlertState.VERIFIED.value == "VERIFIED"


def test_hypothesis_roundtrip():
    h = LeakHypothesis(
        contributing_detection_ids=[uuid4(), uuid4()],
        lat=33.8978,
        lon=35.4828,
        uncertainty_m=45.0,
        pipe_segment_id="P-HAMRA-A12",
        confidence=0.87,
    )
    d = json.loads(h.model_dump_json())
    h2 = LeakHypothesis(**d)
    assert h2.pipe_segment_id == "P-HAMRA-A12"
    assert len(h2.contributing_detection_ids) == 2


def test_work_order_default_status():
    wo = WorkOrder(
        alert_id=uuid4(),
        crew_id="CREW-01",
        eta_minutes=15,
    )
    assert wo.status == WorkOrderStatus.DRAFT
