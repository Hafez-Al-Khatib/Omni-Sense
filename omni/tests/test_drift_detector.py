"""Drift detector: PSI computation, thresholds, bus integration."""
from __future__ import annotations

import asyncio
import contextlib
import math
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

import omni.common.bus as bus_mod
import omni.mlops.drift_detector as drift_mod
from omni.common.bus import InMemoryBus
from omni.common.schemas import DetectionResult
from omni.mlops.drift_detector import PSI_RETRAIN_THRESHOLD, DriftDetector

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detection(
    xgb: float = 0.5,
    rf: float = 0.5,
    fused: float = 0.5,
    ood: float = 0.05,
    is_ood: bool = False,
    sensor_id: str = "S-TEST-01",
    site_id: str = "HAMRA",
) -> DetectionResult:
    return DetectionResult(
        frame_id=uuid4(),
        sensor_id=sensor_id,
        site_id=site_id,
        captured_at=datetime.now(UTC),
        xgb_p_leak=xgb,
        rf_p_leak=rf,
        cnn_p_leak=fused,
        if_anomaly_score=ood,
        fused_p_leak=fused,
        fused_uncertainty=0.05,
        ood_score=ood,
        is_ood=is_ood,
        is_leak=fused >= 0.5,
    )


def _reference_detections(n: int = 100, seed: int = 0) -> list[DetectionResult]:
    """Stable reference distribution centred around 0.3."""
    rng = np.random.default_rng(seed)
    return [
        _make_detection(
            xgb=float(np.clip(rng.normal(0.3, 0.05), 0, 1)),
            rf=float(np.clip(rng.normal(0.3, 0.05), 0, 1)),
            fused=float(np.clip(rng.normal(0.3, 0.05), 0, 1)),
            ood=float(np.clip(rng.normal(0.05, 0.02), 0, 1)),
        )
        for _ in range(n)
    ]


def _shifted_detections(n: int = 100, seed: int = 1) -> list[DetectionResult]:
    """Heavily shifted distribution centred around 0.8 — should exceed PSI_RETRAIN."""
    rng = np.random.default_rng(seed)
    return [
        _make_detection(
            xgb=float(np.clip(rng.normal(0.8, 0.05), 0, 1)),
            rf=float(np.clip(rng.normal(0.8, 0.05), 0, 1)),
            fused=float(np.clip(rng.normal(0.8, 0.05), 0, 1)),
            ood=float(np.clip(rng.normal(0.4, 0.05), 0, 1)),
        )
        for _ in range(n)
    ]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    drift_mod._detector = None
    yield
    drift_mod._detector = None


@pytest.fixture()
def fresh_bus():
    """Provides a fresh bus with a running background task."""
    bus_mod._bus = InMemoryBus()

    async def _runner():
        bus = bus_mod.get_bus()
        task = asyncio.create_task(bus.run())
        yield bus
        bus.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    return _runner


# ── PSI unit tests ─────────────────────────────────────────────────────────────

def test_psi_identical_distributions_is_zero():
    x = np.linspace(0, 1, 100)
    psi = DriftDetector._psi_1d(x, x)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_large_shift_exceeds_retrain_threshold():
    ref = np.random.default_rng(0).normal(0.2, 0.05, 200).clip(0, 1)
    cur = np.random.default_rng(1).normal(0.8, 0.05, 200).clip(0, 1)
    psi = DriftDetector._psi_1d(ref, cur)
    assert psi >= PSI_RETRAIN_THRESHOLD


def test_psi_is_non_negative():
    rng = np.random.default_rng(99)
    ref = rng.uniform(0, 1, 300)
    cur = rng.uniform(0, 0.5, 300)
    assert DriftDetector._psi_1d(ref, cur) >= 0.0


# ── Cosine similarity ──────────────────────────────────────────────────────────

def test_cosine_identical_vectors_is_one():
    v = np.array([0.5, 0.3, 0.4, 0.1])
    assert DriftDetector._cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert DriftDetector._cosine(a, b) == pytest.approx(0.0, abs=1e-6)


# ── Evaluate: guard conditions ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_returns_none_below_min_window():
    det = DriftDetector()
    det.set_reference(_reference_detections(100))
    # Only 10 samples — below MIN_WINDOW_FOR_TEST
    for d in _reference_detections(10, seed=9):
        async with det._lock:
            det._window.append(d)
    assert await det.evaluate() is None


@pytest.mark.asyncio
async def test_evaluate_returns_none_without_reference():
    det = DriftDetector()
    for d in _reference_detections(60, seed=3):
        async with det._lock:
            det._window.append(d)
    assert await det.evaluate() is None


# ── Drift levels ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_no_drift_when_stable():
    """Use identical data for reference and window so PSI ≈ 0."""
    det = DriftDetector()
    stable = _reference_detections(200, seed=7)
    det.set_reference(stable)
    # Same data in the window → PSI must be ~0
    for d in stable[:60]:
        async with det._lock:
            det._window.append(d)
    report = await det.evaluate()
    assert report is not None
    assert not report.should_retrain


@pytest.mark.asyncio
async def test_evaluate_significant_drift_triggers_retrain():
    det = DriftDetector()
    det.set_reference(_reference_detections(100, seed=0))
    for d in _shifted_detections(60, seed=5):
        async with det._lock:
            det._window.append(d)
    report = await det.evaluate()
    assert report is not None
    assert report.drift_level == "significant"
    assert report.should_retrain is True


@pytest.mark.asyncio
async def test_evaluate_high_ood_rate_triggers_retrain():
    det = DriftDetector()
    det.set_reference(_reference_detections(100, seed=0))
    for _ in range(60):
        async with det._lock:
            det._window.append(_make_detection(is_ood=True, ood=0.9, fused=0.9))
    report = await det.evaluate()
    assert report is not None
    assert report.ood_rate > 0.15
    assert report.should_retrain is True


# ── Report fields ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_fields_are_populated():
    det = DriftDetector()
    det.set_reference(_reference_detections(100, seed=0))
    for d in _reference_detections(60, seed=3):
        async with det._lock:
            det._window.append(d)
    report = await det.evaluate()
    assert report is not None
    assert 0.0 <= report.cosine_similarity <= 1.0 + 1e-9
    assert 0.0 <= report.ood_rate <= 1.0
    assert report.n_samples == 60
    assert not math.isnan(report.psi_max)
    assert report.summary != ""
    det.latest_report is report


# ── Bus integration: retrain published on significant drift ────────────────────

@pytest.mark.asyncio
async def test_on_detection_publishes_retrain_on_drift():
    """Verify that significant drift causes a retrain.request on the bus."""
    bus_mod._bus = InMemoryBus()
    bus = bus_mod.get_bus()
    bus_task = asyncio.create_task(bus.run())

    published: list[dict] = []

    async def capture(payload):
        published.append(payload)

    bus.subscribe("mlops.retrain.request.v1", capture)

    det = DriftDetector()
    det.set_reference(_reference_detections(100, seed=0))
    # Pre-fill window with shifted data (> MIN_WINDOW_FOR_TEST)
    for d in _shifted_detections(100, seed=7):
        async with det._lock:
            det._window.append(d)

    # The 25th call to on_detection triggers evaluate()
    det._report_counter = 0
    for d in _shifted_detections(25, seed=8):
        await det.on_detection({
            "frame_id": str(uuid4()),
            "sensor_id": d.sensor_id,
            "site_id": d.site_id,
            "captured_at": datetime.now(UTC).isoformat(),
            "xgb_p_leak": d.xgb_p_leak,
            "rf_p_leak": d.rf_p_leak,
            "cnn_p_leak": d.cnn_p_leak,
            "if_anomaly_score": d.if_anomaly_score,
            "fused_p_leak": d.fused_p_leak,
            "fused_uncertainty": d.fused_uncertainty,
            "ood_score": d.ood_score,
            "is_ood": d.is_ood,
            "is_leak": d.is_leak,
        })

    await asyncio.sleep(0.05)  # drain bus queue

    bus.stop()
    bus_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await bus_task

    assert len(published) >= 1
    assert "psi_max" in published[0]
    assert published[0]["drift_level"] == "significant"
