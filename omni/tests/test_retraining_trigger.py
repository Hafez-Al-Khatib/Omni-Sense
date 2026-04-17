"""Retraining trigger: cooldown, quality gate, history recording."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import omni.mlops.retraining_trigger as retrain_mod
from omni.mlops.retraining_trigger import RetrainingTrigger, COOLDOWN_MINUTES, get_trigger
from omni.common.bus import InMemoryBus
import omni.common.bus as bus_mod


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_all():
    retrain_mod._trigger = None
    bus_mod._bus = InMemoryBus()
    yield
    retrain_mod._trigger = None
    bus_mod._bus = None


def _trigger_payload(psi: float = 0.30, ood: float = 0.18) -> dict:
    return {
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "psi_max": psi,
        "ood_rate": ood,
        "drift_level": "significant",
        "n_samples": 75,
    }


async def _run_with_bus(coro):
    """Run a coroutine while the bus is processing messages in the background."""
    bus = bus_mod.get_bus()
    bus_task = asyncio.create_task(bus.run())
    try:
        await coro
        await asyncio.sleep(0.05)   # drain queue
    finally:
        bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass


# ── Cooldown ──────────────────────────────────────────────────────────────────

def test_no_cooldown_on_first_call():
    t = RetrainingTrigger()
    assert not t._in_cooldown()


def test_in_cooldown_immediately_after_retrain():
    t = RetrainingTrigger()
    t._last_retrain = datetime.now(timezone.utc)
    assert t._in_cooldown()


def test_cooldown_clears_after_elapsed():
    t = RetrainingTrigger()
    t._last_retrain = datetime.now(timezone.utc) - timedelta(
        minutes=COOLDOWN_MINUTES + 1
    )
    assert not t._in_cooldown()


@pytest.mark.asyncio
async def test_retrain_ignored_during_cooldown():
    t = RetrainingTrigger()
    t._last_retrain = datetime.now(timezone.utc)

    await t.retrain(_trigger_payload())

    # Nothing was recorded since we returned early in cooldown
    assert len(t.history) == 0


# ── Quality gate pass → human review ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrain_skipped_when_model_already_passes_gate():
    published: list[dict] = []

    async def capture(payload):
        published.append(payload)

    bus = bus_mod.get_bus()
    bus.subscribe("mlops.human_review.request.v1", capture)

    t = RetrainingTrigger()
    good_metrics = {"f1": 0.987, "roc_auc": 0.991, "n_samples": 1200, "source": "test"}

    async def _do():
        with patch.object(t, "_evaluate_current_model", return_value=good_metrics):
            await t.retrain(_trigger_payload())

    await _run_with_bus(_do())

    assert len(t.history) == 1
    assert t.history[0]["outcome"] == "skipped_gate_pass"
    assert len(published) == 1
    assert published[0]["reason"] == "drift_detected_but_model_healthy"


# ── Quality gate fail → full retrain ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrain_runs_pipeline_when_gate_fails():
    published: list[dict] = []

    async def capture(payload):
        published.append(payload)

    bus = bus_mod.get_bus()
    bus.subscribe("mlops.model.updated.v1", capture)

    t = RetrainingTrigger()
    bad_metrics = {"f1": 0.82, "roc_auc": 0.85, "n_samples": 200, "source": "test"}

    async def fake_pipeline(feedback_n: int) -> bool:
        return True

    async def _do():
        with patch.object(t, "_evaluate_current_model", return_value=bad_metrics), \
             patch.object(t, "_run_retrain_pipeline", new=fake_pipeline):
            await t.retrain(_trigger_payload())

    await _run_with_bus(_do())

    assert len(t.history) == 1
    assert t.history[0]["outcome"] == "retrained"
    assert len(published) == 1
    assert "updated_at" in published[0]


@pytest.mark.asyncio
async def test_retrain_records_failure_when_pipeline_fails():
    t = RetrainingTrigger()
    bad_metrics = {"f1": 0.80, "roc_auc": 0.82, "n_samples": 200, "source": "test"}

    async def fail_pipeline(n: int) -> bool:
        return False

    with patch.object(t, "_evaluate_current_model", return_value=bad_metrics), \
         patch.object(t, "_run_retrain_pipeline", new=fail_pipeline):
        await t.retrain(_trigger_payload())

    assert t.history[0]["outcome"] == "failed"


# ── Model evaluation unavailable ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrain_skipped_when_model_eval_returns_none():
    t = RetrainingTrigger()
    with patch.object(t, "_evaluate_current_model", return_value=None):
        await t.retrain(_trigger_payload())

    assert t.history[0]["outcome"] == "skipped"
    assert "model evaluation failed" in t.history[0]["reason"]


# ── State tracking ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrain_count_increments():
    t = RetrainingTrigger()
    good_metrics = {"f1": 0.987, "roc_auc": 0.991, "n_samples": 500, "source": "test"}

    with patch.object(t, "_evaluate_current_model", return_value=good_metrics):
        await t.retrain(_trigger_payload())
        t._last_retrain = None   # clear cooldown
        await t.retrain(_trigger_payload())

    assert t._retrain_count == 2
    assert len(t.history) == 2


@pytest.mark.asyncio
async def test_history_records_trigger_payload():
    t = RetrainingTrigger()
    good_metrics = {"f1": 0.99, "roc_auc": 0.995, "n_samples": 100, "source": "test"}
    payload = _trigger_payload(psi=0.35, ood=0.20)

    with patch.object(t, "_evaluate_current_model", return_value=good_metrics):
        await t.retrain(payload)

    record = t.history[0]
    assert record["trigger_psi_max"] == pytest.approx(0.35)
    assert record["trigger_ood_rate"] == pytest.approx(0.20)


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_get_trigger_returns_same_instance():
    t1 = get_trigger()
    t2 = get_trigger()
    assert t1 is t2
