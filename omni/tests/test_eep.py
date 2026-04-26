"""EEP v2 orchestrator: head stubs, fusion, timeout budget, OOD gate."""
import asyncio
import base64
import struct

import numpy as np
import pytest

from omni.eep.orchestrator import (
    LEAK_THRESHOLD,
    OOD_THRESHOLD,
    _decode_pcm,
    _with_budget,
    head_cnn,
    head_isolation_forest,
    head_ood,
    head_rf,
    head_xgb,
)


def _make_pcm(regime: str, sr: int = 16000, dur: float = 0.975) -> np.ndarray:
    n = int(sr * dur)
    t = np.linspace(0, dur, n)
    rng = np.random.default_rng(0)
    if regime == "quiet":
        return (rng.normal(0, 0.002, n)).astype(np.float32)
    if regime == "leak":
        x = rng.normal(0, 0.015, n)
        for f in (640, 1120, 1870):
            x += 0.008 * np.sin(2 * np.pi * f * t)
        return x.astype(np.float32)
    # pump
    x = 0.03 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.003, n)
    return x.astype(np.float32)


@pytest.mark.asyncio
async def test_xgb_leak_above_quiet():
    leak_score = await head_xgb(_make_pcm("leak"), 16000)
    quiet_score = await head_xgb(_make_pcm("quiet"), 16000)
    assert leak_score > quiet_score


@pytest.mark.asyncio
async def test_rf_leak_above_quiet():
    assert await head_rf(_make_pcm("leak"), 16000) > await head_rf(_make_pcm("quiet"), 16000)


@pytest.mark.asyncio
async def test_cnn_leak_above_quiet():
    assert await head_cnn(_make_pcm("leak"), 16000) > await head_cnn(_make_pcm("quiet"), 16000)


@pytest.mark.asyncio
async def test_quiet_ood_in_distribution():
    score = await head_ood(_make_pcm("quiet"), 16000)
    assert score < OOD_THRESHOLD, f"quiet should be in-distribution, got ood={score}"


@pytest.mark.asyncio
async def test_leak_ood_in_distribution():
    score = await head_ood(_make_pcm("leak"), 16000)
    assert score < OOD_THRESHOLD, f"leak should be in-distribution, got ood={score}"


@pytest.mark.asyncio
async def test_fused_leak_above_threshold():
    pcm = _make_pcm("leak")
    xgb = await head_xgb(pcm, 16000)
    rf = await head_rf(pcm, 16000)
    cnn = await head_cnn(pcm, 16000)
    ifc = await head_isolation_forest(pcm, 16000)
    fused = 0.45 * xgb + 0.25 * rf + 0.25 * cnn + 0.05 * ifc
    assert fused >= LEAK_THRESHOLD, f"fused={fused:.3f} should be ≥ {LEAK_THRESHOLD}"


@pytest.mark.asyncio
async def test_fused_quiet_below_threshold():
    pcm = _make_pcm("quiet")
    xgb = await head_xgb(pcm, 16000)
    rf = await head_rf(pcm, 16000)
    cnn = await head_cnn(pcm, 16000)
    ifc = await head_isolation_forest(pcm, 16000)
    fused = 0.45 * xgb + 0.25 * rf + 0.25 * cnn + 0.05 * ifc
    assert fused < LEAK_THRESHOLD, f"quiet fused={fused:.3f} should be < {LEAK_THRESHOLD}"


@pytest.mark.asyncio
async def test_budget_timeout_uses_fallback():
    """Budget enforcement: fallback must fire well before the slow coroutine
    finishes. Uses a generous wall-clock bound (1.5s) so the test stays robust
    when HEAD_BUDGET_*_MS is overridden via env (see conftest.py)."""
    from omni.eep.orchestrator import HEAD_BUDGET_MS

    async def slow():
        await asyncio.sleep(999)
        return 0.5

    val, ms = await _with_budget("xgb", slow(), fallback=0.123)
    assert val == pytest.approx(0.123)
    # Should fire shortly after the configured budget — never approach the 999s sleep.
    assert ms < HEAD_BUDGET_MS["xgb"] + 200, (
        f"fallback took {ms:.0f}ms, expected ≲ {HEAD_BUDGET_MS['xgb']}+200ms"
    )


def test_decode_pcm_roundtrip():
    pcm = np.array([100, -200, 300], dtype=np.int16)
    raw = struct.pack(f"<{len(pcm)}h", *pcm.tolist())
    b64 = base64.b64encode(raw).decode()
    out = _decode_pcm(b64, len(pcm))
    assert out.shape == (3,)
    assert abs(out[0] - 100 / 32767) < 1e-4
