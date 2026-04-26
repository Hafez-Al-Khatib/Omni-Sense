"""EEP v2 — Real-Time Detection Orchestrator.

Consumes acoustic frames, fans out to five ML heads (XGB, RF, CNN, IF, OOD),
fuses their outputs, and publishes DetectionResult.

ML head backends (loaded in priority order at startup)
------------------------------------------------------
ONNX Runtime   — preferred: loads from omni/models/{xgb_head,rf_head}.onnx
                 Trained by: scripts/train_omni_heads.py
                 No xgboost / sklearn needed at runtime.
Physics stubs  — fallback when ONNX models are absent. Deterministic spectral
                 heuristics that behave correctly for the simulator scenario.

Feature extraction
------------------
All ML heads share the same 41-d DSP feature vector computed by
omni/eep/features.py (pure numpy, no librosa/scipy needed).

CNN head (IEP4 integration)
---------------------------
head_cnn() calls IEP4's /classify_raw HTTP endpoint when IEP4_URL is set.
Falls back to stub automatically on timeout, connection error, or 503.
IEP4_URL default: http://iep4:8004  (Docker Compose service name)

SCADA pressure fusion
---------------------
The orchestrator subscribes to Topics.SCADA_READING and caches the latest
ScadaReading per site_id.  In handle_frame(), the cached pressure is used:
  1. Passed as pressure_bar to extract_features_with_meta() so the feature
     vector reflects actual operating conditions.
  2. Applied as a physics multiplier: high ΔP amplifies fused leak probability.

This closes the loop: SCADA data now materially affects detection output.

Timeout budgets
---------------
Each head runs under a hard asyncio timeout.  A slow head returns a safe
neutral value (0.3 for classifiers, 0.5 for OOD) instead of stalling
the whole pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path

import numpy as np

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AcousticFrame, DetectionResult, ScadaReading
from omni.common.tracing import get_tracer
from omni.eep.features import extract_features_with_meta
from omni.spatial.fusion import cache_pcm as _cache_pcm

log = logging.getLogger("eep")
tracer = get_tracer("omni.eep.orchestrator")

# ─── Configuration ────────────────────────────────────────────────────────────

HEAD_BUDGET_MS = {
    "xgb": 30,
    "rf":  30,
    "cnn": 120,   # IEP4 HTTP call — tighter than original 150ms proxy
    "if":  20,
    "ood": 40,
}

FUSION_W = {"xgb": 0.45, "rf": 0.25, "cnn": 0.25, "if": 0.05}

LEAK_THRESHOLD = 0.60
OOD_THRESHOLD  = 1.0

# IEP4 service URL — set via environment (Docker Compose: http://iep4:8004)
IEP4_URL = os.getenv("IEP4_URL", "http://iep4:8004")

# ONNX model paths (relative to project root or Docker workdir)
_OMNI_MODEL_DIR = Path("omni/models")
_XGB_ONNX   = _OMNI_MODEL_DIR / "xgb_head.onnx"
_RF_ONNX    = _OMNI_MODEL_DIR / "rf_head.onnx"
_THR_FILE   = _OMNI_MODEL_DIR / "omni_threshold.json"

# SCADA pressure cache: site_id → latest ScadaReading
_scada_cache: dict[str, ScadaReading] = {}

# Default pressure when SCADA unavailable
_DEFAULT_PRESSURE_BAR = 3.0

# Physics pressure multiplier bounds
_PRESSURE_BOOST_MAX  = 1.20   # max 20% uplift at very high pressure
_PRESSURE_BOOST_PBAR = 7.0    # pressure above this triggers uplift


# ─── Model loader (module-level singletons) ────────────────────────────────────

_xgb_session  = None
_rf_session   = None
_thresholds   = {"xgb": 0.60, "rf": 0.55, "fused": 0.60}
_models_loaded = False


def _load_models() -> None:
    """Try to load ONNX models. Falls back to stubs silently on failure."""
    global _xgb_session, _rf_session, _thresholds, _models_loaded
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — using physics stubs for all heads")
        return

    def _try_load(path: Path, name: str):
        if path.exists():
            try:
                sess = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                log.info("EEP loaded %s ONNX head: %s", name, path)
                return sess
            except Exception as exc:
                log.warning("EEP could not load %s: %s — using stub", name, exc)
        else:
            log.info("EEP: %s ONNX model not found at %s — using stub", name, path)
        return None

    _xgb_session = _try_load(_XGB_ONNX, "XGB")
    _rf_session  = _try_load(_RF_ONNX,  "RF")

    if _THR_FILE.exists():
        import json
        _thresholds.update(json.loads(_THR_FILE.read_text()))
        log.info("EEP thresholds loaded: %s", _thresholds)

    _models_loaded = True


def _onnx_predict(session, feat_vec: np.ndarray) -> float:
    """
    Run one ONNX inference. Returns P(Leak) = P(class 0).

    The ONNX output for skl2onnx-exported classifiers is:
      [0] label: (batch,) int
      [1] probabilities: (batch,) dict  OR  (batch, n_classes) float
    We rely on the probability output.
    """
    x = feat_vec.reshape(1, -1).astype(np.float32)
    iname = session.get_inputs()[0].name
    outputs = session.run(None, {iname: x})

    # outputs[1] may be a list of dicts or a 2-D array depending on model type
    proba_out = outputs[1]
    if isinstance(proba_out, list) and len(proba_out) > 0:
        first = proba_out[0]
        if isinstance(first, dict):
            # Scikit-learn classifiers: [{0: p0, 1: p1}, ...]
            return float(first.get(0, 0.5))   # P(Leak)
        else:
            # Some ONNX models return list of arrays
            return float(np.array(first).flat[0])
    elif isinstance(proba_out, np.ndarray):
        if proba_out.ndim == 2:
            return float(proba_out[0, 0])     # P(class 0 = Leak)
        return float(proba_out.flat[0])
    return 0.5


# ─── PCM helpers ─────────────────────────────────────────────────────────────

def _decode_pcm(b64: str, n_samples: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def _band_energy(pcm: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Normalized band energy fraction — used for SHAP feature reporting."""
    spec  = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(len(pcm), 1 / sr)
    mask  = (freqs >= lo) & (freqs < hi)
    total = spec.sum() + 1e-9
    return float(spec[mask].sum() / total) if mask.any() else 0.0


# ─── SCADA pressure helpers ───────────────────────────────────────────────────

def _get_site_pressure(site_id: str) -> float:
    """Return cached SCADA pressure for a site, or default if unavailable."""
    reading = _scada_cache.get(site_id)
    if reading is not None:
        return float(reading.pressure_bar)
    return _DEFAULT_PRESSURE_BAR


def _pressure_leak_multiplier(pressure_bar: float) -> float:
    """
    Physics-informed multiplier for fused leak probability.

    High pressure differential increases leak likelihood and severity.
    At > _PRESSURE_BOOST_PBAR bar, apply a linear uplift up to _PRESSURE_BOOST_MAX.
    This is conservative — we never decrease the base probability.
    """
    if pressure_bar <= _PRESSURE_BOOST_PBAR:
        return 1.0
    excess = pressure_bar - _PRESSURE_BOOST_PBAR
    # 0.05 per bar above threshold, capped at _PRESSURE_BOOST_MAX
    return min(1.0 + excess * 0.05, _PRESSURE_BOOST_MAX)


async def _handle_scada_reading(payload: dict) -> None:
    """Cache incoming SCADA readings by site_id."""
    try:
        reading = ScadaReading(**payload)
        _scada_cache[reading.site_id] = reading
        log.debug(
            "SCADA cache updated: site=%s pressure=%.2f bar",
            reading.site_id, reading.pressure_bar,
        )
    except Exception as exc:
        log.warning("SCADA reading parse failed: %s", exc)


# ─── Physics stubs (active when ONNX models are absent) ─────────────────────

def _stub_xgb(pcm: np.ndarray, sr: int) -> float:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms < 0.005:
        return 0.05
    leak = _band_energy(pcm, sr, 500, 3000)
    pump = _band_energy(pcm, sr, 20, 200)
    return float(np.clip(leak * 1.8 - pump * 1.3, 0.0, 1.0))


def _stub_rf(pcm: np.ndarray, sr: int) -> float:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms < 0.005:
        return 0.08
    hf = _band_energy(pcm, sr, 2000, 4000)
    lf = _band_energy(pcm, sr, 20, 200)
    return float(np.clip(hf * 2.8 - lf * 1.0, 0.0, 1.0))


def _stub_cnn(pcm: np.ndarray, sr: int) -> float:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms < 0.005:
        return 0.07
    mid = _band_energy(pcm, sr, 600, 3500)
    lf  = _band_energy(pcm, sr, 20, 200)
    return float(np.clip(mid * 2.4 - lf * 1.5, 0.0, 1.0))


def _stub_isolation_forest(pcm: np.ndarray, sr: int) -> float:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms < 0.005:
        return 0.05
    return float(np.clip(_band_energy(pcm, sr, 3000, 7000) * 2.5, 0.0, 1.0))


def _stub_ood(pcm: np.ndarray, sr: int) -> float:
    rms = float(np.sqrt(np.mean(pcm ** 2)))
    if rms < 0.005:
        return 0.15
    leak = _band_energy(pcm, sr, 500, 3000)
    pump = _band_energy(pcm, sr, 20, 200)
    d_leak = abs(leak - 0.45) + abs(pump - 0.05)
    d_pump = abs(leak - 0.10) + abs(pump - 0.35)
    return float(min(d_leak, d_pump) * 3.0)


# ─── Async head wrappers ──────────────────────────────────────────────────────

async def head_xgb(pcm: np.ndarray, sr: int, feat: np.ndarray | None = None) -> float:
    await asyncio.sleep(0.001)
    if _xgb_session is not None and feat is not None:
        return _onnx_predict(_xgb_session, feat)
    return _stub_xgb(pcm, sr)


async def head_rf(pcm: np.ndarray, sr: int, feat: np.ndarray | None = None) -> float:
    await asyncio.sleep(0.001)
    if _rf_session is not None and feat is not None:
        return _onnx_predict(_rf_session, feat)
    return _stub_rf(pcm, sr)


async def head_cnn(pcm: np.ndarray, sr: int, feat: np.ndarray | None = None) -> float:
    """CNN head — calls IEP4 /classify_raw over HTTP.

    Falls back to the physics stub on:
      - IEP4 not configured (IEP4_URL empty or unreachable)
      - HTTP 503 (model not yet trained)
      - Any exception or timeout

    The PCM is sent as raw float32 bytes in base64 to avoid WAV encoding
    overhead.  IEP4 must expose a /classify_raw endpoint accepting:
        {"pcm_b64": "<base64 float32 bytes>", "sr": 16000}
    and returning:
        {"p_leak": 0.0–1.0, "label": "...", "confidence": ..., "is_ood": ...}
    """
    if not IEP4_URL:
        return _stub_cnn(pcm, sr)

    try:
        import httpx
        pcm_b64 = base64.b64encode(pcm.astype(np.float32).tobytes()).decode()
        async with httpx.AsyncClient(timeout=0.115) as client:
            resp = await client.post(
                f"{IEP4_URL}/classify_raw",
                json={"pcm_b64": pcm_b64, "sr": sr},
            )
            if resp.status_code == 200:
                data = resp.json()
                p_leak = float(data.get("p_leak", 0.5))
                log.debug(
                    "IEP4 CNN: p_leak=%.3f label=%s is_ood=%s",
                    p_leak, data.get("label"), data.get("is_ood"),
                )
                return float(np.clip(p_leak, 0.0, 1.0))
            elif resp.status_code == 503:
                log.debug("IEP4 not yet trained (503) — using CNN stub")
            else:
                log.debug("IEP4 returned %d — using CNN stub", resp.status_code)
    except ImportError:
        log.debug("httpx not installed — using CNN stub")
    except Exception as exc:
        log.debug("IEP4 CNN call failed: %s — using stub", exc)

    return _stub_cnn(pcm, sr)


async def head_isolation_forest(
    pcm: np.ndarray, sr: int, feat: np.ndarray | None = None
) -> float:
    await asyncio.sleep(0.005)
    return _stub_isolation_forest(pcm, sr)


async def head_ood(pcm: np.ndarray, sr: int, feat: np.ndarray | None = None) -> float:
    await asyncio.sleep(0.008)
    return _stub_ood(pcm, sr)


# ─── Fan-out with budgets ─────────────────────────────────────────────────────

async def _with_budget(
    name: str,
    coro: asyncio.Coroutine[None, None, float],
    fallback: float,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    try:
        v = await asyncio.wait_for(coro, timeout=HEAD_BUDGET_MS[name] / 1000)
    except TimeoutError:
        log.warning("head=%s timed out, using fallback=%.3f", name, fallback)
        v = fallback
    return v, (time.perf_counter() - t0) * 1000


# ─── Main frame handler ───────────────────────────────────────────────────────

async def handle_frame(payload: dict) -> None:
    with tracer.start_as_current_span("eep.handle_frame") as span:
        frame = AcousticFrame(**payload)
        pcm   = _decode_pcm(frame.pcm_b64, frame.n_samples)
        sr    = frame.sample_rate_hz

        span.set_attribute("frame_id", str(frame.frame_id))
        span.set_attribute("sensor_id", frame.sensor_id)
        span.set_attribute("site_id", frame.site_id)

        # ── Feed raw PCM into the TDOA cache so spatial.fusion can cross- ─
        # ── correlate simultaneous frames from neighbouring sensors.       ─
        # This must happen before any await so the cache is populated even
        # if a later await raises; captured_at is the edge timestamp.
        try:
            log.debug("Caching PCM for sensor %s", frame.sensor_id)
            _cache_pcm(frame.sensor_id, pcm, sr, frame.captured_at)
        except Exception as exc:
            log.error("PCM cache failed: %s", exc)
            pass   # never block detection on cache failures

        # ── Resolve live SCADA pressure for this site ─────────────────────
        pressure_bar = _get_site_pressure(frame.site_id)
        span.set_attribute("pressure_bar", pressure_bar)

        # Resolve pipe material from sensor twin (default PVC until wired)
        pipe_material = "PVC"

        # Extract DSP features once — shared across all heads
        feat: np.ndarray | None = None
        if _xgb_session is not None or _rf_session is not None:
            try:
                feat = extract_features_with_meta(
                    pcm, sr=sr,
                    pipe_material=pipe_material,
                    pressure_bar=pressure_bar,   # ← SCADA-sourced, not hardcoded 3.0
                )
            except Exception as exc:
                log.debug("Feature extraction failed: %s — using stubs", exc)

        # ── Fan-out to all heads in parallel ──────────────────────────────
        with tracer.start_as_current_span("eep.fanout"):
            (xgb, t_xgb), (rf, t_rf), (cnn, t_cnn), (if_, t_if), (ood, t_ood) = (
                await asyncio.gather(
                    _with_budget("xgb", head_xgb(pcm, sr, feat),               fallback=0.3),
                    _with_budget("rf",  head_rf(pcm, sr, feat),                 fallback=0.3),
                    _with_budget("cnn", head_cnn(pcm, sr, feat),                fallback=0.3),
                    _with_budget("if",  head_isolation_forest(pcm, sr, feat),   fallback=0.1),
                    _with_budget("ood", head_ood(pcm, sr, feat),                fallback=0.5),
                )
            )

        span.set_attribute("xgb_p_leak", xgb)
        span.set_attribute("rf_p_leak", rf)
        span.set_attribute("cnn_p_leak", cnn)

        # ── Weighted fusion ───────────────────────────────────────────────
        fused_raw = (
            FUSION_W["xgb"] * xgb
            + FUSION_W["rf"]  * rf
            + FUSION_W["cnn"] * cnn
            + FUSION_W["if"]  * if_
        )

        # ── SCADA physics multiplier ──────────────────────────────────────
        # High operating pressure amplifies the leak probability proportionally.
        # This bridges SCADA data into the ML decision path.
        pressure_mult = _pressure_leak_multiplier(pressure_bar)
        fused = float(np.clip(fused_raw * pressure_mult, 0.0, 1.0))

        if pressure_mult > 1.0:
            log.debug(
                "SCADA pressure boost: site=%s p=%.2f bar mult=%.3f "
                "fused %.3f→%.3f",
                frame.site_id, pressure_bar, pressure_mult, fused_raw, fused,
            )
            span.set_attribute("pressure_mult", pressure_mult)

        uncertainty = float(np.std([xgb, rf, cnn]))
        is_ood      = ood > OOD_THRESHOLD
        is_leak     = (fused >= LEAK_THRESHOLD) and (not is_ood)

        span.set_attribute("fused_p_leak", fused)
        span.set_attribute("is_leak", is_leak)
        span.set_attribute("is_ood", is_ood)

        # ── Feature attribution ───────────────────────────────────────────
        if feat is not None:
            feat_names = [
                "rms_mean", "rms_std", "zcr_mean", "zcr_std",
                "kurtosis", "skewness", "crest_factor",
                "centroid_mean", "centroid_std",
                "rolloff_mean",  "rolloff_std",
                "flatness_mean", "flatness_std",
            ] + [f"mfcc{i//2}_{'mean' if i%2==0 else 'std'}" for i in range(26)] \
              + ["pipe_material_enc", "pressure_bar"]
            # feat may be 39-d (no meta) or 41-d — zip safely
            feat_dict = dict(zip(feat_names, feat.tolist(), strict=False))
            top = sorted(feat_dict.items(), key=lambda kv: -abs(kv[1]))[:3]
        else:
            band_feats = {
                "bandpower_500_3000": _band_energy(pcm, sr, 500, 3000),
                "bandpower_50_200":   _band_energy(pcm, sr, 50, 200),
                "rms":                float(np.sqrt(np.mean(pcm ** 2))),
                "edge_snr_db":        frame.edge_snr_db,
            }
            top = sorted(band_feats.items(), key=lambda kv: -abs(kv[1]))[:3]

        result = DetectionResult(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            site_id=frame.site_id,
            captured_at=frame.captured_at,
            xgb_p_leak=xgb,
            rf_p_leak=rf,
            cnn_p_leak=cnn,
            if_anomaly_score=if_,
            ood_score=ood,
            fused_p_leak=fused,
            fused_uncertainty=uncertainty,
            is_leak=is_leak,
            is_ood=is_ood,
            top_shap_features=top,
            latency_ms={
                "xgb": t_xgb,
                "rf":  t_rf,
                "cnn": t_cnn,
                "if":  t_if,
                "ood": t_ood,
            },
        )
        log.debug("Publishing detection: sensor=%s leak=%s p=%.3f", result.sensor_id, result.is_leak, result.fused_p_leak)
        await get_bus().publish(Topics.DETECTION, result)


def wire() -> None:
    _load_models()
    get_bus().subscribe(Topics.ACOUSTIC_FRAME, handle_frame)
    # Subscribe to SCADA readings so the pressure cache stays current
    get_bus().subscribe(Topics.SCADA_READING, _handle_scada_reading)
    log.info("EEP orchestrator wired (SCADA-aware, IEP4_URL=%s)", IEP4_URL or "unset")
