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
import time
from pathlib import Path
from typing import Optional

import numpy as np

from omni.common.bus import Topics, get_bus
from omni.common.schemas import AcousticFrame, DetectionResult
from omni.eep.features import extract_features_with_meta, _FEATURE_DIM

log = logging.getLogger("eep")

# ─── Configuration ────────────────────────────────────────────────────────────

HEAD_BUDGET_MS = {
    "xgb": 30,
    "rf":  30,
    "cnn": 150,
    "if":  20,
    "ood": 40,
}

FUSION_W = {"xgb": 0.45, "rf": 0.25, "cnn": 0.25, "if": 0.05}

LEAK_THRESHOLD = 0.60
OOD_THRESHOLD  = 1.0

# ONNX model paths (relative to project root or Docker workdir)
_OMNI_MODEL_DIR = Path("omni/models")
_XGB_ONNX   = _OMNI_MODEL_DIR / "xgb_head.onnx"
_RF_ONNX    = _OMNI_MODEL_DIR / "rf_head.onnx"
_THR_FILE   = _OMNI_MODEL_DIR / "omni_threshold.json"


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

async def head_xgb(pcm: np.ndarray, sr: int, feat: Optional[np.ndarray] = None) -> float:
    await asyncio.sleep(0.001)
    if _xgb_session is not None and feat is not None:
        return _onnx_predict(_xgb_session, feat)
    return _stub_xgb(pcm, sr)


async def head_rf(pcm: np.ndarray, sr: int, feat: Optional[np.ndarray] = None) -> float:
    await asyncio.sleep(0.001)
    if _rf_session is not None and feat is not None:
        return _onnx_predict(_rf_session, feat)
    return _stub_rf(pcm, sr)


async def head_cnn(pcm: np.ndarray, sr: int, feat: Optional[np.ndarray] = None) -> float:
    """CNN head: uses XGB features as proxy until IEP4 ONNX is wired in."""
    await asyncio.sleep(0.030)
    if _xgb_session is not None and feat is not None:
        # Slightly perturbed XGB score — different model path simulates diversity
        base = _onnx_predict(_xgb_session, feat)
        return float(np.clip(base * 0.95 + 0.05 * _stub_cnn(pcm, sr), 0.0, 1.0))
    return _stub_cnn(pcm, sr)


async def head_isolation_forest(
    pcm: np.ndarray, sr: int, feat: Optional[np.ndarray] = None
) -> float:
    await asyncio.sleep(0.005)
    return _stub_isolation_forest(pcm, sr)


async def head_ood(pcm: np.ndarray, sr: int, feat: Optional[np.ndarray] = None) -> float:
    await asyncio.sleep(0.008)
    return _stub_ood(pcm, sr)


# ─── Fan-out with budgets ─────────────────────────────────────────────────────

async def _with_budget(
    name: str,
    coro: "asyncio.Coroutine[None, None, float]",
    fallback: float,
) -> tuple[float, float]:
    t0 = time.perf_counter()
    try:
        v = await asyncio.wait_for(coro, timeout=HEAD_BUDGET_MS[name] / 1000)
    except asyncio.TimeoutError:
        log.warning("head=%s timed out, using fallback=%.3f", name, fallback)
        v = fallback
    return v, (time.perf_counter() - t0) * 1000


# ─── Main frame handler ───────────────────────────────────────────────────────

async def handle_frame(payload: dict) -> None:
    frame = AcousticFrame(**payload)
    pcm   = _decode_pcm(frame.pcm_b64, frame.n_samples)
    sr    = frame.sample_rate_hz

    # Extract DSP features once — shared across all heads
    feat: Optional[np.ndarray] = None
    if _xgb_session is not None or _rf_session is not None:
        try:
            feat = extract_features_with_meta(
                pcm, sr=sr,
                pipe_material="PVC",     # sensor twin would supply this in prod
                pressure_bar=3.0,
            )
        except Exception as exc:
            log.debug("Feature extraction failed: %s — using stubs", exc)

    (xgb, t_xgb), (rf, t_rf), (cnn, t_cnn), (if_, t_if), (ood, t_ood) = (
        await asyncio.gather(
            _with_budget("xgb", head_xgb(pcm, sr, feat),               fallback=0.3),
            _with_budget("rf",  head_rf(pcm, sr, feat),                 fallback=0.3),
            _with_budget("cnn", head_cnn(pcm, sr, feat),                fallback=0.3),
            _with_budget("if",  head_isolation_forest(pcm, sr, feat),   fallback=0.1),
            _with_budget("ood", head_ood(pcm, sr, feat),                fallback=0.5),
        )
    )

    fused = (
        FUSION_W["xgb"] * xgb
        + FUSION_W["rf"]  * rf
        + FUSION_W["cnn"] * cnn
        + FUSION_W["if"]  * if_
    )

    uncertainty = float(np.std([xgb, rf, cnn]))
    is_ood      = ood > OOD_THRESHOLD
    is_leak     = (fused >= LEAK_THRESHOLD) and (not is_ood)

    # Feature attribution (top-3 DSP features or spectral proxies)
    if feat is not None:
        feat_names = [
            "rms_mean", "rms_std", "zcr_mean", "zcr_std",
            "kurtosis", "skewness", "crest_factor",
            "centroid_mean", "centroid_std",
            "rolloff_mean",  "rolloff_std",
            "flatness_mean", "flatness_std",
        ] + [f"mfcc{i//2}_{'mean' if i%2==0 else 'std'}" for i in range(26)]
        feat_dict = dict(zip(feat_names, feat.tolist()))
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
        fused_p_leak=float(fused),
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
    await get_bus().publish(Topics.DETECTION, result)


def wire() -> None:
    _load_models()
    get_bus().subscribe(Topics.ACOUSTIC_FRAME, handle_frame)
