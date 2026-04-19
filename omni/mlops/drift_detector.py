"""Input drift detector for the EEP v2 pipeline.

Monitors the distribution of incoming acoustic features (band-energy ratios,
RMS, OOD scores) against a reference window captured at training time.  When
the Population Stability Index (PSI) or cosine drift exceeds a configurable
threshold, it fires a DRIFT_ALERT event on the bus, which the retraining
trigger picks up.

Metrics tracked
---------------
- PSI (Population Stability Index) on each feature dimension
- Cosine similarity to reference centroid
- Rolling mean/std of fused_p_leak predictions
- Fraction of OOD-flagged frames (ood_rate)

PSI interpretation
------------------
  PSI < 0.10  → no significant drift
  PSI < 0.25  → moderate drift, monitor
  PSI ≥ 0.25  → significant drift → trigger retraining
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from omni.common.bus import Topics, get_bus
from omni.common.schemas import DetectionResult

log = logging.getLogger("drift")

PSI_WARN_THRESHOLD   = 0.10
PSI_RETRAIN_THRESHOLD = 0.25
OOD_RATE_THRESHOLD   = 0.15   # >15% OOD frames → distribution shift
WINDOW_SIZE          = 200    # detections in sliding window
MIN_WINDOW_FOR_TEST  = 50     # don't test until we have this many samples
N_BINS               = 10


@dataclass
class DriftReport:
    evaluated_at: datetime
    n_samples: int
    psi_xgb: float
    psi_rf: float
    psi_fused: float
    psi_ood: float
    psi_max: float
    cosine_similarity: float
    ood_rate: float
    mean_p_leak: float
    std_p_leak: float
    drift_level: str        # "none" | "moderate" | "significant"
    should_retrain: bool
    summary: str


class DriftDetector:
    """Sliding-window drift detector wired to the detection bus topic."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window: deque[DetectionResult] = deque(maxlen=window_size)
        self._reference: np.ndarray | None = None   # (n_ref, 4) matrix
        self._reference_centroid: np.ndarray | None = None
        self._lock = asyncio.Lock()
        self.latest_report: DriftReport | None = None
        self._report_counter: int = 0

    # ── Reference ────────────────────────────────────────────────────────────

    def set_reference(self, detections: list[DetectionResult]) -> None:
        """Call once at startup with recent training-distribution detections."""
        mat = self._to_matrix(detections)
        self._reference = mat
        self._reference_centroid = mat.mean(axis=0)
        log.info("drift reference set: %d samples, shape %s", len(detections), mat.shape)

    def _to_matrix(self, detections: list[DetectionResult]) -> np.ndarray:
        return np.array([
            [d.xgb_p_leak, d.rf_p_leak, d.fused_p_leak, d.ood_score]
            for d in detections
        ], dtype=np.float64)

    # ── PSI ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _psi_1d(ref: np.ndarray, cur: np.ndarray, n_bins: int = N_BINS) -> float:
        """Population Stability Index for one feature dimension.

        Uses Laplace smoothing (α = 0.5 per bin) instead of a hard floor so
        that empty bins in small samples don't produce degenerate log ratios.
        This is the standard production-grade approach for monitoring models
        on data slices where some bins may be sparsely populated.
        """
        lo = min(ref.min(), cur.min()) - 1e-9
        hi = max(ref.max(), cur.max()) + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)

        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)

        # Laplace smoothing: prevents log(0) and reduces noise for small samples
        alpha = 0.5
        ref_pct = (ref_counts + alpha) / (len(ref) + alpha * n_bins)
        cur_pct = (cur_counts + alpha) / (len(cur) + alpha * n_bins)

        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    # ── Cosine similarity ─────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    # ── Evaluate ─────────────────────────────────────────────────────────────

    async def evaluate(self) -> DriftReport | None:
        async with self._lock:
            window_list = list(self._window)

        if len(window_list) < MIN_WINDOW_FOR_TEST:
            return None
        if self._reference is None:
            return None

        cur = self._to_matrix(window_list)
        ref = self._reference

        psi_xgb   = self._psi_1d(ref[:, 0], cur[:, 0])
        psi_rf    = self._psi_1d(ref[:, 1], cur[:, 1])
        psi_fused = self._psi_1d(ref[:, 2], cur[:, 2])
        psi_ood   = self._psi_1d(ref[:, 3], cur[:, 3])
        psi_max   = max(psi_xgb, psi_rf, psi_fused, psi_ood)

        cur_centroid = cur.mean(axis=0)
        cosine_sim   = self._cosine(self._reference_centroid, cur_centroid)

        ood_rate   = float(np.mean([d.is_ood for d in window_list]))
        mean_p     = float(np.mean(cur[:, 2]))
        std_p      = float(np.std(cur[:, 2]))

        if psi_max >= PSI_RETRAIN_THRESHOLD or ood_rate >= OOD_RATE_THRESHOLD:
            level = "significant"
            retrain = True
        elif psi_max >= PSI_WARN_THRESHOLD:
            level = "moderate"
            retrain = False
        else:
            level = "none"
            retrain = False

        summary = (
            f"PSI_max={psi_max:.3f} (xgb={psi_xgb:.3f} rf={psi_rf:.3f} "
            f"fused={psi_fused:.3f} ood={psi_ood:.3f}) "
            f"cosine={cosine_sim:.3f} ood_rate={ood_rate:.2%} "
            f"mean_p={mean_p:.3f}±{std_p:.3f}"
        )

        report = DriftReport(
            evaluated_at=datetime.now(UTC),
            n_samples=len(window_list),
            psi_xgb=psi_xgb,
            psi_rf=psi_rf,
            psi_fused=psi_fused,
            psi_ood=psi_ood,
            psi_max=psi_max,
            cosine_similarity=cosine_sim,
            ood_rate=ood_rate,
            mean_p_leak=mean_p,
            std_p_leak=std_p,
            drift_level=level,
            should_retrain=retrain,
            summary=summary,
        )
        self.latest_report = report
        return report

    # ── Bus handler ───────────────────────────────────────────────────────────

    async def on_detection(self, payload: dict) -> None:
        det = DetectionResult(**payload)
        async with self._lock:
            self._window.append(det)
        self._report_counter += 1
        # Re-evaluate every 25 detections
        if self._report_counter % 25 == 0:
            report = await self.evaluate()
            if report:
                log.info("drift [%s] %s", report.drift_level.upper(), report.summary)
                if report.should_retrain:
                    log.warning(
                        "DRIFT THRESHOLD EXCEEDED — publishing retrain request"
                    )
                    await get_bus().publish(
                        "mlops.retrain.request.v1",
                        {
                            "triggered_at": report.evaluated_at.isoformat(),
                            "psi_max": report.psi_max,
                            "ood_rate": report.ood_rate,
                            "drift_level": report.drift_level,
                            "n_samples": report.n_samples,
                        },
                    )

    def wire(self) -> None:
        get_bus().subscribe(Topics.DETECTION, self.on_detection)


# Module-level singleton
_detector: DriftDetector | None = None


def get_detector() -> DriftDetector:
    global _detector
    if _detector is None:
        _detector = DriftDetector()
    return _detector
