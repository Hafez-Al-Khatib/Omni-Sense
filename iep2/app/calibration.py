"""
Dynamic Calibration Manager
==============================
Manages OOD threshold adjustments based on ambient environment recordings.
Used for the POST /calibrate endpoint to adapt the system to new deployment sites.
"""

import logging
import threading
from datetime import datetime

import numpy as np

logger = logging.getLogger("iep2.calibration")


class CalibrationManager:
    """
    Thread-safe manager for dynamic OOD threshold calibration.

    The default threshold is None (use model's built-in boundary).
    After calibration, the threshold is set to:
        mean(ambient_scores) - 2 * std(ambient_scores)

    This ensures that the system accepts sounds similar to the ambient
    environment while still flagging truly anomalous inputs.
    """

    def __init__(self):
        self._threshold: float | None = None
        self._calibrated_at: datetime | None = None
        self._lock = threading.Lock()

    def get_threshold(self) -> float | None:
        """Get the current calibration threshold (None = use default)."""
        with self._lock:
            return self._threshold

    def calibrate(
        self,
        ambient_scores: list[float],
        n_sigma: float = 2.0,
    ) -> float:
        """
        Set a new OOD threshold based on ambient acoustic scores.

        The threshold is set to: mean - n_sigma * std
        Anything below this threshold is considered OOD.

        Args:
            ambient_scores: List of IF decision_function scores from ambient recordings
            n_sigma: Number of standard deviations below the mean for the threshold

        Returns:
            The new threshold value
        """
        scores = np.array(ambient_scores, dtype=np.float64)

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Threshold: anything more anomalous than n_sigma below ambient mean
        new_threshold = mean_score - n_sigma * std_score

        with self._lock:
            self._threshold = new_threshold
            self._calibrated_at = datetime.utcnow()

        logger.info(
            f"Calibrated: mean={mean_score:.4f}, std={std_score:.4f}, "
            f"threshold={new_threshold:.4f} (n_sigma={n_sigma})"
        )

        return new_threshold

    def reset(self):
        """Reset to default (model built-in) threshold."""
        with self._lock:
            self._threshold = None
            self._calibrated_at = None
        logger.info("Calibration reset to default.")

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._threshold is not None

    @property
    def calibrated_at(self) -> datetime | None:
        with self._lock:
            return self._calibrated_at
