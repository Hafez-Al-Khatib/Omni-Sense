"""
Out-of-Distribution Detector
==============================
Wraps the Isolation Forest model for OOD/anomaly detection.

Supports ONNX (preferred) and joblib fallback. The joblib fallback is used
when ONNX is miscalibrated (e.g., offset_ mismatch from sklearn version drift).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.ood")

# Default model paths (relative to the app's working directory)
ONNX_PATH = Path("models/isolation_forest.onnx")
JOBLIB_PATH = Path("models/isolation_forest.joblib")


class OODDetector:
    """Isolation Forest–based Out-of-Distribution detector (ONNX + joblib fallback)."""

    def __init__(self):
        self._session = None
        self._joblib_model = None
        self._backend: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self, onnx_path: str | None = None, joblib_path: str | None = None):
        """Load the Isolation Forest model.

        Tries ONNX first, then falls back to joblib if ONNX is miscalibrated.
        """
        onnx_p = Path(onnx_path) if onnx_path else ONNX_PATH
        joblib_p = Path(joblib_path) if joblib_path else JOBLIB_PATH

        # Try ONNX first
        if onnx_p.exists():
            try:
                self._load_onnx(onnx_p)
                # Validate ONNX is not miscalibrated (all-negative scores)
                if self._validate_onnx():
                    logger.info("ONNX model passed validation.")
                    return
                else:
                    logger.warning("ONNX model is miscalibrated (all scores negative). Trying joblib fallback.")
                    self._session = None
                    self._backend = None
            except Exception as exc:
                logger.warning(f"ONNX load failed: {exc}. Trying joblib fallback.")
                self._session = None
                self._backend = None

        # Fallback to joblib
        if joblib_p.exists():
            self._load_joblib(joblib_p)
            return

        raise FileNotFoundError(
            f"No OOD model found at {onnx_p} or {joblib_p}. "
            "Run scripts/train_models.py before deploying."
        )

    def _load_onnx(self, path: Path) -> None:
        """Load ONNX model via ONNX Runtime."""
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._backend = "onnx"
        logger.info(f"Loaded Isolation Forest (ONNX) from {path}")

    def _load_joblib(self, path: Path) -> None:
        """Load joblib model as fallback."""
        import joblib

        self._joblib_model = joblib.load(path)
        self._backend = "joblib"
        logger.info(f"Loaded Isolation Forest (joblib fallback) from {path}")

    def _validate_onnx(self) -> bool:
        """Quick sanity check: ONNX should produce at least one positive score on a zero vector."""
        try:
            test_input = np.zeros((1, self._session.get_inputs()[0].shape[1]), dtype=np.float32)
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: test_input})
            score_val = float(result[1].flatten()[0])
            return score_val >= -0.05  # Allow slightly negative; reject catastrophically negative
        except Exception as exc:
            logger.warning(f"ONNX validation failed: {exc}")
            return False

    def score(self, embedding: np.ndarray) -> float:
        """Return the decision-function anomaly score.

        Higher = more normal; lower (more negative) = more anomalous.
        """
        if not self.is_loaded:
            raise RuntimeError("OOD model not loaded — call load() first")

        embedding_2d = embedding.reshape(1, -1).astype(np.float32)

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: embedding_2d})
            if len(result) < 2:
                logger.warning("ONNX model returned fewer than 2 outputs. Using fallback score.")
                return 1.0 if result[0][0][0] == 1 else -1.0
            score_val = result[1].flatten()[0]
            return float(score_val)
        else:
            # joblib fallback
            return float(self._joblib_model.decision_function(embedding_2d)[0])

    def is_anomalous(
        self,
        embedding: np.ndarray,
        threshold_override: float | None = None,
    ) -> bool:
        """Return True if the embedding is Out-of-Distribution."""
        if not self.is_loaded:
            raise RuntimeError("OOD model not loaded — call load() first")

        if threshold_override is not None:
            return self.score(embedding) < threshold_override

        if self._backend == "onnx":
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: embedding_2d})
            return int(result[0][0]) == -1
        else:
            # joblib fallback uses sklearn's built-in predict
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            return int(self._joblib_model.predict(embedding_2d)[0]) == -1
