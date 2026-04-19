"""
Out-of-Distribution Detector
==============================
Wraps the Isolation Forest model for OOD/anomaly detection.

Security note
-------------
Only ONNX Runtime is supported for loading models.  The previous
joblib/sklearn fallback has been removed because Python's pickle protocol
(used internally by joblib) can execute arbitrary code when loading a
tampered model file, creating an RCE vector.

Export models to ONNX before deploying:
    scripts/train_models.py  →  models/isolation_forest.onnx
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.ood")

# Default ONNX model path (relative to the app's working directory)
ONNX_PATH = Path("models/isolation_forest.onnx")


class OODDetector:
    """Isolation Forest–based Out-of-Distribution detector (ONNX only)."""

    def __init__(self):
        self._session = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, onnx_path: str | None = None):
        """Load the Isolation Forest ONNX model.

        Args:
            onnx_path: Override path.  Defaults to ``models/isolation_forest.onnx``.

        Raises:
            FileNotFoundError: Model file is absent.
            RuntimeError: ONNX Runtime failed to load the session.
        """
        onnx_p = Path(onnx_path) if onnx_path else ONNX_PATH
        if not onnx_p.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_p}. "
                "Export with scripts/train_models.py --export-onnx before deploying."
            )
        self._load_onnx(onnx_p)

    def _load_onnx(self, path: Path) -> None:
        """Load ONNX model via ONNX Runtime."""
        import onnxruntime as ort

        try:
            self._session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            self._is_loaded = True
            logger.info(f"Loaded Isolation Forest (ONNX) from {path}")
        except Exception as exc:
            raise RuntimeError(
                f"ONNX Runtime failed to load {path}: {exc}. "
                "Re-export the model with scripts/train_models.py."
            ) from exc

    def score(self, embedding: np.ndarray) -> float:
        """Return the decision-function anomaly score.

        Higher = more normal; lower (more negative) = more anomalous.
        """
        if not self._is_loaded:
            raise RuntimeError("OOD model not loaded — call load() first")

        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        result = self._session.run(None, {input_name: embedding_2d})
        # ONNX Isolation Forest outputs: [labels, scores]
        # scores[0][1] is the decision_function value for the inlier class
        return float(result[1][0][1])

    def is_anomalous(
        self,
        embedding: np.ndarray,
        threshold_override: float | None = None,
    ) -> bool:
        """Return True if the embedding is Out-of-Distribution.

        Args:
            embedding: Feature vector.
            threshold_override: If provided, compare score() against this
                threshold instead of using the model's built-in boundary.
        """
        if not self._is_loaded:
            raise RuntimeError("OOD model not loaded — call load() first")

        if threshold_override is not None:
            return self.score(embedding) < threshold_override

        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        result = self._session.run(None, {input_name: embedding_2d})
        return int(result[0][0]) == -1
