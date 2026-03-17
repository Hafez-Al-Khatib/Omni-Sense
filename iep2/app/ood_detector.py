"""
Out-of-Distribution Detector
==============================
Wraps the Isolation Forest model for OOD/anomaly detection.
Supports both ONNX Runtime and joblib-loaded scikit-learn models.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.ood")

# Default model paths (relative to the app's working directory)
ONNX_PATH = Path("models/isolation_forest.onnx")
JOBLIB_PATH = Path("models/isolation_forest.joblib")


class OODDetector:
    """Isolation Forest–based Out-of-Distribution detector."""

    def __init__(self):
        self._model = None
        self._session = None
        self._backend = None  # "onnx" or "sklearn"
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, onnx_path: str | None = None, joblib_path: str | None = None):
        """
        Load the Isolation Forest model.
        Prefers ONNX; falls back to joblib/sklearn.
        """
        onnx_p = Path(onnx_path) if onnx_path else ONNX_PATH
        joblib_p = Path(joblib_path) if joblib_path else JOBLIB_PATH

        if onnx_p.exists():
            self._load_onnx(onnx_p)
        elif joblib_p.exists():
            self._load_sklearn(joblib_p)
        else:
            raise FileNotFoundError(
                f"No model found at {onnx_p} or {joblib_p}. "
                "Train models first with scripts/train_models.py."
            )

    def _load_onnx(self, path: Path):
        """Load ONNX model via ONNX Runtime."""
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._backend = "onnx"
        self._is_loaded = True
        logger.info(f"Loaded Isolation Forest (ONNX) from {path}")

    def _load_sklearn(self, path: Path):
        """Load scikit-learn model via joblib."""
        import joblib

        self._model = joblib.load(path)
        self._backend = "sklearn"
        self._is_loaded = True
        logger.info(f"Loaded Isolation Forest (sklearn) from {path}")

    def score(self, embedding: np.ndarray) -> float:
        """
        Compute the anomaly score for a single embedding.

        Returns:
            float: Higher is more normal; lower (more negative) = more anomalous
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        embedding_2d = embedding.reshape(1, -1).astype(np.float32)

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: embedding_2d})
            # ONNX Isolation Forest outputs: [labels, scores]
            # scores is the decision_function output
            return float(result[1][0][1])  # score for "inlier" class
        else:
            return float(self._model.decision_function(embedding_2d)[0])

    def is_anomalous(
        self,
        embedding: np.ndarray,
        threshold_override: float | None = None,
    ) -> bool:
        """
        Check if an embedding is Out-of-Distribution.

        If threshold_override is given, use it instead of the model's
        default decision boundary (0 for sklearn IF).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        embedding_2d = embedding.reshape(1, -1).astype(np.float32)

        if threshold_override is not None:
            score = self.score(embedding)
            return score < threshold_override

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: embedding_2d})
            label = int(result[0][0])
            return label == -1
        else:
            prediction = self._model.predict(embedding_2d)[0]
            return prediction == -1
