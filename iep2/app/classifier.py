"""
Leak Classifier
=================
Wraps the XGBoost model for leak vs background classification.
Supports both ONNX Runtime and joblib-loaded models.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.classifier")

ONNX_PATH = Path("models/xgboost_classifier.onnx")
JOBLIB_PATH = Path("models/xgboost_classifier.joblib")

# Must match training-time encoding
PIPE_MATERIAL_MAP = {
    "PVC": 0,
    "Steel": 1,
    "Cast_Iron": 2,
}

LABEL_MAP = {
    0: "background",
    1: "leak",
}


class LeakClassifier:
    """XGBoost-based leak classifier with metadata features."""

    def __init__(self):
        self._model = None
        self._session = None
        self._backend = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, onnx_path: str | None = None, joblib_path: str | None = None):
        """Load XGBoost model. Prefers ONNX; falls back to joblib."""
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
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._backend = "onnx"
        self._is_loaded = True
        logger.info(f"Loaded XGBoost (ONNX) from {path}")

    def _load_sklearn(self, path: Path):
        import joblib

        self._model = joblib.load(path)
        self._backend = "sklearn"
        self._is_loaded = True
        logger.info(f"Loaded XGBoost (joblib) from {path}")

    def _encode_metadata(
        self,
        pipe_material: str,
        pressure_bar: float,
    ) -> np.ndarray:
        """Encode metadata features to match training schema."""
        pipe_code = PIPE_MATERIAL_MAP.get(pipe_material, 0)
        return np.array([pipe_code, pressure_bar], dtype=np.float32)

    def predict(
        self,
        embedding: np.ndarray,
        pipe_material: str = "PVC",
        pressure_bar: float = 3.0,
    ) -> dict:
        """
        Classify an acoustic embedding with metadata context.

        Args:
            embedding: 1024-d float32 YAMNet embedding
            pipe_material: "PVC", "Steel", or "Cast_Iron"
            pressure_bar: Pipe pressure in bar

        Returns:
            dict with keys: label, confidence, probabilities
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        # Build feature vector: [embedding_0..1023, pipe_material, pressure_bar]
        metadata = self._encode_metadata(pipe_material, pressure_bar)
        features = np.concatenate([embedding, metadata]).reshape(1, -1).astype(np.float32)

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            result = self._session.run(None, {input_name: features})
            # ONNX XGBoost outputs: [labels, probabilities]
            label_idx = int(result[0][0])
            proba_dict = result[1][0]
            # Convert ONNX probability output to simple dict
            if isinstance(proba_dict, dict):
                probabilities = {
                    LABEL_MAP.get(int(k), str(k)): float(v)
                    for k, v in proba_dict.items()
                }
            else:
                # Handle array output
                probabilities = {
                    LABEL_MAP[i]: float(p)
                    for i, p in enumerate(proba_dict)
                }
        else:
            proba = self._model.predict_proba(features)[0]
            label_idx = int(self._model.predict(features)[0])
            probabilities = {
                LABEL_MAP[i]: float(p)
                for i, p in enumerate(proba)
            }

        label = LABEL_MAP.get(label_idx, "unknown")
        confidence = max(probabilities.values())

        return {
            "label": label,
            "confidence": float(confidence),
            "probabilities": probabilities,
        }
