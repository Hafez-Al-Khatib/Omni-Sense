"""
Leak Classifier
=================
Wraps the XGBoost model for multi-class fault classification.
Supports both ONNX Runtime and joblib-loaded models.

Class-index → label mapping is loaded from models/label_map.json,
written by scripts/train_models.py at training time.  This means the
classifier automatically adapts to binary or multi-class models without
any code change — just retrain and the label map updates.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.classifier")

ONNX_PATH = Path("models/xgboost_classifier.onnx")
JOBLIB_PATH = Path("models/xgboost_classifier.joblib")
LABEL_MAP_PATH = Path("models/label_map.json")

# Must match training-time encoding (scripts/train_models.py: PIPE_MATERIAL_MAP)
PIPE_MATERIAL_MAP = {
    "PVC":       0,
    "Steel":     1,
    "Cast_Iron": 2,
}

# Fallback label map used only if label_map.json is missing.
# Matches the original binary baseline so existing deployments keep working.
_FALLBACK_LABEL_MAP: dict[int, str] = {
    0: "background",
    1: "leak",
}


class LeakClassifier:
    """XGBoost-based fault classifier with metadata features."""

    def __init__(self):
        self._model = None
        self._session = None
        self._backend = None
        self._is_loaded = False
        self._label_map: dict[int, str] = _FALLBACK_LABEL_MAP.copy()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _load_label_map(self, model_dir: Path) -> None:
        """Load class-index → label mapping written by train_models.py."""
        label_map_p = model_dir / LABEL_MAP_PATH.name
        if label_map_p.exists():
            with open(label_map_p) as f:
                raw = json.load(f)
            self._label_map = {int(k): v for k, v in raw.items()}
            logger.info(f"Loaded label map ({len(self._label_map)} classes): {self._label_map}")
        else:
            logger.warning(
                f"label_map.json not found at {label_map_p}. "
                "Using fallback binary map {0: background, 1: leak}."
            )

    def load(self, onnx_path: str | None = None, joblib_path: str | None = None):
        """Load XGBoost model and label map. Prefers ONNX; falls back to joblib."""
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

        # Load the label map from the same directory as the model
        self._load_label_map(onnx_p.parent if onnx_p.exists() else joblib_p.parent)

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
        Classify a vibration feature vector with metadata context.

        Args:
            embedding: N-d float32 feature vector (208-d vibration features)
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
            if isinstance(proba_dict, dict):
                probabilities = {
                    self._label_map.get(int(k), str(k)): float(v)
                    for k, v in proba_dict.items()
                }
            else:
                probabilities = {
                    self._label_map.get(i, str(i)): float(p)
                    for i, p in enumerate(proba_dict)
                }
        else:
            proba = self._model.predict_proba(features)[0]
            label_idx = int(self._model.predict(features)[0])
            probabilities = {
                self._label_map.get(i, str(i)): float(p)
                for i, p in enumerate(proba)
            }

        label = self._label_map.get(label_idx, "unknown")
        confidence = max(probabilities.values())

        return {
            "label": label,
            "confidence": float(confidence),
            "probabilities": probabilities,
        }
