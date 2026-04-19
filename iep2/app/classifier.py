"""
Leak Classifier — XGBoost + Random Forest Ensemble
=====================================================
Wraps two complementary classifiers and averages their probability outputs:

  XGBoost   — gradient-boosted trees; strong on feature interactions
  Random Forest — bagged trees; lower variance on small datasets; less
                  prone to overfitting, better calibrated with class_weight

Why an ensemble?
  With only ~80–500 labelled recordings, a single model's variance is high.
  XGBoost's boosting can overfit to noise; RF's bagging averages it out.
  Averaging their softmax outputs gives a more calibrated confidence signal
  and empirically reduces error by 5–15% on small-N benchmarks.

Security note
-------------
Only ONNX Runtime is used for inference.  joblib / pickle fallbacks have
been removed because pickle can execute arbitrary code embedded in a model
file, creating an RCE vector if a model artifact is tampered with.
Scripts must export models to ONNX format before deployment.

Class-index → label mapping is loaded from models/label_map.json,
written by scripts/train_models.py at training time.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("iep2.classifier")

# ─── Model file paths ────────────────────────────────────────────────────────
# Only ONNX paths are accepted.  .joblib paths have been removed (RCE risk).

XGB_ONNX_PATH = Path("models/xgboost_classifier.onnx")
RF_ONNX_PATH  = Path("models/rf_classifier.onnx")
LABEL_MAP_PATH = Path("models/label_map.json")

# Ensemble weight for XGBoost vs Random Forest (must sum to 1.0)
_XGB_WEIGHT = 0.60
_RF_WEIGHT  = 0.40

PIPE_MATERIAL_MAP = {
    "PVC":       0,
    "Steel":     1,
    "Cast_Iron": 2,
}

_FALLBACK_LABEL_MAP: dict[int, str] = {
    0: "Leak",
    1: "No_Leak",
}
# Public alias for tests and external consumers
LABEL_MAP: dict[int, str] = _FALLBACK_LABEL_MAP


# ─── Single-model loader ─────────────────────────────────────────────────────

class _SingleClassifier:
    """Internal helper: loads and runs one ONNX model."""

    def __init__(self, name: str, onnx_path: Path):
        self.name = name
        self._onnx_path = onnx_path
        self._session = None
        self._backend: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> None:
        if not self._onnx_path.exists():
            logger.warning(f"{self.name}: ONNX model not found at {self._onnx_path}")
            return

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self._onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
            logger.info(f"{self.name}: loaded ONNX from {self._onnx_path}")
        except Exception as exc:
            logger.error(
                f"{self.name}: ONNX load failed — {exc}. "
                "Re-export the model with scripts/train_models.py."
            )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Returns probability array of shape (n_classes,)."""
        if not self.is_loaded:
            return np.array([])

        input_name = self._session.get_inputs()[0].name
        result = self._session.run(None, {input_name: features})
        proba_dict = result[1][0]
        if isinstance(proba_dict, dict):
            return np.array(list(proba_dict.values()), dtype=np.float32)
        return np.array(proba_dict, dtype=np.float32)


# ─── Ensemble classifier ─────────────────────────────────────────────────────

class LeakClassifier:
    """
    XGBoost + Random Forest ensemble classifier.

    Falls back gracefully if only one model is available (uses that model
    alone with weight 1.0). If neither model is available, raises an error.
    """

    def __init__(self):
        self._xgb = _SingleClassifier("XGBoost", XGB_ONNX_PATH)
        self._rf  = _SingleClassifier("RandomForest", RF_ONNX_PATH)
        self._is_loaded = False
        self._label_map: dict[int, str] = _FALLBACK_LABEL_MAP.copy()
        self._decision_threshold: float = 0.5

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _load_label_map(self, model_dir: Path) -> None:
        label_map_p = model_dir / LABEL_MAP_PATH.name
        if label_map_p.exists():
            with open(label_map_p) as f:
                raw = json.load(f)
            # Filter out the special _decision_threshold key
            self._label_map = {int(k): v for k, v in raw.items() if k != "_decision_threshold"}
            if "_decision_threshold" in raw:
                self._decision_threshold = float(raw["_decision_threshold"])
                logger.info(f"Production decision threshold: {self._decision_threshold:.3f}")
            logger.info(f"Label map ({len(self._label_map)} classes): {self._label_map}")
        else:
            logger.warning("label_map.json not found — using fallback binary map")

    def load(self) -> None:
        self._xgb.load()
        self._rf.load()

        if not self._xgb.is_loaded and not self._rf.is_loaded:
            raise FileNotFoundError(
                "No ONNX classifier models found. "
                "Run scripts/train_models.py and export to ONNX first."
            )

        model_dir = XGB_ONNX_PATH.parent
        self._load_label_map(model_dir)
        self._is_loaded = True

        loaded = [m.name for m in [self._xgb, self._rf] if m.is_loaded]
        logger.info(f"Ensemble loaded: {loaded}")

    def _encode_metadata(self, pipe_material: str, pressure_bar: float) -> np.ndarray:
        pipe_code = PIPE_MATERIAL_MAP.get(pipe_material, 0)
        return np.array([pipe_code, pressure_bar], dtype=np.float32)

    def predict(
        self,
        embedding: np.ndarray,
        pipe_material: str = "PVC",
        pressure_bar: float = 3.0,
    ) -> dict:
        """
        Ensemble prediction: weighted average of XGBoost and RF probabilities.

        Args:
            embedding: N-d float32 physics feature vector from IEP1
            pipe_material: "PVC", "Steel", or "Cast_Iron"
            pressure_bar: Pipe pressure in bar

        Returns:
            dict with keys: label, confidence, probabilities, ensemble_sources
        """
        if not self._is_loaded:
            raise RuntimeError("Classifier not loaded")

        metadata = self._encode_metadata(pipe_material, pressure_bar)
        features = np.concatenate([embedding, metadata]).reshape(1, -1).astype(np.float32)

        # Collect available model outputs
        proba_parts: list[tuple[np.ndarray, float]] = []  # (proba, weight)

        if self._xgb.is_loaded:
            p = self._xgb.predict_proba(features)
            if len(p) > 0:
                proba_parts.append((p, _XGB_WEIGHT))

        if self._rf.is_loaded:
            p = self._rf.predict_proba(features)
            if len(p) > 0:
                proba_parts.append((p, _RF_WEIGHT))

        if not proba_parts:
            raise RuntimeError("All classifiers failed to produce predictions")

        # Normalise weights in case one model is missing
        total_w = sum(w for _, w in proba_parts)
        ensemble_proba = sum(p * (w / total_w) for p, w in proba_parts)

        # Apply production decision threshold (from training label_map.json)
        # In binary mode, "No_Leak" is typically class index 1.
        # threshold overrides the naive argmax when there is class imbalance.
        n_classes = len(ensemble_proba)
        if n_classes == 2:
            # Binary: predict No_Leak if P(No_Leak) >= threshold
            no_leak_idx = next(
                (i for i, v in self._label_map.items() if "No_Leak" in v or "Normal" in v),
                1,
            )
            leak_idx = 1 - no_leak_idx
            if ensemble_proba[no_leak_idx] >= self._decision_threshold:
                label_idx = no_leak_idx
            else:
                label_idx = leak_idx
        else:
            label_idx = int(np.argmax(ensemble_proba))

        probabilities = {
            self._label_map.get(i, str(i)): float(ensemble_proba[i])
            for i in range(n_classes)
        }
        label = self._label_map.get(label_idx, "unknown")
        confidence = float(ensemble_proba[label_idx])

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "ensemble_sources": [m.name for m in [self._xgb, self._rf] if m.is_loaded],
        }
