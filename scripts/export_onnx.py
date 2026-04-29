"""
ONNX Export Script — converts existing joblib model artifacts to ONNX.

Run this inside the iep2 Linux container where onnxmltools works:

    docker compose run --rm iep2 python /app/../scripts/export_onnx.py \
        --models-dir /app/models

On Windows, onnxmltools/xgboost have a DLL conflict that causes a segfault.
This script is safe to run on Linux (Docker container) regardless.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


def _n_features(model) -> int:
    """Read feature count from fitted model (sklearn API)."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    raise AttributeError(f"Cannot determine n_features_in_ from {type(model)}")


def export_xgboost(model, output_path: Path, n_features: int) -> None:
    import xgboost as xgb
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    update_registered_converter(
        xgb.XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  XGBoost ONNX → {output_path}")


def export_rf(model, output_path: Path, n_features: int) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Random Forest ONNX → {output_path}")


def export_isolation_forest(model, output_path: Path, n_features: int) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Isolation Forest ONNX → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export joblib → ONNX for IEP2 models.")
    parser.add_argument("--models-dir", default="iep2/models", help="Path to models directory")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    print(f"Models directory: {models_dir.resolve()}")

    label_map_path = models_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            raw = json.load(f)
        print(f"  Label map: { {k:v for k,v in raw.items() if k != '_decision_threshold'} }")
        if "_decision_threshold" in raw:
            print(f"  Decision threshold: {raw['_decision_threshold']:.4f}")

    errors = []

    # XGBoost
    xgb_joblib = models_dir / "xgboost_classifier.joblib"
    xgb_onnx   = models_dir / "xgboost_classifier.onnx"
    if xgb_joblib.exists():
        print(f"\nExporting XGBoost from {xgb_joblib}...")
        try:
            model = joblib.load(xgb_joblib)
            export_xgboost(model, xgb_onnx, n_features=_n_features(model))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(f"XGBoost: {e}")
    else:
        print(f"  SKIP: {xgb_joblib} not found")

    # Random Forest
    rf_joblib = models_dir / "rf_classifier.joblib"
    rf_onnx   = models_dir / "rf_classifier.onnx"
    if rf_joblib.exists():
        print(f"\nExporting Random Forest from {rf_joblib}...")
        try:
            model = joblib.load(rf_joblib)
            export_rf(model, rf_onnx, n_features=_n_features(model))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(f"RandomForest: {e}")
    else:
        print(f"  SKIP: {rf_joblib} not found")

    # Isolation Forest
    if_joblib = models_dir / "isolation_forest.joblib"
    if_onnx   = models_dir / "isolation_forest.onnx"
    if if_joblib.exists():
        print(f"\nExporting Isolation Forest from {if_joblib}...")
        try:
            model = joblib.load(if_joblib)
            export_isolation_forest(model, if_onnx, n_features=_n_features(model))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors.append(f"IsolationForest: {e}")
    else:
        print(f"  SKIP: {if_joblib} not found")

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All ONNX exports successful.")


if __name__ == "__main__":
    main()
