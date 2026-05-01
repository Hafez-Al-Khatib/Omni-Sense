#!/usr/bin/env python3
"""Fix Isolation Forest OOD calibration: recompute offset_ from training data.

The saved model has offset_=-0.5114, but training data score_samples range
from -0.665 to -0.589. This causes 100% of samples to be flagged anomalous.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from sklearn.ensemble import IsolationForest

MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthesized"

def main() -> int:
    print("Loading Isolation Forest (joblib)...")
    if_model: IsolationForest = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    print(f"  Old offset_: {if_model.offset_:.6f}")
    print(f"  contamination: {if_model.contamination}")

    print("\nLoading training data (eep_features.parquet)...")
    df = pd.read_parquet(DATA_DIR / "eep_features.parquet")
    emb_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
    X = df[emb_cols].values.astype(np.float32)
    print(f"  Shape: {X.shape}")

    # Recompute offset_ from training data
    print("\nRecomputing offset_...")
    scores = if_model.score_samples(X)
    pct = if_model.contamination * 100
    new_offset = np.percentile(scores, pct)
    print(f"  score_samples range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  {pct}th percentile: {new_offset:.6f}")

    # Apply new offset
    if_model.offset_ = float(new_offset)
    print(f"  New offset_: {if_model.offset_:.6f}")

    # Verify
    decision = if_model.decision_function(X)
    preds = if_model.predict(X)
    n_anom = np.sum(preds == -1)
    print(f"\nVerification on training data:")
    print(f"  decision_function range: [{decision.min():.4f}, {decision.max():.4f}]")
    print(f"  Predicted anomalies: {n_anom} / {len(X)} ({100*n_anom/len(X):.1f}%)")
    print(f"  Expected: ~{if_model.contamination*100:.1f}%")

    # Save corrected joblib
    joblib_path = MODEL_DIR / "isolation_forest.joblib"
    print(f"\nSaving corrected joblib -> {joblib_path}")
    joblib.dump(if_model, joblib_path)

    # Export corrected ONNX
    print("Exporting corrected ONNX...")
    n_features = X.shape[1]
    if_onnx_path = MODEL_DIR / "isolation_forest.onnx"
    try:
        # Use sklearn-onnx converter (same method as train_models.py)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(if_model, initial_types=initial_type, target_opset={'ai.onnx.ml': 3})
        with open(if_onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  Saved ONNX -> {if_onnx_path}")
    except Exception as exc:
        print(f"  ONNX export failed: {exc}")
        print("  The joblib model is fixed; ONNX export must be done manually or in Docker.")
        return 1

    # Verify ONNX matches joblib
    print("\nVerifying ONNX matches joblib...")
    sess = ort.InferenceSession(str(if_onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_result = sess.run(None, {input_name: X[:100]})
    onnx_labels = onnx_result[0].flatten()
    onnx_scores = onnx_result[1].flatten()
    joblib_labels = if_model.predict(X[:100])
    joblib_scores = if_model.decision_function(X[:100])

    labels_match = np.array_equal(onnx_labels, joblib_labels)
    score_corr = np.corrcoef(onnx_scores, joblib_scores)[0, 1]
    print(f"  Labels match: {labels_match}")
    print(f"  Score correlation: {score_corr:.6f}")

    # Recommend new threshold for config.py
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Set OOD_IF_THRESHOLD in omni/common/config.py to:")
    print(f"  OOD_IF_THRESHOLD = {new_offset:.4f}")
    print(f"\nThis will flag ~{if_model.contamination*100:.0f}% of training data as OOD,")
    print("which matches the contamination parameter used during training.")
    print("\nFor a more conservative threshold (fewer false rejections), use:")
    print(f"  OOD_IF_THRESHOLD = {np.percentile(scores, 1):.4f}  (flags ~1%)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
