"""
Deployment-rate parity evaluation
==================================
Compares the archived 16 kHz baseline models against the freshly-trained
3200 Hz models, **both evaluated on 3200 Hz features** — i.e. the audio
distribution the ESP32-S3 + ADXL345 actually produces in the field.

This isolates the deployment-rate parity problem cleanly:

    A. 16 kHz models on 3200 Hz features  →  what production does today
    B.  3200 Hz models on 3200 Hz features →  what production should do

The delta (A → B) is the entire justification for the retraining work
documented in coordination/decisions.md (2026-05-02 entry).

Usage
-----
    py -3.12 scripts/eval_deployment_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))

PARQUET     = PROJ / "data/synthesized/eep_features_3200hz.parquet"
MODELS_NEW  = PROJ / "iep2/models"
MODELS_OLD  = PROJ / "iep2/models/_archive_16khz"
THRESHOLD   = 0.952       # class-prior decision threshold (matches label_map.json)

MATERIAL_MAP = {"PVC": 0, "Steel": 1, "Cast_Iron": 2}

# Binary collapse — must match the convention used by train_models.py.
# Both raw labels "No_Leak" (LeakDB negatives) and "Normal_Operation"
# (MIMII pump hard-negatives) collapse to the negative class.
NO_LEAK_LABELS = {"No_Leak", "Normal_Operation"}


def load_data() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    df = pd.read_parquet(PARQUET)
    emb_cols = sorted(
        (c for c in df.columns if c.startswith("embedding_")),
        key=lambda x: int(x.split("_")[1]),
    )
    # Collapse to binary BEFORE aggregating so groupby cannot pick a
    # "first" raw label and silently invert the class.
    df["material_code"] = df["pipe_material"].map(MATERIAL_MAP).fillna(0)
    df["is_leak"] = (~df["label"].isin(NO_LEAK_LABELS)).astype(int)

    agg = df.groupby("source_wav", as_index=False).agg(
        {**{c: "mean" for c in emb_cols},
         "material_code": "first", "pressure_bar": "mean",
         "is_leak": "max", "label": "first"}
    )
    X = np.hstack([
        agg[emb_cols].values.astype(np.float32),
        agg[["material_code", "pressure_bar"]].values.astype(np.float32),
    ])
    y = agg["is_leak"].values.astype(int)   # Leak=1, No_Leak=0
    return X, y, agg["label"]


def evaluate_pair(name: str, models_dir: Path, X: np.ndarray, y: np.ndarray) -> dict:
    xgb_path = models_dir / "xgboost_classifier.joblib"
    rf_path  = models_dir / "rf_classifier.joblib"
    if not xgb_path.exists() or not rf_path.exists():
        print(f"[{name}] SKIP — missing models in {models_dir}")
        return {}

    xgb = joblib.load(xgb_path)
    rf  = joblib.load(rf_path)

    # Defensive: training/eval feature-count parity
    if xgb.n_features_in_ != X.shape[1]:
        print(f"[{name}] FATAL — model expects {xgb.n_features_in_} features, "
              f"got {X.shape[1]}")
        return {"f1": float("nan"), "roc_auc": float("nan"),
                "feature_dim_mismatch": True}

    # The label_map convention: class index 0 = Leak (positive class for AUC)
    # train_models.py uses ensemble = 0.6*XGB + 0.4*RF
    p_xgb_leak = xgb.predict_proba(X)[:, 0]
    p_rf_leak  = rf.predict_proba(X)[:, 0]
    p_leak     = 0.6 * p_xgb_leak + 0.4 * p_rf_leak

    # Production decision rule: predict No_Leak iff P(No_Leak) >= threshold
    y_pred = (p_leak > (1.0 - THRESHOLD)).astype(int)   # Leak=1 if p_leak high

    metrics = {
        "name":       name,
        "n_samples":  int(len(y)),
        "n_leak":     int(y.sum()),
        "n_no_leak":  int((y == 0).sum()),
        "accuracy":   float(accuracy_score(y, y_pred)),
        "precision":  float(precision_score(y, y_pred, zero_division=0)),
        "recall":     float(recall_score(y, y_pred, zero_division=0)),
        "f1":         float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y, p_leak)),
    }
    return metrics


def main() -> None:
    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found. Run extract_eep_features.py first.")
        sys.exit(1)

    X, y, _ = load_data()
    print(f"Eval set: {len(y)} recordings ({int(y.sum())} Leak / {int((y==0).sum())} No_Leak)")
    print(f"Feature shape: {X.shape}\n")

    new = evaluate_pair("3200 Hz (current)",  MODELS_NEW, X, y)
    old = evaluate_pair("16 kHz (archived)", MODELS_OLD, X, y)

    print("=" * 78)
    print(f"  {'Metric':<14} {'16 kHz models':>18} {'3200 Hz models':>18} {'Delta':>10}")
    print("-" * 78)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        a = old.get(k, float("nan"))
        b = new.get(k, float("nan"))
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<14} {a:>18.4f} {b:>18.4f}  {sign}{delta:>8.4f}")
    print("=" * 78)

    if old.get("feature_dim_mismatch"):
        print("\nNote: 16 kHz models cannot be evaluated on 3200 Hz features "
              "because they were trained at a different feature dimension.")

    # Persist for the Tradeoffs doc / MODEL_REPORT.md
    import json
    out_path = PROJ / "iep2/models/parity_eval.json"
    out_path.write_text(json.dumps({"new_3200hz": new, "old_16khz": old}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
