"""
Omni EEP Head Training Script
================================
Trains XGBoost + Random Forest classifiers on the 208-d DSP physics features
extracted by ``omni/eep/features.py`` (pure numpy, no YAMNet dependency).
Exports ONNX models to ``omni/models/`` for runtime inference by the EEP
orchestrator (via ONNX Runtime — no xgboost/sklearn needed at inference).

Usage
-----
    python scripts/train_omni_heads.py \\
        --clips-dir  data/synthesized \\
        --output-dir omni/models \\
        [--binary] [--n-estimators 200] [--seed 42]

Output
------
    omni/models/xgb_head.onnx          — XGBoost binary classifier
    omni/models/rf_head.onnx           — Random Forest binary classifier
    omni/models/omni_label_map.json    — {0: "Leak", 1: "No_Leak"}
    omni/models/omni_threshold.json    — {"xgb": 0.62, "rf": 0.55, "fused": 0.60}

Design notes
------------
BINARY vs MULTI-CLASS
  Pass ``--binary`` (recommended) to collapse all leak types into one class.
  Multi-class is supported but IEP2's fusion assumes binary p(leak) outputs.

RECORDING-LEVEL CV SPLIT
  We split by ``source_wav`` to prevent clips from the same recording ending
  up in both train and val (see train_cnn.py for the full rationale).

DSP FEATURES (39-d + 2 metadata = 41-d input)
  Features extracted with omni/eep/features.py — no scipy/librosa dependency.
  This makes the omni platform self-contained and suitable for edge deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("train_omni_heads")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from omni.eep.features import extract_features_with_meta

LEAK_LABELS   = {"Circumferential_Crack", "Gasket_Leak", "Longitudinal_Crack", "Orifice_Leak"}
NORMAL_LABELS = {"No_Leak", "Normal_Operation"}

PIPE_MATERIAL_MAP = {"PVC": 0.0, "Steel": 1.0, "Cast_Iron": 2.0}


# ─── Feature extraction over a clips directory ────────────────────────────────

def extract_dataset(
    clips_dir: Path,
    binary: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load all WAV clips from clips_dir and extract DSP features.

    Returns
    -------
    X             : (n, 41) float32 — DSP features + metadata
    y             : (n,) int        — class indices
    source_wavs   : (n,) str        — for recording-level CV split
    class_names   : list[str]
    """
    import pandas as pd

    meta_path = clips_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {clips_dir}")

    df = pd.read_csv(meta_path)
    log.info("Total clips in metadata: %d", len(df))
    log.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    X_list      = []
    y_list      = []
    source_list = []
    skipped     = 0

    for _, row in df.iterrows():
        wav_path = clips_dir / row["filename"]
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            import soundfile as sf
            pcm, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            log.debug("Skip %s: %s", row["filename"], exc)
            skipped += 1
            continue

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        # Pad/trim to 5 seconds
        target = sr * 5
        pcm = np.pad(pcm, (0, target - len(pcm))) if len(pcm) < target else pcm[:target]

        try:
            feat = extract_features_with_meta(
                pcm,
                sr=sr,
                pipe_material=str(row.get("pipe_material", "PVC")),
                pressure_bar=float(row.get("pressure_bar", 3.0)),
            )
        except Exception as exc:
            log.debug("Feature error %s: %s", row["filename"], exc)
            skipped += 1
            continue

        if not np.all(np.isfinite(feat)):
            skipped += 1
            continue

        label = str(row["label"])
        if binary:
            if label in LEAK_LABELS:
                y_list.append(0)   # Leak
            else:
                y_list.append(1)   # No_Leak
        else:
            y_list.append(label)

        X_list.append(feat)
        source_list.append(str(row.get("source_wav", row["filename"])))

    if skipped:
        log.warning("Skipped %d clips (missing/corrupt)", skipped)

    if len(X_list) == 0:
        raise RuntimeError("No valid clips found!")

    X = np.stack(X_list).astype(np.float32)

    if binary:
        y = np.array(y_list, dtype=np.int32)
        class_names = ["Leak", "No_Leak"]
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y  = le.fit_transform(y_list).astype(np.int32)
        class_names = list(le.classes_)

    source_wavs = np.array(source_list)

    log.info("Extracted: X=%s  classes=%s", X.shape, class_names)
    for i, name in enumerate(class_names):
        log.info("  class %d (%s): %d samples", i, name, (y == i).sum())

    return X, y, source_wavs, class_names


# ─── Recording-level train/val split ─────────────────────────────────────────

def recording_split(
    source_wavs: np.ndarray,
    val_frac:    float = 0.15,
    seed:        int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split by unique source recording so no augmented clip from the same
    recording appears in both train and val.

    Returns (train_idx, val_idx).
    """
    rng = np.random.default_rng(seed)
    unique_sources = np.unique(source_wavs)
    rng.shuffle(unique_sources)
    n_val = max(3, int(len(unique_sources) * val_frac))
    val_sources = set(unique_sources[:n_val])

    train_idx = np.where([s not in val_sources for s in source_wavs])[0]
    val_idx   = np.where([s in val_sources     for s in source_wavs])[0]
    return train_idx, val_idx


# ─── Training ─────────────────────────────────────────────────────────────────

def train_and_export(
    X: np.ndarray,
    y: np.ndarray,
    source_wavs: np.ndarray,
    class_names: list[str],
    output_dir:  Path,
    n_estimators: int = 200,
    seed:         int = 42,
) -> dict:
    """Train XGB + RF, evaluate on val, export ONNX. Returns metrics dict."""
    import xgboost as xgb

    # Export helpers
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score, roc_auc_score

    train_idx, val_idx = recording_split(source_wavs, seed=seed)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    log.info("Train: %d  Val: %d", len(X_tr), len(X_val))

    n_classes = len(class_names)
    binary = (n_classes == 2)

    # ── XGBoost ──────────────────────────────────────────────────────────
    scale_pos = (y_tr == 0).sum() / ((y_tr == 1).sum() + 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos if binary else 1,
        eval_metric="logloss" if binary else "mlogloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Random Forest ─────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        class_weight="balanced",
        oob_score=True,
        random_state=seed,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)

    # ── Evaluate ──────────────────────────────────────────────────────────
    xgb_proba = xgb_model.predict_proba(X_val)
    rf_proba  = rf_model.predict_proba(X_val)
    fused     = 0.6 * xgb_proba + 0.4 * rf_proba

    xgb_pred  = np.argmax(xgb_proba, axis=1)
    rf_pred   = np.argmax(rf_proba,  axis=1)
    fus_pred  = np.argmax(fused, axis=1)

    xgb_f1  = f1_score(y_val, xgb_pred,  average="binary" if binary else "weighted")
    rf_f1   = f1_score(y_val, rf_pred,   average="binary" if binary else "weighted")
    fus_f1  = f1_score(y_val, fus_pred,  average="binary" if binary else "weighted")

    metrics = {"xgb_f1": xgb_f1, "rf_f1": rf_f1, "fused_f1": fus_f1}

    if binary:
        xgb_auc = roc_auc_score(y_val, xgb_proba[:, 0])
        rf_auc  = roc_auc_score(y_val, rf_proba[:, 0])
        metrics.update({"xgb_auc": xgb_auc, "rf_auc": rf_auc})
        # Operating threshold: maximise F1 on val
        for name, proba in [("xgb", xgb_proba[:, 0]), ("rf", rf_proba[:, 0])]:
            thresholds = np.linspace(0.1, 0.9, 81)
            best_thr, best_f1 = 0.5, 0.0
            for thr in thresholds:
                pred = (proba >= thr).astype(int)
                f = f1_score(y_val, pred, pos_label=0, zero_division=0)
                if f > best_f1:
                    best_f1, best_thr = f, thr
            metrics[f"{name}_threshold"] = best_thr

    log.info("Validation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    if binary and (y_val == 0).sum() > 0:
        log.info("Classification report (XGB):\n%s",
                 classification_report(y_val, xgb_pred, target_names=class_names))
    log.info("RF OOB score: %.4f", rf_model.oob_score_)
    metrics["rf_oob"] = rf_model.oob_score_

    # ── ONNX export ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = X.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    # XGBoost → ONNX via skl2onnx
    try:
        xgb_onnx = convert_sklearn(xgb_model, initial_types=initial_type,
                                   target_opset=13)
        xgb_path = output_dir / "xgb_head.onnx"
        with open(xgb_path, "wb") as f:
            f.write(xgb_onnx.SerializeToString())
        log.info("XGB exported → %s", xgb_path)
    except Exception as exc:
        log.error("XGB ONNX export failed: %s", exc)
        # Fallback: try XGBoost's native ONNX export
        try:
            booster = xgb_model.get_booster()
            booster.save_model(str(output_dir / "xgb_head.json"))
            log.info("XGB saved as JSON (ONNX export failed); convert manually.")
        except Exception:
            pass

    # RF → ONNX
    try:
        rf_onnx = convert_sklearn(rf_model, initial_types=initial_type,
                                  target_opset=13)
        rf_path = output_dir / "rf_head.onnx"
        with open(rf_path, "wb") as f:
            f.write(rf_onnx.SerializeToString())
        log.info("RF exported → %s", rf_path)
    except Exception as exc:
        log.error("RF ONNX export failed: %s", exc)

    # Label map + thresholds
    label_map = {str(i): name for i, name in enumerate(class_names)}
    (output_dir / "omni_label_map.json").write_text(json.dumps(label_map, indent=2))

    thresholds = {
        "xgb":   metrics.get("xgb_threshold", 0.60),
        "rf":    metrics.get("rf_threshold",  0.55),
        "fused": 0.60,
    }
    (output_dir / "omni_threshold.json").write_text(json.dumps(thresholds, indent=2))

    # Summary
    print("\n" + "═" * 72)
    print("  OMNI EEP HEAD TRAINING COMPLETE")
    print("═" * 72)
    print(f"  Classes     : {class_names}")
    print(f"  Train / Val : {len(X_tr)} / {len(X_val)}")
    print(f"  XGB F1      : {xgb_f1:.4f}  AUC: {metrics.get('xgb_auc', 0):.4f}")
    print(f"  RF  F1      : {rf_f1:.4f}   OOB: {rf_model.oob_score_:.4f}")
    print(f"  Fused F1    : {fus_f1:.4f}")
    print(f"  Output dir  : {output_dir}")
    print("═" * 72)

    return metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train XGB + RF EEP heads on DSP features, export ONNX"
    )
    ap.add_argument("--clips-dir",    default="data/synthesized")
    ap.add_argument("--output-dir",   default="omni/models")
    ap.add_argument("--binary",       action="store_true", default=True,
                    help="Collapse all leak types into a single 'Leak' class")
    ap.add_argument("--no-binary",    dest="binary", action="store_false")
    ap.add_argument("--n-estimators", type=int,   default=200)
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    X, y, source_wavs, class_names = extract_dataset(
        Path(args.clips_dir), binary=args.binary
    )
    train_and_export(
        X, y, source_wavs, class_names,
        output_dir=Path(args.output_dir),
        n_estimators=args.n_estimators,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
