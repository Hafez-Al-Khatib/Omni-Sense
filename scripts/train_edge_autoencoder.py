"""
Train + quantise a tiny feature-space autoencoder for the ESP32-S3 OOD gate.
=============================================================================

Input parquet  : data/synthesized/eep_features_3200hz.parquet (39-d at 3200 Hz)
Output models  : iep2/models/edge_autoencoder.tflite        (int8 quantised)
                 iep2/models/edge_autoencoder.h5            (Keras float32)
                 iep2/models/edge_threshold.json            (calibrated p95)
                 iep2/models/edge_metrics.json              (eval evidence)

Architecture
------------
    Input(39) -> Dense(16, relu) -> Dense(8, relu)        # encoder (bottleneck)
              -> Dense(16, relu) -> Dense(39, linear)     # decoder

~1.5 K parameters; ~3-5 KB after int8 quantisation. Fits in ESP32-S3 SRAM
several times over with TFLite-Micro arena overhead included.

Training data
-------------
Trained ONLY on No_Leak / Normal_Operation recordings (the "normal"
acoustic distribution). At inference, leak audio produces high
reconstruction error because the model has never seen that signature.
This is the same OOD principle as the cloud autoencoder, just at the
feature-space level instead of the spectrogram level (so it fits on a
microcontroller).

Threshold calibration
---------------------
Reconstruction MSE is computed on a held-out validation slice of the
No_Leak corpus. Threshold = 95th percentile, matching the cloud
calibration convention (~5 % of normal samples flagged as OOD; chosen
to favour recall of true anomalies over silencing of borderline normals).

Usage
-----
    py -3.12 scripts/train_edge_autoencoder.py
    py -3.12 scripts/train_edge_autoencoder.py --epochs 200 --bottleneck 4

If tensorflow is not installed, the script writes a placeholder TFLite
model (~16 bytes) and a marker JSON so the ESP32-S3 firmware still
compiles. Install with:  pip install tensorflow==2.16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ      = Path(__file__).parent.parent
PARQUET   = PROJ / "data/synthesized/eep_features_3200hz.parquet"
OUT_DIR   = PROJ / "iep2/models"

NO_LEAK_LABELS = {"No_Leak", "Normal_Operation"}


def load_normal_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_normal, X_anomaly):
      X_normal  : (n, 39) float32 — recordings labelled No_Leak / Normal_Operation
      X_anomaly : (n, 39) float32 — held-out leak recordings for AUC sanity check
    """
    df = pd.read_parquet(path)
    emb_cols = sorted(
        (c for c in df.columns if c.startswith("embedding_")),
        key=lambda x: int(x.split("_")[1]),
    )
    df["is_normal"] = df["label"].isin(NO_LEAK_LABELS)
    # Recording-level mean-pool to match how cloud features are aggregated
    agg = df.groupby("source_wav", as_index=False).agg(
        {**{c: "mean" for c in emb_cols}, "is_normal": "min"}
    )
    X = agg[emb_cols].values.astype(np.float32)
    is_normal = agg["is_normal"].values.astype(bool)
    return X[is_normal], X[~is_normal]


def write_placeholder(reason: str) -> None:
    """Write a tiny stub model + threshold so firmware compiles."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = bytes(16)   # zero bytes — TFLite-Micro will reject; firmware checks
    (OUT_DIR / "edge_autoencoder.tflite").write_bytes(placeholder)
    (OUT_DIR / "edge_threshold.json").write_text(json.dumps({
        "threshold": 0.5,
        "placeholder": True,
        "reason": reason,
    }, indent=2))
    print(f"[stub] wrote placeholder model + threshold ({reason})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=120)
    p.add_argument("--bottleneck", type=int,   default=8)
    p.add_argument("--hidden",     type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val-frac",   type=float, default=0.20)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} not found. Run extract_eep_features.py first.")
        sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError as exc:
        write_placeholder(f"tensorflow not installed ({exc})")
        sys.exit(0)   # not a hard failure — firmware can still compile

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    X_normal, X_anom = load_normal_features(PARQUET)
    print(f"Normal recordings:   {len(X_normal)}")
    print(f"Anomaly recordings:  {len(X_anom)}")
    if len(X_normal) < 20:
        write_placeholder(f"too few normal recordings ({len(X_normal)})")
        sys.exit(0)

    # Per-feature standardisation. Saved alongside the model so the
    # device can apply the same affine transform before inference.
    mean = X_normal.mean(axis=0)
    std  = X_normal.std(axis=0)  + 1e-9
    Xn   = (X_normal - mean) / std
    Xa   = (X_anom   - mean) / std

    # Train/val split (recording-level; we already aggregated above)
    n_val = max(1, int(len(Xn) * args.val_frac))
    idx   = np.random.permutation(len(Xn))
    Xv    = Xn[idx[:n_val]]
    Xt    = Xn[idx[n_val:]]
    print(f"Train / val:         {len(Xt)} / {len(Xv)}")

    # ── Model ──────────────────────────────────────────────────────────
    inp = tf.keras.layers.Input(shape=(39,))
    x   = tf.keras.layers.Dense(args.hidden,     activation="relu")(inp)
    x   = tf.keras.layers.Dense(args.bottleneck, activation="relu", name="bottleneck")(x)
    x   = tf.keras.layers.Dense(args.hidden,     activation="relu")(x)
    out = tf.keras.layers.Dense(39, activation="linear")(x)
    model = tf.keras.Model(inp, out, name="omni_edge_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse")
    model.summary(print_fn=print)

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                          restore_best_weights=True)
    model.fit(Xt, Xt, validation_data=(Xv, Xv),
              epochs=args.epochs, batch_size=32, verbose=2, callbacks=[es])

    # ── Threshold calibration on held-out normal val ───────────────────
    val_recon = model.predict(Xv, verbose=0)
    val_mse   = np.mean((Xv - val_recon) ** 2, axis=1)
    threshold = float(np.percentile(val_mse, args.percentile))

    # Anomaly sanity: AUC on (normal val) vs (all anomaly)
    if len(Xa) > 0:
        anom_recon = model.predict(Xa, verbose=0)
        anom_mse   = np.mean((Xa - anom_recon) ** 2, axis=1)
        from sklearn.metrics import roc_auc_score
        scores = np.concatenate([val_mse, anom_mse])
        labels = np.concatenate([np.zeros_like(val_mse), np.ones_like(anom_mse)])
        anom_auc = float(roc_auc_score(labels, scores))
    else:
        anom_mse, anom_auc = np.array([]), float("nan")

    # ── Save Keras float ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / "edge_autoencoder.h5")

    # ── TFLite int8 quantisation ───────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Representative dataset: normal training samples
    rep = Xt[:200].astype(np.float32)
    def rep_gen():
        for i in range(len(rep)):
            yield [rep[i:i+1]]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    (OUT_DIR / "edge_autoencoder.tflite").write_bytes(tflite_bytes)

    # ── Save threshold + per-feature stats (the device needs both) ─────
    (OUT_DIR / "edge_threshold.json").write_text(json.dumps({
        "threshold":  threshold,
        "percentile": args.percentile,
        "n_normal_val": int(len(Xv)),
        "feature_mean": mean.tolist(),
        "feature_std":  std.tolist(),
        "placeholder": False,
    }, indent=2))

    metrics = {
        "n_normal_train":     int(len(Xt)),
        "n_normal_val":       int(len(Xv)),
        "n_anomaly":          int(len(Xa)),
        "val_mse_mean":       float(val_mse.mean()),
        "val_mse_p95":        threshold,
        "anomaly_mse_mean":   float(anom_mse.mean()) if len(anom_mse) else None,
        "anomaly_vs_normal_auc": anom_auc,
        "tflite_bytes":       len(tflite_bytes),
        "params":             int(model.count_params()),
        "bottleneck":         args.bottleneck,
        "hidden":             args.hidden,
    }
    (OUT_DIR / "edge_metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    print("=" * 72)
    print(f"  TFLite size      : {len(tflite_bytes)} bytes")
    print(f"  Params           : {model.count_params()}")
    print(f"  Val MSE p95      : {threshold:.5f}  ({args.percentile}th pct on held-out normal)")
    print(f"  Anomaly vs normal AUC: {anom_auc:.4f}")
    print(f"  Output dir       : {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
