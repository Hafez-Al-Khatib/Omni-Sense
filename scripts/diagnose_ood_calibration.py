#!/usr/bin/env python3
"""Diagnose OOD calibration: compare joblib vs ONNX outputs on same data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthesized"

# Load models
print("Loading models...")
if_joblib = joblib.load(MODEL_DIR / "isolation_forest.joblib")
sess_if = ort.InferenceSession(
    str(MODEL_DIR / "isolation_forest.onnx"), providers=["CPUExecutionProvider"]
)

# Load training data (what the model was trained on)
print("Loading training data...")
for fname in ["eep_features.parquet", "embeddings.parquet"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        df = pd.read_parquet(fpath)
        print(f"  Loaded {fname}: {df.shape}")
        break
else:
    print("ERROR: No parquet found")
    sys.exit(1)

# Extract embedding columns
emb_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
X = df[emb_cols].values.astype(np.float32)
print(f"  Embeddings shape: {X.shape}")
print(f"  Labels: {df['label'].value_counts().to_dict()}")

# Score with joblib
print("\n--- Joblib Model Scores ---")
joblib_scores = if_joblib.decision_function(X)
joblib_preds = if_joblib.predict(X)
print(f"  Mean: {joblib_scores.mean():.4f}, Std: {joblib_scores.std():.4f}")
print(f"  Min: {joblib_scores.min():.4f}, Max: {joblib_scores.max():.4f}")
print(f"  % flagged anomalous (pred=-1): {100*np.mean(joblib_preds == -1):.1f}%")
print(f"  % below 0.37: {100*np.mean(joblib_scores < 0.37):.1f}%")
print(f"  % below 0.0: {100*np.mean(joblib_scores < 0.0):.1f}%")

# Score with ONNX
print("\n--- ONNX Model Scores ---")
input_name = sess_if.get_inputs()[0].name
onnx_result = sess_if.run(None, {input_name: X})
onnx_labels = onnx_result[0].flatten()
onnx_scores = onnx_result[1].flatten()
print(f"  Mean: {onnx_scores.mean():.4f}, Std: {onnx_scores.std():.4f}")
print(f"  Min: {onnx_scores.min():.4f}, Max: {onnx_scores.max():.4f}")
print(f"  % flagged anomalous (pred=-1): {100*np.mean(onnx_labels == -1):.1f}%")
print(f"  % below 0.37: {100*np.mean(onnx_scores < 0.37):.1f}%")
print(f"  % below 0.0: {100*np.mean(onnx_scores < 0.0):.1f}%")

# Compare
print("\n--- Comparison ---")
print(f"  Score correlation: {np.corrcoef(joblib_scores, onnx_scores)[0,1]:.4f}")
print(f"  Mean absolute diff: {np.abs(joblib_scores - onnx_scores).mean():.6f}")
print(f"  Max absolute diff: {np.abs(joblib_scores - onnx_scores).max():.6f}")

# Suggest proper threshold
print("\n--- Recommended Thresholds ---")
for percentile in [1, 5, 10]:
    thresh = np.percentile(joblib_scores, percentile)
    print(f"  {percentile}th percentile (joblib): {thresh:.4f}  -> flags {percentile}% as OOD")

# Check golden dataset
GOLDEN_CSV = Path(__file__).resolve().parent.parent / "data" / "golden" / "golden_dataset_v1.csv"
if GOLDEN_CSV.exists():
    print("\n--- Golden Dataset ---")
    gdf = pd.read_csv(GOLDEN_CSV)
    g_emb_cols = sorted([c for c in gdf.columns if c.startswith("embedding_")])
    if len(g_emb_cols) == len(emb_cols):
        X_g = gdf[g_emb_cols].values.astype(np.float32)
        g_scores = if_joblib.decision_function(X_g)
        print(f"  Golden mean: {g_scores.mean():.4f}, std: {g_scores.std():.4f}")
        print(f"  Golden % below 0.37: {100*np.mean(g_scores < 0.37):.1f}%")
        print(f"  Golden % below 0.0: {100*np.mean(g_scores < 0.0):.1f}%")
    else:
        print(f"  Golden has {len(g_emb_cols)} dims vs training {len(emb_cols)} dims — MISMATCH!")
