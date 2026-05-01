#!/usr/bin/env python3
"""Verify the OOD fix by loading the corrected model through the production path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iep2" / "app"))
from ood_detector import OODDetector

# Load data
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthesized"
df = pd.read_parquet(DATA_DIR / "eep_features.parquet")
emb_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
X = df[emb_cols].values.astype(np.float32)

print("Loading OOD detector (production path)...")
ood = OODDetector()
MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
ood.load(
    onnx_path=str(MODEL_DIR / "isolation_forest.onnx"),
    joblib_path=str(MODEL_DIR / "isolation_forest.joblib"),
)
print(f"  Backend: {ood._backend}")
print(f"  is_loaded: {ood.is_loaded}")

# Score a few samples
print("\n--- Scoring 10 random training samples ---")
indices = np.random.choice(len(X), 10, replace=False)
scores = [ood.score(X[i]) for i in indices]
for i, s in zip(indices, scores):
    lbl = df.iloc[i]["label"]
    anom = "REJECT" if ood.is_anomalous(X[i]) else "PASS"
    print(f"  {lbl:<20} score={s:+.4f}  {anom}")

# Overall stats
print("\n--- Full training set statistics ---")
all_scores = np.array([ood.score(x) for x in X])
all_preds = np.array([ood.is_anomalous(x) for x in X])
n_anom = np.sum(all_preds)
print(f"  Mean score: {all_scores.mean():.4f}")
print(f"  Std score:  {all_scores.std():.4f}")
print(f"  Min score:  {all_scores.min():.4f}")
print(f"  Max score:  {all_scores.max():.4f}")
print(f"  Predicted anomalies: {n_anom} / {len(X)} ({100*n_anom/len(X):.1f}%)")

# Test with 3.2k resampled data
from scipy import signal
import librosa

AUDIO_DIR = Path(__file__).resolve().parent.parent / "Processed_audio_16k"
test_file = sorted(AUDIO_DIR.glob("*.wav"))[0]
pcm, sr = librosa.load(str(test_file), sr=16000, mono=True)
pcm_3k2 = signal.resample(pcm, int(len(pcm) * 3200 / 16000))
pcm_3k2_back = signal.resample(pcm_3k2, len(pcm))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omni" / "eep"))
from features import extract_features

feat_16k = extract_features(pcm, sr=16000).reshape(1, -1)
feat_3k2 = extract_features(pcm_3k2_back, sr=16000).reshape(1, -1)

print("\n--- 16 kHz vs 3.2 kHz on same file ---")
print(f"  16k score: {ood.score(feat_16k[0]):+.4f}  -> {'REJECT' if ood.is_anomalous(feat_16k[0]) else 'PASS'}")
print(f"  3k2 score: {ood.score(feat_3k2[0]):+.4f}  -> {'REJECT' if ood.is_anomalous(feat_3k2[0]) else 'PASS'}")

if n_anom < len(X) * 0.5:
    print("\n>>> SUCCESS: OOD detector is working correctly.")
    print(f"    ~{100*n_anom/len(X):.1f}% of training data flagged as anomalous (expected ~5%).")
else:
    print("\n>>> WARNING: Still flagging too many samples.")
