#!/usr/bin/env python3
"""Deep diagnostic: inspect Isolation Forest internals and data distribution."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthesized"

# Load model
if_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
print(f"Model type: {type(if_model)}")
print(f"Sklearn version (from model): {getattr(if_model, '_sklearn_version', 'unknown')}")
print(f"n_estimators: {if_model.n_estimators}")
print(f"contamination: {if_model.contamination}")
print(f"max_samples: {if_model.max_samples}")
print(f"max_features: {if_model.max_features}")
print(f"offset_: {if_model.offset_}")
print(f"n_features_in_: {if_model.n_features_in_}")

# Load data
df = pd.read_parquet(DATA_DIR / "eep_features.parquet")
emb_cols = sorted([c for c in df.columns if c.startswith("embedding_")])
X = df[emb_cols].values.astype(np.float32)

print(f"\nData shape: {X.shape}")
print(f"Data mean: {X.mean():.4f}, std: {X.std():.4f}")
print(f"Data min: {X.min():.4f}, max: {X.max():.4f}")
print(f"Any NaN: {np.isnan(X).any()}")
print(f"Any Inf: {np.isinf(X).any()}")

# Check score_samples vs decision_function
print("\n--- sklearn internals ---")
scores_samples = if_model.score_samples(X)
decision = if_model.decision_function(X)
preds = if_model.predict(X)

print(f"score_samples mean: {scores_samples.mean():.4f}, std: {scores_samples.std():.4f}")
print(f"score_samples min: {scores_samples.min():.4f}, max: {scores_samples.max():.4f}")
print(f"decision_function mean: {decision.mean():.4f}, std: {decision.std():.4f}")
print(f"decision_function min: {decision.min():.4f}, max: {decision.max():.4f}")
print(f"% predicted -1: {100*np.mean(preds == -1):.1f}%")

# The relationship: decision_function(x) = score_samples(x) - offset_
print(f"\nOffset check: {(scores_samples - if_model.offset_ - decision).max():.10f}")

# Check if offset makes sense
print(f"\nscore_samples 5th percentile: {np.percentile(scores_samples, 5):.4f}")
print(f"offset_: {if_model.offset_:.4f}")
print(f"Expected: offset should be ~5th percentile of score_samples for contamination=0.05")

# What if we manually compute the threshold?
manual_thresh = np.percentile(scores_samples, if_model.contamination * 100)
print(f"Manual {if_model.contamination*100}th percentile threshold: {manual_thresh:.4f}")
print(f"% below manual thresh: {100*np.mean(scores_samples < manual_thresh):.1f}%")

# Check by label
print("\n--- Scores by label ---")
labels = df["label"].values
for lbl in sorted(set(labels)):
    mask = labels == lbl
    s = decision[mask]
    print(f"  {lbl:<20} n={mask.sum():4d}  mean={s.mean():.4f}  std={s.std():.4f}  %anom={100*np.mean(preds[mask]==-1):.1f}%")
