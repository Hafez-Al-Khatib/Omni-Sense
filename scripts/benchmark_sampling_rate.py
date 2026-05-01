#!/usr/bin/env python3
"""Benchmark IEP2 inference at 16 kHz vs 3.2 kHz resampled.

Models were trained on 16 kHz features. This script quantifies the performance
drop when the same audio is downsampled to 3.2 kHz and FFT-resampled back
to 16 kHz (matching the honest hardware pipeline).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omni" / "eep"))

try:
    from features import extract_features_with_meta
except Exception as exc:
    print(f"ERROR: Cannot import feature extractor: {exc}")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed")
    sys.exit(1)

try:
    from scipy import signal
except ImportError:
    print("ERROR: scipy not installed")
    sys.exit(1)

# Load models
MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
sess_if = ort.InferenceSession(
    str(MODEL_DIR / "isolation_forest.onnx"), providers=["CPUExecutionProvider"]
)
sess_xgb = ort.InferenceSession(
    str(MODEL_DIR / "xgboost_classifier.onnx"), providers=["CPUExecutionProvider"]
)

# Pick representative files: 3 leak, 3 no-leak
AUDIO_DIR = Path(__file__).resolve().parent.parent / "Processed_audio_16k"
files = sorted(AUDIO_DIR.glob("*.wav"))
# Select a balanced subset
leak_files = [f for f in files if "Leak" in f.name or "leak" in f.name.lower()][:3]
noleak_files = [f for f in files if f not in leak_files][:3]
test_files = leak_files + noleak_files

print(f"Benchmarking {len(test_files)} files...\n")
print(
    f"{'File':<55} {'True':<10} {'16k_pred':<10} {'16k_conf':<10} "
    f"{'3k2_pred':<10} {'3k2_conf':<10} {'OOD_16k':<10} {'OOD_3k2':<10}"
)
print("-" * 135)

results = []

for f in test_files:
    pcm, sr = librosa.load(str(f), sr=16000, mono=True)

    # True label from filename heuristic
    true_label = "Leak" if "Leak" in f.name or "leak" in f.name.lower() else "No_Leak"

    # 16 kHz path
    feat_16k = extract_features_with_meta(
        pcm, sr=16000, pipe_material="PVC", pressure_bar=3.0
    ).reshape(1, -1)

    # 3.2 kHz path: downsample to 3200, then FFT-resample back to 16000
    pcm_3k2 = signal.resample(pcm, int(len(pcm) * 3200 / 16000))
    pcm_3k2_back = signal.resample(pcm_3k2, len(pcm))
    feat_3k2 = extract_features_with_meta(
        pcm_3k2_back, sr=16000, pipe_material="PVC", pressure_bar=3.0
    ).reshape(1, -1)

    # OOD (Isolation Forest) — uses first 39 features
    if_16k = sess_if.run(None, {"float_input": feat_16k[:, :39].astype(np.float32)})[0]
    if_3k2 = sess_if.run(None, {"float_input": feat_3k2[:, :39].astype(np.float32)})[0]

    # Classification
    xgb_16k = sess_xgb.run(None, {"float_input": feat_16k.astype(np.float32)})[0]
    xgb_3k2 = sess_xgb.run(None, {"float_input": feat_3k2.astype(np.float32)})[0]

    pred_16k = "Leak" if xgb_16k[0] > 0.5 else "No_Leak"
    pred_3k2 = "Leak" if xgb_3k2[0] > 0.5 else "No_Leak"

    ood_16k = "REJECT" if if_16k[0] < 0 else "PASS"
    ood_3k2 = "REJECT" if if_3k2[0] < 0 else "PASS"

    print(
        f"{f.name:<55} {true_label:<10} {pred_16k:<10} {xgb_16k[0]:.3f}      "
        f"{pred_3k2:<10} {xgb_3k2[0]:.3f}      {ood_16k:<10} {ood_3k2:<10}"
    )

    results.append(
        {
            "file": f.name,
            "true": true_label,
            "pred_16k": pred_16k,
            "conf_16k": float(xgb_16k[0]),
            "pred_3k2": pred_3k2,
            "conf_3k2": float(xgb_3k2[0]),
            "ood_16k": ood_16k,
            "ood_3k2": ood_3k2,
        }
    )

# Summary stats
acc_16k = sum(1 for r in results if r["pred_16k"] == r["true"]) / len(results)
acc_3k2 = sum(1 for r in results if r["pred_3k2"] == r["true"]) / len(results)
ood_reject_16k = sum(1 for r in results if r["ood_16k"] == "REJECT") / len(results)
ood_reject_3k2 = sum(1 for r in results if r["ood_3k2"] == "REJECT") / len(results)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Accuracy @ 16 kHz (native):  {acc_16k:.1%} ({sum(1 for r in results if r['pred_16k']==r['true'])}/{len(results)})")
print(f"Accuracy @ 3.2k->16k resample: {acc_3k2:.1%} ({sum(1 for r in results if r['pred_3k2']==r['true'])}/{len(results)})")
print(f"OOD rejection rate @ 16 kHz:  {ood_reject_16k:.1%}")
print(f"OOD rejection rate @ 3.2k:    {ood_reject_3k2:.1%}")

if ood_reject_3k2 > 0:
    print("\nNOTE: 3.2 kHz resampled data triggers OOD rejections because the")
    print("feature distribution shifts (spectral centroid/bandwidth drop).")
    print("This is SAFELY CORRECT behavior — the system says 'I don't know'")
    print("rather than guessing on out-of-distribution data.")
