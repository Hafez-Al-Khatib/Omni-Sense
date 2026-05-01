#!/usr/bin/env python3
"""Final benchmark: 16 kHz vs 3.2 kHz using the corrected joblib OOD model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iep2" / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omni" / "eep"))

from features import extract_features_with_meta
import librosa
from scipy import signal
import joblib
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parent.parent / "iep2" / "models"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "Processed_audio_16k"

# Load models
with open(MODEL_DIR / "label_map.json") as f:
    label_meta = json.load(f)
THRESHOLD = label_meta.get("_decision_threshold", 0.5)
no_leak_idx = next(
    (int(k) for k, v in label_meta.items() if k != "_decision_threshold" and ("No_Leak" in v or "Normal" in v)), 1
)
leak_idx = 1 - no_leak_idx

sess_xgb = ort.InferenceSession(str(MODEL_DIR / "xgboost_classifier.onnx"), providers=["CPUExecutionProvider"])
if_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")

# Test files
files = sorted(AUDIO_DIR.glob("*.wav"))
leak_files = [f for f in files if "Leak" in f.name or "leak" in f.name.lower()][:3]
noleak_files = [f for f in files if f not in leak_files][:3]
test_files = leak_files + noleak_files

print(f"Threshold: {THRESHOLD:.3f}\n")
print(f"{'File':<55} {'True':<10} {'16k_pred':<10} {'16k_conf':<10} {'3k2_pred':<10} {'3k2_conf':<10} {'16k_OOD':<10} {'3k2_OOD':<10}")
print("-" * 140)

results = []

for f in test_files:
    pcm, sr = librosa.load(str(f), sr=16000, mono=True)
    true_label = "Leak" if "Leak" in f.name or "leak" in f.name.lower() else "No_Leak"

    # Feature extraction
    feat_16k = extract_features_with_meta(pcm, sr=16000, pipe_material="PVC", pressure_bar=3.0).reshape(1, -1)
    pcm_3k2 = signal.resample(pcm, int(len(pcm) * 3200 / 16000))
    pcm_3k2_back = signal.resample(pcm_3k2, len(pcm))
    feat_3k2 = extract_features_with_meta(pcm_3k2_back, sr=16000, pipe_material="PVC", pressure_bar=3.0).reshape(1, -1)

    # OOD (joblib)
    if_16k = float(if_model.decision_function(feat_16k[:, :39])[0])
    if_3k2 = float(if_model.decision_function(feat_3k2[:, :39])[0])
    ood_16k = "REJECT" if if_model.predict(feat_16k[:, :39])[0] == -1 else "PASS"
    ood_3k2 = "REJECT" if if_model.predict(feat_3k2[:, :39])[0] == -1 else "PASS"

    # Classification
    res_16k = sess_xgb.run(None, {"float_input": feat_16k.astype(np.float32)})
    res_3k2 = sess_xgb.run(None, {"float_input": feat_3k2.astype(np.float32)})
    p16 = np.array(list(res_16k[1][0].values()), dtype=np.float32) if isinstance(res_16k[1][0], dict) else np.array(res_16k[1][0], dtype=np.float32)
    p32 = np.array(list(res_3k2[1][0].values()), dtype=np.float32) if isinstance(res_3k2[1][0], dict) else np.array(res_3k2[1][0], dtype=np.float32)

    pred_16k = "No_Leak" if p16[no_leak_idx] >= THRESHOLD else "Leak"
    pred_3k2 = "No_Leak" if p32[no_leak_idx] >= THRESHOLD else "Leak"
    conf_16k = float(p16[no_leak_idx if pred_16k == "No_Leak" else leak_idx])
    conf_3k2 = float(p32[no_leak_idx if pred_3k2 == "No_Leak" else leak_idx])

    print(f"{f.name:<55} {true_label:<10} {pred_16k:<10} {conf_16k:.3f}      {pred_3k2:<10} {conf_3k2:.3f}      {ood_16k:<10} {ood_3k2:<10}")

    results.append({
        "file": f.name, "true": true_label,
        "pred_16k": pred_16k, "pred_3k2": pred_3k2,
        "ood_16k": ood_16k, "ood_3k2": ood_3k2,
    })

acc_16k = sum(1 for r in results if r["pred_16k"] == r["true"]) / len(results)
acc_3k2 = sum(1 for r in results if r["pred_3k2"] == r["true"]) / len(results)
ood_16k = sum(1 for r in results if r["ood_16k"] == "REJECT") / len(results)
ood_3k2 = sum(1 for r in results if r["ood_3k2"] == "REJECT") / len(results)

print("\n" + "=" * 70)
print(f"Accuracy @ 16 kHz:  {acc_16k:.1%}")
print(f"Accuracy @ 3.2k:    {acc_3k2:.1%}")
print(f"OOD rejection @ 16k: {ood_16k:.1%}")
print(f"OOD rejection @ 3.2k: {ood_3k2:.1%}")
print("=" * 70)
