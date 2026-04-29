# IEP2 Model Report

**Trained:** 2026-04-30
**Pipeline:** `scripts/train_models.py` (recording-level aggregation, 5-fold stratified CV)
**Dataset source:** `data/synthesized/eep_features.parquet` (39-d EEP-aligned features)
**Feature extractor:** `omni/eep/features.py` (39-d pure NumPy — matches production EEP inference)

---

## Dataset

| Stage | Count |
|---|---|
| Raw clips (5-second windows, augmented) | **4,192** |
| Unique source recordings (after `groupby(source_wav).mean()`) | **1,336** |
| Recording-level class balance (binary) | 64 Leak / 1,272 No_Leak |
| Majority-class baseline accuracy | **95.21%** |

Hard negatives — `Normal_Operation` clips drawn from MIMII pump audio
(`scripts/extract_mimii_negatives.py`) — make up 1,256 of the 1,272
No_Leak recordings. Without them the dataset was 64 Leak vs 16 No_Leak
and the model could not exceed the 80% majority baseline.

---

## Cross-Validation Results

5-fold stratified CV at the recording level (no clip-level leakage).

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9907** |
| Accuracy @ default threshold (0.50) | 98.73% |
| Accuracy @ class-prior threshold (0.952) | 98.73% |
| Accuracy @ Youden-J optimal (1.00) | 98.80% |
| F1 (weighted) | 0.9879 |

### Classification report (default threshold)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Leak    | 0.80 | 0.98 | 0.88 | 64 |
| No_Leak | 1.00 | 0.99 | 0.99 | 1,272 |

Operationally critical: **Leak recall = 0.98** — the model catches 63/64
leak recordings in CV. The 0.80 Leak precision means about 1 in 5 leak
alarms is a false positive (operators investigate; no harm done). This is
the correct trade-off for a safety-critical alarm system.

### Random Forest companion
OOB score: **0.9880** (matches XGBoost). Used for ensemble fusion in EEP.

---

## Production Configuration

- **Decision threshold:** 0.952 (= class prior of No_Leak)
- **Threshold persisted in:** `iep2/models/label_map.json` under
  `_decision_threshold` — IEP2 must use this value, not the
  default 0.5, when calling `predict()`.
- **Models loaded by IEP2:**
  - `xgboost_classifier.joblib` (primary classifier)
  - `rf_classifier.joblib` (ensemble companion)
  - `isolation_forest.joblib` (legacy OOD; retained alongside the
    CNN autoencoder)
  - `centroid.npy` (drift monitor reference)

---

## Known issues

**ONNX export crashes on Windows** (`onnxmltools` / `xgboost` DLL
incompatibility — segfault). The joblib artifacts are saved as a fallback.
`skl2onnx` and `onnxmltools` are included in `iep2/requirements.txt` so the
Linux container can run the ONNX export without extra installs:

```bash
docker compose run --rm iep2 python /app/../scripts/export_onnx.py \
  --models-dir /app/models
```

---

## Reproduce

```bash
# 1. Augment audio clips (LeakDB + MIMII hard negatives)
py -3.12 scripts/augment_data.py --input-dir Processed_audio_16k --output-dir data/synthesized
py -3.12 scripts/extract_mimii_negatives.py --output-dir data/synthesized

# 2. Extract 39-d EEP-aligned features (matches production inference)
py -3.12 scripts/extract_eep_features.py \
  --input-dir data/synthesized \
  --output    data/synthesized/eep_features.parquet

# 3. Train models
py -3.12 scripts/train_models.py \
  --embeddings data/synthesized/eep_features.parquet \
  --output-dir iep2/models

# 4. Export ONNX (must run in Linux / Docker due to Windows DLL issue)
docker compose run --rm iep2 python /app/../scripts/export_onnx.py \
  --models-dir /app/models
```
