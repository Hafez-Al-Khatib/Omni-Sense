"""
Train IEP2 models on REAL accelerometer data from Accelerometer/ directory.
Resamples to 3.2 kHz to match ESP32 ADXL345, extracts DSP features,
trains Isolation Forest + XGBoost, exports to ONNX.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.signal import resample

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Config ───────────────────────────────────────────────────────────────────
SRC_SR = 25_600          # Approximate sample rate of the CSV accelerometer data
DST_SR = 3_200           # Match ESP32 ADXL345
CLIP_DURATION = 5.0      # seconds per training clip
CLIP_SAMPLES = int(DST_SR * CLIP_DURATION)  # 16,000 samples
N_FEATURES = 39

def load_csv_accel(csv_path: Path) -> np.ndarray:
    """Load accelerometer CSV and return raw samples."""
    df = pd.read_csv(csv_path)
    # Column 1 is time, Column 2 is acceleration value
    samples = df.iloc[:, 1].values.astype(np.float32)
    return samples


def resample_to_3200(samples: np.ndarray) -> np.ndarray:
    """Resample from ~25.6 kHz to 3.2 kHz."""
    target_len = int(len(samples) * DST_SR / SRC_SR)
    return resample(samples, target_len).astype(np.float32)


def chop_clips(samples: np.ndarray, clip_samples: int, overlap: float = 0.5):
    """Chop long recording into overlapping clips."""
    hop = int(clip_samples * (1 - overlap))
    clips = []
    i = 0
    while i + clip_samples <= len(samples):
        clips.append(samples[i:i + clip_samples])
        i += hop
    return clips


# Import feature extractor from bridge
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "omni" / "edge" / "mqtt_bridge"))
from features import extract_features


def collect_labeled_clips(accel_dir: Path):
    """
    Walk Accelerometer/ directory and yield (samples, label) tuples.
    Labels derived from folder names.
    """
    label_map = {
        "No-leak": "Healthy",
        "Gasket Leak": "Leak",
        "Orifice Leak": "Leak",
        "Circumferential Crack": "Crack",
        "Longitudinal Crack": "Crack",
    }

    all_clips = []
    all_labels = []

    for csv_path in sorted(accel_dir.rglob("*.csv")):
        # Derive label from parent folder name
        folder_name = csv_path.parent.name
        label = label_map.get(folder_name)
        if label is None:
            print(f"  [SKIP] Unknown folder: {folder_name}")
            continue

        print(f"  Loading {csv_path.name} -> {label}")
        samples = load_csv_accel(csv_path)
        samples_3200 = resample_to_3200(samples)
        clips = chop_clips(samples_3200, CLIP_SAMPLES, overlap=0.5)
        print(f"    {len(samples)} samples @ ~25.6kHz -> {len(samples_3200)} @ 3.2kHz -> {len(clips)} clips")

        all_clips.extend(clips)
        all_labels.extend([label] * len(clips))

    return all_clips, all_labels


def main():
    accel_dir = Path("Accelerometer")
    output_dir = Path("iep2/models/accel")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training IEP2 on REAL accelerometer data")
    print("=" * 60)

    # ── Load and chop ──
    print("\n[1/5] Loading accelerometer CSVs...")
    clips, labels = collect_labeled_clips(accel_dir)
    print(f"  Total clips: {len(clips)}")
    print(f"  Labels: {pd.Series(labels).value_counts().to_dict()}")

    # ── Extract features ──
    print("\n[2/5] Extracting DSP features at 3.2 kHz...")
    features_list = []
    for i, clip in enumerate(clips):
        feat = extract_features(clip, sr=DST_SR)
        features_list.append(feat)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(clips)}...")
    features = np.array(features_list, dtype=np.float32)

    # Build DataFrame
    df = pd.DataFrame(features, columns=[f"embedding_{i}" for i in range(N_FEATURES)])
    df["label"] = labels
    df["pipe_material"] = "PVC"
    df["pressure_bar"] = 3.0

    # ── Prepare feature matrix ──
    print("\n[3/5] Preparing feature matrix...")
    embedding_cols = [f"embedding_{i}" for i in range(N_FEATURES)]
    embeddings = df[embedding_cols].values.astype(np.float32)
    pipe_encoded = np.zeros((len(df), 1), dtype=np.float32)  # PVC = 0
    pressure = np.full((len(df), 1), 3.0, dtype=np.float32)
    X = np.hstack([embeddings, pipe_encoded, pressure])

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    print(f"  Classes: {list(le.classes_)}")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    emb_train = X_train[:, :N_FEATURES]
    emb_test = X_test[:, :N_FEATURES]

    # ── Train Isolation Forest on Healthy only ──
    print("\n[4/5] Training Isolation Forest on Healthy clips...")
    healthy_mask = y_train == le.transform(["Healthy"])[0]
    healthy_embeddings = emb_train[healthy_mask]

    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        max_samples="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(healthy_embeddings)

    train_scores = iforest.decision_function(healthy_embeddings)
    print(f"  Healthy score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(f"  Healthy score mean:  {train_scores.mean():.4f}")
    print(f"  Healthy score std:   {train_scores.std():.4f}")

    # Evaluate IF on test set
    test_scores = iforest.decision_function(emb_test)
    test_healthy_mask = y_test == le.transform(["Healthy"])[0]
    print(f"  Test Healthy scores: mean={test_scores[test_healthy_mask].mean():.4f}")
    print(f"  Test Fault scores:   mean={test_scores[~test_healthy_mask].mean():.4f}")

    # Suggest threshold
    suggested_threshold = train_scores.mean() - 2 * train_scores.std()
    print(f"  Suggested OOD threshold (mean - 2*std): {suggested_threshold:.4f}")

    # ── Train XGBoost ──
    print("\n[5/5] Training XGBoost classifier...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(le.classes_),
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    print("\n  Test Results:")
    print(f"    Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"    F1:        {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"    Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"    Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    # ── Export ──
    print("\n[6/6] Exporting to ONNX...")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    # IF ONNX
    if_onnx = convert_sklearn(
        iforest,
        initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))],
        target_opset={"": 17, "ai.onnx.ml": 3}
    )
    with open(output_dir / "isolation_forest.onnx", "wb") as f:
        f.write(if_onnx.SerializeToString())
    print("  Isolation Forest -> isolation_forest.onnx")

    # XGB ONNX
    update_registered_converter(
        xgb.XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    xgb_onnx = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, N_FEATURES + 2]))],
        target_opset={"": 17, "ai.onnx.ml": 3}
    )
    with open(output_dir / "xgboost_classifier.onnx", "wb") as f:
        f.write(xgb_onnx.SerializeToString())
    print("  XGBoost -> xgboost_classifier.onnx")

    # Label map
    label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Threshold
    with open(output_dir / "threshold.json", "w") as f:
        json.dump({"threshold": float(suggested_threshold)}, f, indent=2)

    # Joblib fallbacks
    import joblib
    joblib.dump(iforest, output_dir / "isolation_forest.joblib")
    joblib.dump(model, output_dir / "xgboost_classifier.joblib")

    print(f"\n{'=' * 60}")
    print(f"Models saved to: {output_dir}")
    print(f"  Suggested OOD threshold: {suggested_threshold:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
