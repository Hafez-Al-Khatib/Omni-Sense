"""
Omni-Sense Model Training Pipeline
====================================
Trains the Isolation Forest (OOD detector) and XGBoost (classifier)
on vibration feature vectors, with MLflow experiment tracking and ONNX export.

Usage:
    python scripts/train_models.py \
        --embeddings data/synthesized/embeddings.parquet \
        --output-dir iep2/models \
        --experiment-name omni-sense-training

Pipeline:
    1. Load embeddings + metadata
    2. Train Isolation Forest on in-distribution data only
    3. Train XGBoost on embeddings + metadata features
    4. Evaluate on held-out test set
    5. Export both models to ONNX format
    6. Log everything to MLflow
"""

import argparse
import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Feature Engineering ────────────────────────────────────────────────────

N_FEATURES = 208  # must match iep1/app/feature_extractor.py N_FEATURES
EMBEDDING_COLS = [f"embedding_{i}" for i in range(N_FEATURES)]

PIPE_MATERIAL_MAP = {
    "PVC":       0,
    "Steel":     1,
    "Cast_Iron": 2,
}


def prepare_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Prepare feature matrix and labels from embeddings DataFrame.

    Features = [embedding_0..207, pipe_material_encoded, pressure_bar]
              = 210 dimensions total (208 vibration features + 2 metadata)

    Labels: auto-detected as binary (2 classes) or multi-class (>2 classes).
    pipe_material values outside the known map default to 0 (PVC).
    pressure_bar NaNs default to 3.0 bar.

    Returns:
        X             : (n_samples, 1026) float32 feature matrix
        y             : (n_samples,) int labels
        embeddings    : (n_samples, 1024) float32 — used for Isolation Forest
        label_encoder : fitted LabelEncoder (class order → index mapping)
    """
    # Select only the feature columns that exist (handles any N_FEATURES)
    feature_cols = [c for c in EMBEDDING_COLS if c in df.columns]
    embeddings = df[feature_cols].values.astype(np.float32)

    pipe_encoded = (
        df["pipe_material"]
        .map(PIPE_MATERIAL_MAP)
        .fillna(0)
        .values.reshape(-1, 1)
        .astype(np.float32)
    )
    pressure = df["pressure_bar"].fillna(3.0).values.reshape(-1, 1).astype(np.float32)

    X = np.hstack([embeddings, pipe_encoded, pressure])

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}): {list(le.classes_)}")

    return X, y, embeddings, le


# ─── Isolation Forest Training ───────────────────────────────────────────────

def train_isolation_forest(
    embeddings_train: np.ndarray,
    contamination: float = 0.05,
) -> IsolationForest:
    """
    Train Isolation Forest for Out-of-Distribution detection.
    Trained ONLY on in-distribution data (both leak and background from target env).
    """
    print(f"\n  Training Isolation Forest (contamination={contamination})...")
    print(f"  Training samples: {embeddings_train.shape[0]}")

    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(embeddings_train)

    # Evaluate on training data
    train_scores = iforest.decision_function(embeddings_train)
    train_preds = iforest.predict(embeddings_train)
    n_anomalies = np.sum(train_preds == -1)

    print(f"  Training anomalies detected: {n_anomalies}/{len(train_preds)}")
    print(f"  Score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    print(f"  Score mean: {train_scores.mean():.4f}")

    return iforest


def export_isolation_forest_onnx(
    model: IsolationForest,
    output_path: Path,
    n_features: int = 1024,
):
    """Export Isolation Forest to ONNX format."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  Isolation Forest exported to: {output_path}")


# ─── XGBoost Training ────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with early stopping.

    Automatically selects binary or multi-class objective based on the
    number of unique label values in y_train.
    """
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2

    objective = "multi:softprob" if is_multiclass else "binary:logistic"
    eval_metric = "mlogloss" if is_multiclass else "logloss"

    print(f"\n  Training XGBoost classifier...")
    print(f"  Mode: {'multi-class' if is_multiclass else 'binary'} ({n_classes} classes)")
    print(f"  Training: {X_train.shape[0]} samples | Validation: {X_val.shape[0]} samples")
    print(f"  Feature dim: {X_train.shape[1]}")

    kwargs = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective=objective,
        eval_metric=eval_metric,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    if is_multiclass:
        kwargs["num_class"] = n_classes

    model = xgb.XGBClassifier(**kwargs)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    return model


def evaluate_xgboost(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Evaluate XGBoost and return metrics dict.
    Handles both binary and multi-class automatically.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    n_classes = len(label_encoder.classes_)

    if n_classes == 2:
        roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        roc_auc = float(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        )

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall":    float(recall_score(y_test, y_pred, average="weighted")),
        "roc_auc":   roc_auc,
    }

    print(f"\n  XGBoost Test Results:")
    print(f"  {'─' * 40}")
    for k, v in metrics.items():
        print(f"    {k:>12}: {v:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(label_encoder.classes_),
    ))

    return metrics


def export_xgboost_onnx(
    model: xgb.XGBClassifier,
    output_path: Path,
    n_features: int = 1026,
):
    """Export XGBoost model to ONNX format."""
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
    )
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
        convert_xgboost,
    )

    update_registered_converter(
        xgb.XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"  XGBoost exported to: {output_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest & XGBoost models for Omni-Sense."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/synthesized/embeddings.parquet",
        help="Path to embeddings Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="iep2/models",
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="omni-sense-training",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Isolation Forest contamination parameter.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (default: local mlruns directory).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    embeddings_path = Path(args.embeddings)

    if not embeddings_path.exists():
        print(f"[ERROR] Embeddings not found: {embeddings_path}")
        print("  Run extract_embeddings.py first.")
        return

    print("=" * 60)
    print("Omni-Sense Model Training Pipeline")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/6] Loading embeddings...")
    df = pd.read_parquet(str(embeddings_path))
    print(f"  Loaded {len(df)} samples.")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # ── Prepare features ──
    print("\n[2/6] Preparing features...")
    X, y, embeddings_only, label_encoder = prepare_features(df)
    n_feat = embeddings_only.shape[1]
    print(f"  Feature matrix: {X.shape} ({n_feat} vibration features + 2 metadata)")
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"    class {cls_idx} ({label_encoder.classes_[cls_idx]}): {cnt} samples")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    emb_train = X_train[:, :embeddings_only.shape[1]]

    # Further split train into train/val for XGBoost early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train,
    )

    # ── MLflow setup ──
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="train_if_xgb") as run:
        print(f"\n  MLflow Run ID: {run.info.run_id}")

        # ── Train Isolation Forest ──
        print("\n[3/6] Training Isolation Forest...")
        iforest = train_isolation_forest(emb_train, contamination=args.contamination)

        mlflow.log_param("if_contamination", args.contamination)
        mlflow.log_param("if_n_estimators", 200)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Log IF scores on test embeddings
        test_emb = X_test[:, :embeddings_only.shape[1]]
        if_scores = iforest.decision_function(test_emb)
        mlflow.log_metric("if_score_mean", float(np.mean(if_scores)))
        mlflow.log_metric("if_score_std", float(np.std(if_scores)))

        # ── Train XGBoost ──
        print("\n[4/6] Training XGBoost...")
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)

        # ── Evaluate ──
        print("\n[5/6] Evaluating...")
        metrics = evaluate_xgboost(xgb_model, X_test, y_test, label_encoder)

        for k, v in metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)

        mlflow.log_params({
            "xgb_n_estimators": 500,
            "xgb_max_depth": 6,
            "xgb_learning_rate": 0.05,
        })

        # ── Export to ONNX ──
        print("\n[6/6] Exporting models to ONNX...")
        if_onnx_path = output_dir / "isolation_forest.onnx"
        xgb_onnx_path = output_dir / "xgboost_classifier.onnx"

        n_emb = embeddings_only.shape[1]
        export_isolation_forest_onnx(iforest, if_onnx_path, n_features=n_emb)
        export_xgboost_onnx(xgb_model, xgb_onnx_path, n_features=n_emb + 2)

        # Also save as joblib fallback
        import joblib
        joblib.dump(iforest, output_dir / "isolation_forest.joblib")
        joblib.dump(xgb_model, output_dir / "xgboost_classifier.joblib")

        # Save label map so IEP2 can decode class indices at inference time.
        # Format: {"0": "Circumferential_Crack", "1": "Gasket_Leak", ...}
        label_map = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}
        label_map_path = output_dir / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"  Label map saved: {label_map_path}")
        mlflow.log_artifact(str(label_map_path))

        # Save training centroid (vibration features only, no metadata).
        centroid = emb_train.mean(axis=0)
        centroid_path = output_dir / "centroid.npy"
        np.save(centroid_path, centroid)
        print(f"  Centroid saved : {centroid_path}")
        mlflow.log_artifact(str(centroid_path))

        # Log model artifacts to MLflow
        mlflow.log_artifacts(str(output_dir), artifact_path="models")

        # Save metrics summary
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Models: {output_dir}")
    print(f"  MLflow: {args.mlflow_tracking_uri}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
