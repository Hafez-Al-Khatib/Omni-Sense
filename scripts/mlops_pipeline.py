"""
Omni-Sense MLOps Pipeline
===========================
Automated train → evaluate → promote/rollback pipeline.

Usage:
    python scripts/mlops_pipeline.py \
        --embeddings data/synthesized/embeddings.parquet \
        --output-dir iep2/models \
        --mlflow-tracking-uri http://localhost:5000 \
        --promotion-threshold-f1 0.85 \
        --promotion-threshold-auc 0.80

Pipeline stages:
    1. Train new candidate models (IF + XGBoost)
    2. Evaluate against test set + golden dataset
    3. Compare with current production model metrics
    4. Promote if candidate beats thresholds, rollback otherwise
    5. Archive old model artifacts
"""

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime
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
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

EMBEDDING_COLS = [f"embedding_{i}" for i in range(1024)]
PIPE_MATERIAL_MAP = {"PVC": 0, "Steel": 1, "Cast_Iron": 2}


def prepare_features(df):
    embeddings = df[EMBEDDING_COLS].values.astype(np.float32)
    pipe_encoded = df["pipe_material"].map(PIPE_MATERIAL_MAP).fillna(0).values.reshape(-1, 1)
    pressure = df["pressure_bar"].fillna(3.0).values.reshape(-1, 1)
    X = np.hstack([embeddings, pipe_encoded, pressure]).astype(np.float32)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    return X, y, embeddings


def train_candidate(X_train, y_train, X_val, y_val, emb_train, contamination=0.05):
    """Train candidate Isolation Forest + XGBoost models."""
    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=200, contamination=contamination,
        max_samples="auto", random_state=42, n_jobs=-1,
    )
    iforest.fit(emb_train)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    return iforest, xgb_model


def evaluate_model(xgb_model, X_test, y_test):
    """Evaluate and return metrics dict."""
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def evaluate_golden_dataset(xgb_model, iforest, golden_path):
    """Evaluate candidate on golden dataset if available."""
    if not golden_path.exists():
        print(f"  [WARN] Golden dataset not found at {golden_path}, skipping.")
        return None

    df = pd.read_csv(golden_path)
    X, y, embeddings = prepare_features(df)

    # OOD check: all golden samples should be in-distribution
    if_scores = iforest.decision_function(embeddings)
    ood_count = int(np.sum(if_scores < 0))

    # Classification metrics
    metrics = evaluate_model(xgb_model, X, y)
    metrics["golden_ood_false_positives"] = ood_count
    metrics["golden_ood_fp_rate"] = ood_count / len(df) if len(df) > 0 else 0.0

    return metrics


def load_current_metrics(output_dir):
    """Load metrics from current production model."""
    metrics_path = Path(output_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def promote_model(iforest, xgb_model, metrics, output_dir):
    """Promote candidate to production: save models and metrics."""
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Archive old models
    archive_dir = output_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    for artifact in ["isolation_forest.joblib", "xgboost_classifier.joblib", "metrics.json"]:
        src = output_dir / artifact
        if src.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, archive_dir / artifact)

    # Save new models
    joblib.dump(iforest, output_dir / "isolation_forest.joblib")
    joblib.dump(xgb_model, output_dir / "xgboost_classifier.joblib")

    # Export ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # IF ONNX
        initial_type = [("float_input", FloatTensorType([None, 1024]))]
        onnx_model = convert_sklearn(iforest, initial_types=initial_type)
        with open(output_dir / "isolation_forest.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # XGBoost ONNX
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from skl2onnx import update_registered_converter

        update_registered_converter(
            xgb.XGBClassifier, "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        initial_type = [("float_input", FloatTensorType([None, 1026]))]
        onnx_model = convert_sklearn(xgb_model, initial_types=initial_type)
        with open(output_dir / "xgboost_classifier.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
    except ImportError:
        print("  [WARN] skl2onnx not available, skipping ONNX export.")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Models promoted to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Omni-Sense MLOps Pipeline")
    parser.add_argument("--embeddings", type=str, default="data/synthesized/embeddings.parquet")
    parser.add_argument("--output-dir", type=str, default="iep2/models")
    parser.add_argument("--golden-dataset", type=str, default="data/golden/golden_dataset_v1.csv")
    parser.add_argument("--experiment-name", type=str, default="omni-sense-training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--promotion-threshold-f1", type=float, default=0.85)
    parser.add_argument("--promotion-threshold-auc", type=float, default=0.80)
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    output_dir = Path(args.output_dir)
    golden_path = Path(args.golden_dataset)

    if not embeddings_path.exists():
        print(f"[ERROR] Embeddings not found: {embeddings_path}")
        sys.exit(1)

    print("=" * 60)
    print("Omni-Sense MLOps Pipeline — Train / Evaluate / Promote")
    print("=" * 60)

    # Stage 1: Load and prepare data
    print("\n[Stage 1] Loading data...")
    df = pd.read_parquet(str(embeddings_path))
    X, y, embeddings_only = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    emb_train = X_train[:, :1024]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train,
    )

    # Stage 2: Train candidate
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"\n[Stage 2] Training candidate models (MLflow run: {run.info.run_id})...")
        iforest, xgb_model = train_candidate(X_tr, y_tr, X_val, y_val, emb_train, args.contamination)

        # Stage 3: Evaluate candidate
        print("\n[Stage 3] Evaluating candidate...")
        candidate_metrics = evaluate_model(xgb_model, X_test, y_test)
        print(f"  Candidate F1:      {candidate_metrics['f1']:.4f}")
        print(f"  Candidate ROC-AUC: {candidate_metrics['roc_auc']:.4f}")

        for k, v in candidate_metrics.items():
            mlflow.log_metric(f"candidate_{k}", v)

        # Golden dataset evaluation
        golden_metrics = evaluate_golden_dataset(xgb_model, iforest, golden_path)
        if golden_metrics:
            print(f"  Golden F1:         {golden_metrics['f1']:.4f}")
            print(f"  Golden OOD FPs:    {golden_metrics['golden_ood_false_positives']}")
            for k, v in golden_metrics.items():
                mlflow.log_metric(f"golden_{k}", v)

        # Stage 4: Promotion decision
        print("\n[Stage 4] Promotion decision...")
        current_metrics = load_current_metrics(output_dir)

        meets_f1 = candidate_metrics["f1"] >= args.promotion_threshold_f1
        meets_auc = candidate_metrics["roc_auc"] >= args.promotion_threshold_auc

        if current_metrics:
            beats_current_f1 = candidate_metrics["f1"] >= current_metrics.get("f1", 0)
            beats_current_auc = candidate_metrics["roc_auc"] >= current_metrics.get("roc_auc", 0)
            print(f"  Current F1:        {current_metrics.get('f1', 'N/A')}")
            print(f"  Current ROC-AUC:   {current_metrics.get('roc_auc', 'N/A')}")
        else:
            beats_current_f1 = True
            beats_current_auc = True
            print("  No current production model found (first deployment).")

        promote = meets_f1 and meets_auc and beats_current_f1 and beats_current_auc

        decision = "PROMOTE" if promote else "ROLLBACK"
        mlflow.log_param("decision", decision)
        mlflow.log_param("promotion_threshold_f1", args.promotion_threshold_f1)
        mlflow.log_param("promotion_threshold_auc", args.promotion_threshold_auc)

        print(f"\n  Decision: {decision}")
        print(f"    Meets F1 threshold ({args.promotion_threshold_f1}): {meets_f1}")
        print(f"    Meets AUC threshold ({args.promotion_threshold_auc}): {meets_auc}")
        print(f"    Beats current F1: {beats_current_f1}")
        print(f"    Beats current AUC: {beats_current_auc}")

        if promote:
            print("\n[Stage 5] Promoting candidate to production...")
            promote_model(iforest, xgb_model, candidate_metrics, output_dir)
            mlflow.log_artifacts(str(output_dir), artifact_path="production_models")
        else:
            print("\n[Stage 5] Keeping current production model (rollback).")
            print("  Candidate did not meet promotion criteria.")

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete! Decision: {decision}")
    print(f"{'=' * 60}")
    sys.exit(0 if promote else 1)


if __name__ == "__main__":
    main()
