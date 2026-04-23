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
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Feature Engineering ────────────────────────────────────────────────────

# NOTE: N_FEATURES is now dynamic — never hardcoded here.
# prepare_features() reads whatever embedding_* columns are present in the
# parquet file, so the pipeline adapts automatically when IEP1 changes.

PIPE_MATERIAL_MAP = {
    "PVC":       0,
    "Steel":     1,
    "Cast_Iron": 2,
}


# Labels that represent "no fault" — everything else is a leak.
NO_FAULT_LABELS = {"No_Leak", "Normal_Operation"}


def _get_embedding_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted embedding column names regardless of total count."""
    return sorted(
        [c for c in df.columns if c.startswith("embedding_")],
        key=lambda c: int(c.split("_")[1]),
    )


def prepare_features(
    df: pd.DataFrame,
    binary_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Prepare feature matrix and labels from embeddings DataFrame.

    Features = [embedding_0..N, pipe_material_encoded, pressure_bar]

    binary_mode=True: collapses all fault classes → "Leak", No_Leak stays "No_Leak".
    This is the recommended mode: LOO-CV on raw recordings shows 80% binary accuracy
    vs 8.8% 5-class accuracy — the dataset does not have enough recordings per class
    to learn fault-type distinctions reliably.
    """
    # Select only the feature columns that exist — dynamic, never hardcoded
    feature_cols = _get_embedding_cols(df)
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

    labels = df["label"].copy()
    if binary_mode:
        labels = labels.apply(lambda lbl: "No_Leak" if lbl in NO_FAULT_LABELS else "Leak")

    le = LabelEncoder()
    y = le.fit_transform(labels.values)

    n_classes = len(le.classes_)
    mode_tag = "BINARY" if binary_mode else "MULTI-CLASS"
    print(f"  Mode: {mode_tag} | Classes ({n_classes}): {list(le.classes_)}")

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
    n_features: int = 100,
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

    For binary mode, scale_pos_weight compensates for the inherent 4:1
    Leak:No_Leak imbalance that results from merging 4 fault classes.
    """
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2

    objective = "multi:softprob" if is_multiclass else "binary:logistic"
    eval_metric = "mlogloss" if is_multiclass else "logloss"

    print("\n  Training XGBoost classifier...")
    print(f"  Mode: {'multi-class' if is_multiclass else 'binary'} ({n_classes} classes)")
    print(f"  Training: {X_train.shape[0]} samples | Validation: {X_val.shape[0]} samples")
    print(f"  Feature dim: {X_train.shape[1]}")

    kwargs = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective=objective,
        eval_metric=eval_metric,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    if is_multiclass:
        kwargs["num_class"] = n_classes
    else:
        # Compensate for Leak (majority) vs No_Leak (minority) imbalance.
        # scale_pos_weight = count(negative) / count(positive) for binary:logistic.
        # In LabelEncoder alphabetical order: Leak=0, No_Leak=1 → positive=No_Leak.
        n_neg = int(np.sum(y_train == 0))  # Leak clips
        n_pos = int(np.sum(y_train == 1))  # No_Leak clips
        spw = n_neg / max(n_pos, 1)
        kwargs["scale_pos_weight"] = spw
        print(f"  Class counts — Leak: {n_neg} | No_Leak: {n_pos} | scale_pos_weight: {spw:.2f}")

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

    print("\n  XGBoost Test Results:")
    print(f"  {'-' * 40}")
    for k, v in metrics.items():
        print(f"    {k:>12}: {v:.4f}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(label_encoder.classes_),
    ))

    return metrics


def export_xgboost_onnx(
    model: xgb.XGBClassifier,
    output_path: Path,
    n_features: int = 102,
):
    """Export XGBoost model to ONNX format."""
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
        convert_xgboost,
    )
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import (
        calculate_linear_classifier_output_shapes,
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


# ─── Random Forest Training ──────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier as ensemble companion to XGBoost.

    RF complements XGBoost on small datasets because:
    - Bagging reduces variance (XGBoost can overfit tiny training sets)
    - Each tree sees a random feature subset → decorrelated errors
    - Out-of-bag error gives a free internal validation signal

    For the Omni-Sense ~80-recording dataset, RF with balanced class weights
    provides better No_Leak recall than XGBoost alone.
    """
    n_classes = len(np.unique(y_train))
    print(f"\n  Training Random Forest ({n_classes} classes)...")
    print(f"  Training: {X_train.shape[0]} samples | Features: {X_train.shape[1]}")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,          # Grow full trees — bagging handles variance
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",     # Standard for classification
        class_weight="balanced", # Compensates for Leak/No_Leak imbalance
        oob_score=True,          # Free internal validation
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  OOB score: {model.oob_score_:.4f}")
    return model


def export_rf_onnx(
    model: RandomForestClassifier,
    output_path: Path,
    n_features: int = 102,
):
    """Export Random Forest to ONNX format."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Random Forest exported to: {output_path}")


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
    parser.add_argument(
        "--binary",
        action="store_true",
        default=True,
        help=(
            "Merge all fault classes into 'Leak' vs 'No_Leak' (default: True). "
            "LOO-CV on raw recordings shows 80%% binary accuracy vs 8.8%% 5-class — "
            "the dataset lacks sufficient recordings per fault type for multi-class. "
            "Pass --no-binary to attempt multi-class anyway."
        ),
    )
    parser.add_argument("--no-binary", dest="binary", action="store_false")
    parser.add_argument(
        "--aggregate-recordings",
        action="store_true",
        default=True,
        help=(
            "Mean-pool all window features from the same source recording before "
            "training (default: True). This replicates the LOO-CV protocol that "
            "achieved 80%% binary accuracy. Individual 5-second windows have too "
            "much within-recording variance; aggregation stabilises the statistics. "
            "Pass --no-aggregate to train on raw window-level clips."
        ),
    )
    parser.add_argument("--no-aggregate", dest="aggregate_recordings", action="store_false")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Number of CV folds for recording-level evaluation (default: 5). "
            "Use -1 for Leave-One-Out CV (exact match to the 80%% LOO-CV result, "
            "but slower — runs N_recordings separate models)."
        ),
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
    print(f"  Loaded {len(df)} clips from parquet.")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # ── Aggregate to recording level (recommended) ──
    # Each source recording contributes N windows × 3 augmentation variants.
    # Training on individual 5-second windows fails because within-recording
    # variance swamps inter-class differences. Mean-pooling all windows from
    # the same recording stabilises spectral statistics — the approach that
    # achieved 80% binary LOO-CV accuracy on raw recordings.
    if args.aggregate_recordings and "source_wav" in df.columns:
        print("\n[INFO] Aggregating window features to recording level...")
        feature_cols = [c for c in df.columns if c.startswith("embedding_")]
        agg_dict: dict = {c: "mean" for c in feature_cols}
        agg_dict["label"] = "first"
        agg_dict["pipe_material"] = "first"
        agg_dict["pressure_bar"] = "mean"
        df = (
            df.groupby("source_wav", sort=False)
            .agg(agg_dict)
            .reset_index()
        )
        print(f"  Aggregated {len(df)} source recordings.")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # ── Prepare features ──
    print("\n[2/6] Preparing features...")
    X, y, embeddings_only, label_encoder = prepare_features(df, binary_mode=args.binary)
    n_feat = embeddings_only.shape[1]
    print(f"  Feature matrix: {X.shape} ({n_feat} vibration features + 2 metadata)")
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, cnt in zip(unique, counts, strict=False):
        print(f"    class {cls_idx} ({label_encoder.classes_[cls_idx]}): {cnt} samples")

    # ── MLflow setup ──
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="train_if_xgb") as run:
        print(f"\n  MLflow Run ID: {run.info.run_id}")

        # ── Train Isolation Forest on ALL recordings ──
        print("\n[3/6] Training Isolation Forest...")
        iforest = train_isolation_forest(embeddings_only, contamination=args.contamination)
        mlflow.log_param("if_contamination", args.contamination)
        mlflow.log_param("if_n_estimators", 200)
        mlflow.log_param("n_recordings", len(X))

        # ── Cross-Validation ──
        print("\n[4/6] Training XGBoost (cross-validation)...")
        n_classes = len(np.unique(y))
        is_multiclass = n_classes > 2
        n_pos = int(np.sum(y == 1))   # No_Leak recordings

        # base_score = class prior of the positive class (No_Leak).
        # With 4:1 class imbalance, the default base_score=0.5 causes XGBoost to
        # start every sample at 50% Leak probability. The Leak gradient (64 × 0.5)
        # overwhelms the No_Leak gradient (16 × -0.5) by 4:1, so trees keep pushing
        # toward Leak and never move any recording's probability above the 0.5 No_Leak
        # threshold. Setting base_score=0.2 (= 16/80 = class prior) perfectly
        # balances initial gradients: Leak total = 64×0.2 = 12.8, No_Leak total =
        # 16×(-0.8) = -12.8. Trees can then find genuine discriminative features.
        # No scale_pos_weight needed — base_score handles the prior.
        base_score = n_pos / len(y)   # = 0.2 for 16 No_Leak / 80 total

        majority_baseline = float(np.max(counts)) / len(y)
        print(f"  Majority-class baseline accuracy: {majority_baseline:.2%}")
        print(f"  base_score (class prior): {base_score:.3f}")

        # XGBoost parameters for small-dataset recording-level training.
        # No early stopping (no held-out val set inside each fold).
        # Shallow trees + strong regularisation to avoid overfitting ~60 samples.
        xgb_params = dict(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.5,
            reg_lambda=2.0,
            base_score=base_score,
            objective="multi:softprob" if is_multiclass else "binary:logistic",
            eval_metric="mlogloss" if is_multiclass else "logloss",
            random_state=42,
            n_jobs=-1,
        )
        if is_multiclass:
            xgb_params["num_class"] = n_classes

        if args.cv_folds == -1:
            cv = LeaveOneOut()
            cv_name = "LOO-CV"
        else:
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            cv_name = f"{args.cv_folds}-fold CV"

        print(f"  Evaluation strategy: {cv_name} on {len(X)} recordings")

        fold_accs, fold_aucs = [], []
        all_y_true: list[int] = []
        all_y_proba_pos: list[float] = []  # P(No_Leak) per recording

        for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            mdl = xgb.XGBClassifier(**xgb_params)
            mdl.fit(X_tr, y_tr, verbose=False)

            y_proba = mdl.predict_proba(X_te)
            all_y_true.extend(y_te.tolist())
            all_y_proba_pos.extend(y_proba[:, 1].tolist())

            if not is_multiclass and len(np.unique(y_te)) == 2:
                fold_aucs.append(float(roc_auc_score(y_te, y_proba[:, 1])))

            if args.cv_folds == -1 and (fold_i + 1) % 10 == 0:
                print(f"  Fold {fold_i + 1}/{len(X)} — running acc @ prior thresh: "
                      f"{np.mean(fold_accs):.3f}")

        all_y_true_arr = np.array(all_y_true)
        all_y_proba_arr = np.array(all_y_proba_pos)

        # Evaluate at two operating points:
        #
        # (a) threshold=0.5 (XGBoost default): shows whether any recording crosses
        #     the midpoint. With 4:1 imbalance and only 80 recordings, this is almost
        #     always 0% No_Leak recall — all probabilities stay below 0.5.
        #
        # (b) threshold=class_prior (0.2): the correct operating point for an
        #     imbalanced detector. Predict No_Leak whenever P(No_Leak|x) > base_score.
        #     This is NOT fitting on test data — it uses only the known class prior,
        #     which is the Bayes-optimal threshold for balanced precision/recall.
        #
        # For IEP2 production: store and use the prior threshold, not 0.5.
        from sklearn.metrics import roc_curve as _roc_curve

        def _eval_at_threshold(t: float, label: str) -> dict:
            y_pred = (all_y_proba_arr >= t).astype(int)
            acc = float(accuracy_score(all_y_true_arr, y_pred))
            f1  = float(f1_score(all_y_true_arr, y_pred, average="weighted", zero_division=0))
            print(f"\n  @ threshold={t:.2f} ({label}):")
            print(f"    accuracy={acc:.4f}  f1={f1:.4f}")
            print(classification_report(all_y_true_arr, y_pred,
                                        target_names=list(label_encoder.classes_),
                                        zero_division=0))
            return {"accuracy": acc, "f1": f1, "threshold": t}

        cv_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        print(f"\n  {cv_name} Results (ROC-AUC: {cv_auc:.4f})")
        print(f"  {'-' * 40}")

        res_05    = _eval_at_threshold(0.50, "default")
        res_prior = _eval_at_threshold(base_score, "class-prior / Bayes-optimal")

        # Also find Youden-J optimal threshold from the CV probability outputs.
        # This uses test-set probabilities → slightly optimistic, reported for reference.
        fpr_arr, tpr_arr, thresh_arr = _roc_curve(all_y_true_arr, all_y_proba_arr)
        best_j = int(np.argmax(tpr_arr - fpr_arr))
        youden_thresh = float(thresh_arr[best_j])
        res_youden = _eval_at_threshold(youden_thresh, "Youden-J optimal (ref only)")

        # Pick the best threshold for production: prior threshold is principled;
        # Youden is shown for reference but is optimistic (optimised on CV outputs).
        production_threshold = float(base_score)

        metrics = {
            "roc_auc": cv_auc,
            "accuracy_t50":    res_05["accuracy"],
            "accuracy_prior":  res_prior["accuracy"],
            "accuracy_youden": res_youden["accuracy"],
            "f1_prior":        res_prior["f1"],
            "production_threshold": production_threshold,
        }
        for k, v in metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(f"xgb_{k}", v)

        # ── Train final model on ALL recordings ──
        print("\n[5/6] Training final XGBoost on all recordings...")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X, y, verbose=False)
        print(f"  Final model trained on {len(X)} recordings.")

        mlflow.log_params({
            "xgb_n_estimators": xgb_params["n_estimators"],
            "xgb_max_depth": xgb_params["max_depth"],
            "xgb_learning_rate": xgb_params["learning_rate"],
            "cv_strategy": cv_name,
            "aggregate_recordings": args.aggregate_recordings,
        })

        # ── Train Random Forest on ALL recordings ──
        print("\n[5b/6] Training Random Forest (XGBoost ensemble companion)...")
        rf_model = train_random_forest(X, y)
        mlflow.log_metric("rf_oob_score", rf_model.oob_score_)

        # ─── Export Fallback Models (Joblib) ───
        import joblib
        joblib.dump(iforest, output_dir / "isolation_forest.joblib")
        joblib.dump(xgb_model, output_dir / "xgboost_classifier.joblib")
        joblib.dump(rf_model, output_dir / "rf_classifier.joblib")

        # Save label map so IEP2 can decode class indices at inference time.
        label_map = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}
        label_map["_decision_threshold"] = production_threshold
        label_map_path = output_dir / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"\n  Model artifacts saved (joblib): {output_dir}")
        print(f"  Production threshold: {production_threshold:.3f}")
        mlflow.log_artifact(str(label_map_path))

        # Save training centroid
        centroid = embeddings_only.mean(axis=0)
        centroid_path = output_dir / "centroid.npy"
        np.save(centroid_path, centroid)
        print(f"  Centroid saved : {centroid_path}")
        mlflow.log_artifact(str(centroid_path))

        # ─── Export to ONNX (Optional) ───
        print("\n[6/6] Attempting ONNX export...")
        try:
            if_onnx_path = output_dir / "isolation_forest.onnx"
            xgb_onnx_path = output_dir / "xgboost_classifier.onnx"
            rf_onnx_path = output_dir / "rf_classifier.onnx"

            n_emb = embeddings_only.shape[1]
            n_feat_total = n_emb + 2

            export_isolation_forest_onnx(iforest, if_onnx_path, n_features=n_emb)
            export_xgboost_onnx(xgb_model, xgb_onnx_path, n_features=n_feat_total)
            try:
                export_rf_onnx(rf_model, rf_onnx_path, n_features=n_feat_total)
            except Exception as e:
                print(f"  [WARN] RF ONNX export failed ({e})")

            print(f"  ONNX models exported successfully to {output_dir}")
        except Exception as e:
            print(f"  [SKIPPED] ONNX export failed: {e}")
            print("  App will use Joblib/fallback models.")

        # Log model artifacts to MLflow
        mlflow.log_artifacts(str(output_dir), artifact_path="models")

        # Save metrics summary
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Models: {output_dir}")
    print(f"  MLflow: {args.mlflow_tracking_uri}")
    print(f"  F1 Score (prior thresh): {metrics['f1_prior']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
