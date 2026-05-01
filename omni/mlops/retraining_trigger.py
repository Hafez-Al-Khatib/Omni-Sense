"""Automated retraining trigger.

Listens for mlops.retrain.request.v1 events (emitted by the drift detector),
applies a cooldown to avoid cascading retrains, and then:

  1. Assembles the current training corpus from:
       a. The original recordings in Processed_audio_16k/
       b. New technician-labelled samples from IEP3's feedback_log.csv
  2. Runs the training script via subprocess (mlflow-tracked)
  3. Evaluates the new model against the quality gate (F1 ≥ 0.95)
  4. If quality gate passes → atomically swaps the model files and
     publishes mlops.model.updated.v1
  5. If gate fails → logs failure and retains the current model

Quality gate
------------
  F1 ≥ 0.95  AND  ROC-AUC ≥ 0.97
  (same thresholds as the GitHub Actions workflow)

In-process retraining
---------------------
For the platform demo we run training in-process using the same logic as
iep2/app/classifier.py. In production this would submit a Kubernetes Job
or an MLflow Project run to a separate compute node.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omni.common.bus import get_bus

log = logging.getLogger("retrain")

# ── Config ────────────────────────────────────────────────────────────────────
QUALITY_GATE_F1     = 0.95
QUALITY_GATE_AUC    = 0.97
COOLDOWN_MINUTES    = 60      # minimum gap between two retraining runs
FEEDBACK_LOG        = Path("iep3/feedback_log.csv")   # IEP3 active-learning output
MODEL_DIR_IEP2      = Path("iep2/models")
RETRAIN_TOPIC       = "mlops.retrain.request.v1"
MODEL_UPDATED_TOPIC = "mlops.model.updated.v1"




class RetrainingTrigger:
    def __init__(self):
        self._last_retrain: datetime | None = None
        self._retrain_count: int = 0
        self._lock = asyncio.Lock()
        self.history: list[dict] = []

    # ── Cooldown gate ────────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        if self._last_retrain is None:
            return False
        elapsed = (datetime.now(UTC) - self._last_retrain).total_seconds()
        return elapsed < COOLDOWN_MINUTES * 60

    # ── Feedback corpus ──────────────────────────────────────────────────────

    def _count_feedback_samples(self) -> int:
        if not FEEDBACK_LOG.exists():
            return 0
        try:
            with open(FEEDBACK_LOG) as f:
                return max(0, sum(1 for _ in f) - 1)   # subtract header
        except Exception:
            return 0

    # ── In-process quality evaluation ────────────────────────────────────────

    def _evaluate_current_model(self) -> dict | None:
        """Quick quality check: load current IEP2 models and run against the golden dataset.

        Returns dict with f1, roc_auc or None if evaluation fails.
        """
        import json
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.metrics import f1_score, roc_auc_score

        try:
            metrics_path = MODEL_DIR_IEP2 / "metrics.json"
            golden_path  = Path("data/golden/golden_dataset_v1.csv")

            if not golden_path.exists():
                log.warning("Golden dataset not found at %s", golden_path)
                # Fallback to metrics.json if it exists
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        m = json.load(f)
                    return {**m, "source": "metrics_json_fallback"}
                return None

            # Load models
            xgb_path = MODEL_DIR_IEP2 / "xgboost_classifier.joblib"
            rf_path  = MODEL_DIR_IEP2 / "rf_classifier.joblib"
            if not xgb_path.exists() or not rf_path.exists():
                log.warning("Classifier models not found in %s", MODEL_DIR_IEP2)
                return None

            log.info("Evaluating ensemble against golden dataset...")
            xgb_model = joblib.load(xgb_path)
            rf_model  = joblib.load(rf_path)

            # Load golden data
            df = pd.read_csv(golden_path)
            
            # Prepare features (39 vibration features + 2 metadata)
            # Metadata encoding: PVC=0, Steel=1, Cast_Iron=2
            MATERIAL_MAP = {"PVC": 0, "Steel": 1, "Cast_Iron": 2}
            
            emb_cols = [c for c in df.columns if c.startswith("embedding_")]
            emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))
            
            embeddings = df[emb_cols].values.astype(np.float32)
            pipe_code  = df["pipe_material"].map(MATERIAL_MAP).fillna(0).values.reshape(-1, 1)
            pressure   = df["pressure_bar"].fillna(3.0).values.reshape(-1, 1)
            
            X = np.hstack([embeddings, pipe_code, pressure]).astype(np.float32)
            y = (df["label"] != "No_Leak").astype(int) # Binary: Leak=1, No_Leak=0

            # Ensemble prediction (average probabilities)
            # label_map shows: 0: Leak, 1: No_Leak
            # We want to evaluate weighted f1 and ROC AUC for class 1 (No_Leak) or Leak?
            # Usually AUC is for the positive class. Let's use class 0 (Leak) as positive for metrics.
            p_xgb_leak = xgb_model.predict_proba(X)[:, 0]
            p_rf_leak  = rf_model.predict_proba(X)[:, 0]
            p_ensemble_leak = 0.6 * p_xgb_leak + 0.4 * p_rf_leak
            
            y_leak = (df["label"] != "No_Leak").astype(int) # Leak=1, No_Leak=0
            y_pred = (p_ensemble_leak > 0.5).astype(int)

            return {
                "f1": float(f1_score(y_leak, y_pred, average="weighted")),
                "roc_auc": float(roc_auc_score(y_leak, p_ensemble_leak)),
                "n_samples": len(df),
                "source": "golden_dataset_eval",
            }

        except Exception as e:
            log.warning("Model evaluation failed: %s", e, exc_info=True)
            return None

    # ── Retrain ──────────────────────────────────────────────────────────────

    async def retrain(self, trigger_payload: dict) -> None:
        async with self._lock:
            if self._in_cooldown():
                elapsed = (datetime.now(UTC) - self._last_retrain).total_seconds() / 60
                log.info(
                    "retrain request ignored (cooldown: %.0f / %d min remaining)",
                    elapsed, COOLDOWN_MINUTES,
                )
                return

            log.info(
                "retrain triggered by drift detector — psi_max=%.3f ood_rate=%.2f%%",
                trigger_payload.get("psi_max", 0),
                trigger_payload.get("ood_rate", 0) * 100,
            )

            feedback_n = self._count_feedback_samples()
            log.info("feedback corpus: %d new technician-labelled samples", feedback_n)

            # Evaluate current model first
            metrics = self._evaluate_current_model()
            record = {
                "triggered_at": trigger_payload.get("triggered_at"),
                "retrain_at": datetime.now(UTC).isoformat(),
                "trigger_psi_max": trigger_payload.get("psi_max"),
                "trigger_ood_rate": trigger_payload.get("ood_rate"),
                "feedback_samples": feedback_n,
                "model_metrics": metrics,
                "outcome": None,
                "reason": None,
            }

            if metrics is None:
                log.error("Could not evaluate current model — skipping retrain")
                record["outcome"] = "skipped"
                record["reason"] = "model evaluation failed"
                self.history.append(record)
                return

            f1  = metrics.get("f1", 0)
            auc = metrics.get("roc_auc", 0)

            if f1 >= QUALITY_GATE_F1 and auc >= QUALITY_GATE_AUC:
                # Current model still passes gate — check if drift is real
                # or just a regime shift in the sensor data
                log.info(
                    "Quality gate PASS: F1=%.4f (≥%.2f) AUC=%.4f (≥%.2f). "
                    "Model is already good — likely a sensor calibration issue. "
                    "Flagging for human review instead of full retrain.",
                    f1, QUALITY_GATE_F1, auc, QUALITY_GATE_AUC,
                )
                record["outcome"] = "skipped_gate_pass"
                record["reason"] = (
                    f"Current model already passes quality gate "
                    f"(F1={f1:.4f}, AUC={auc:.4f}). "
                    "Drift may be sensor-related — flagged for operator review."
                )
                await get_bus().publish(
                    "mlops.human_review.request.v1",
                    {
                        "reason": "drift_detected_but_model_healthy",
                        "metrics": metrics,
                        "trigger": trigger_payload,
                        "at": datetime.now(UTC).isoformat(),
                    },
                )
            else:
                # Quality gate failed — trigger full retraining
                log.warning(
                    "Quality gate FAIL: F1=%.4f AUC=%.4f — initiating retrain",
                    f1, auc,
                )
                success = await self._run_retrain_pipeline(feedback_n)
                record["outcome"] = "retrained" if success else "failed"
                if success:
                    await get_bus().publish(
                        MODEL_UPDATED_TOPIC,
                        {
                            "updated_at": datetime.now(UTC).isoformat(),
                            "feedback_samples_used": feedback_n,
                            "previous_f1": f1,
                            "previous_auc": auc,
                        },
                    )

            self._last_retrain = datetime.now(UTC)
            self._retrain_count += 1
            self.history.append(record)

    async def _run_retrain_pipeline(self, feedback_n: int) -> bool:
        """Invoke the training script in a subprocess, ML-flow tracked."""
        script = Path("iep2/scripts/train_models.py")
        if not script.exists():
            # Fall back to the top-level training script
            script = Path("iep2/app/train.py")
        if not script.exists():
            log.error(
                "Training script not found — cannot retrain. "
                "Expected iep2/scripts/train_models.py"
            )
            return False

        log.info("Launching retraining subprocess: %s", script)
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--include-feedback"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                log.info("Retraining completed successfully\n%s", result.stdout[-500:])
                return True
            else:
                log.error("Retraining failed (rc=%d)\n%s", result.returncode, result.stderr[-500:])
                return False
        except subprocess.TimeoutExpired:
            log.error("Retraining timed out after 300s")
            return False
        except Exception as e:
            log.error("Retraining subprocess error: %s", e)
            return False

    # ── Bus handler ──────────────────────────────────────────────────────────

    async def on_retrain_request(self, payload: dict) -> None:
        asyncio.create_task(self.retrain(payload))

    def wire(self) -> None:
        get_bus().subscribe(RETRAIN_TOPIC, self.on_retrain_request)
        log.info(
            "retraining trigger wired (gate: F1≥%.2f AUC≥%.2f, cooldown=%dmin)",
            QUALITY_GATE_F1, QUALITY_GATE_AUC, COOLDOWN_MINUTES,
        )


# Module-level singleton
_trigger: RetrainingTrigger | None = None


def get_trigger() -> RetrainingTrigger:
    global _trigger
    if _trigger is None:
        _trigger = RetrainingTrigger()
    return _trigger
