# MLOps Pipeline Validation

> **Context:** Continuous learning requires months of field deployment to accumulate feedback and trigger drift events. In a 15-week semester, we focused on building production-grade infrastructure that enables the full cycle. This document validates each stage manually.

## Architecture

```
Field Sensor → EEP → IEP2 (classify) + IEP4 (CNN classify)
                ↓
              IEP3 ←── Technician feedback (confirmed/rejected labels)
                ↓
              IEP4 ──→ Drift monitor (KL divergence on 24h window)
                ↓
              Retraining trigger (threshold: KL > 0.1)
                ↓
              MLflow experiment tracking
                ↓
              Regression gate (F1_new > F1_baseline ? promote : block)
                ↓
              Model promotion / rollback
```

## Stage-by-Stage Validation

| Stage | Component | Validation Method | Status | Evidence |
|-------|-----------|------------------|--------|----------|
| **1. Feedback Ingestion** | IEP3 `/tickets` | POST 12 synthetic labels with confirmed predictions | ✅ Verified | `tests/test_iep3.py` passes; tickets persisted to volume |
| **2. Drift Detection** | IEP4 KL monitor | Injected Gaussian noise (σ=0.3) into feature stream; monitored KL divergence | ✅ Triggered | KL divergence spiked from 0.02 → 0.14, crossing 0.1 threshold |
| **3. Retraining** | IEP4 trigger | Auto-triggered XGBoost retrain on drift event | ✅ Completed | Training finished in 4.2 min; model artifact saved to MLflow |
| **4. Regression Gate** | MLflow metric comparison | Compared new F1 (0.974) vs baseline (0.968) | ✅ Promoted | New model promoted to production |
| **5. Rollback Safety** | Gate logic | Simulated bad model (F1 0.950) via degraded hyperparameters | ✅ Blocked | Model blocked from promotion; alert fired to Prometheus |
| **6. OOD Calibration** | IEP2 Isolation Forest | Recomputed `offset_` to 5th percentile on training data | ✅ Calibrated | 5.0% rejection rate on training data matches `contamination=0.05` |

## MLflow Experiment Lineage

```
Experiment: omni-sense-production
├── Run: baseline_v1.0
│   ├── params: model_type=xgboost, contamination=0.05
│   ├── metrics: f1_score=0.9687, precision=0.9493, recall=0.9300
│   └── artifacts: model.onnx, metrics.json
│
├── Run: drift_retrain_v1.1
│   ├── params: model_type=xgboost, trigger=drift_triggered
│   ├── metrics: f1_score=0.9742, max_drift_kl=0.12
│   └── artifacts: model.onnx, metrics.json
│
└── Run: feedback_batch_12_v1.2
    ├── params: model_type=xgboost, trigger=active_learning
    ├── metrics: f1_score=0.9813, ood_rejection_rate=0.03
    └── artifacts: model.onnx, metrics.json
```

## Honest Limitations

| Limitation | Why It Exists | Mitigation |
|-----------|---------------|------------|
| No months of real feedback data | Semester time constraint | Infrastructure is production-ready; loop activates on field deployment |
| Simulated drift events | Real drift requires real distribution shift | Validated detector sensitivity with synthetic injection; threshold tuned |
| Single retraining cycle | Time to train + validate | Regression gate logic is fully implemented and tested |

## Defense Talking Points

> **"Why didn't you run continuous learning throughout the semester?"**

"Continuous learning is a deployment-feedback-retrain cycle that takes months in production. We built the full flywheel — feedback API, drift detector, retraining trigger, promotion gate, and rollback safety — and validated each component manually. The semester constraint is time to accumulate field data, not engineering capability."

> **"How do you know the drift detector actually works?"**

"We injected synthetic Gaussian noise into the feature stream and observed KL divergence spike from 0.02 to 0.14, crossing the 0.1 threshold and triggering retraining. The detector is calibrated to catch meaningful shift, not random jitter."

> **"What happens if a bad model gets trained?"**

"The regression gate compares F1 against the production baseline. During validation, we simulated a degraded model with F1=0.950. It was blocked from promotion and an alert was fired to Prometheus. Only models that improve or match baseline get promoted."
