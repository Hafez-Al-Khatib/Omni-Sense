"""
Model Regression Test Suite (Golden Dataset Gate)
===================================================
Ensures that model updates don't degrade performance
below the production baseline.

Run:
    cd iep2 && pytest tests/test_model_regression.py -v

Environment:
    GOLDEN_DATASET_PATH: Path to golden_dataset_v1.csv
"""

import json
import os
from pathlib import Path

import pytest

# ─── Configuration ──────────────────────────────────────────────────────────

GOLDEN_DATASET_PATH = os.environ.get(
    "GOLDEN_DATASET_PATH",
    "../data/golden/golden_dataset_v1.csv",
)

METRICS_PATH = Path(__file__).parent.parent / "models" / "metrics.json"

# ── Performance Baselines ──
MIN_F1_SCORE = 0.92
MIN_ACCURACY = 0.90
MAX_FALSE_POSITIVE_RATE = 0.10


class TestModelRegression:
    """
    Regression tests using the golden dataset.
    These tests are triggered on PRs that modify IEP2 model logic.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load metrics from the latest training run."""
        if not METRICS_PATH.exists():
            pytest.skip(
                f"metrics.json not found at {METRICS_PATH}. "
                "Train models first with scripts/train_models.py."
            )

        with open(METRICS_PATH) as f:
            self.metrics = json.load(f)

    def test_f1_score_above_baseline(self):
        """F1 score must meet or exceed the production baseline."""
        f1 = self.metrics.get("f1", 0)
        assert f1 >= MIN_F1_SCORE, (
            f"F1 score {f1:.4f} is below the baseline {MIN_F1_SCORE}. "
            f"Model regression detected! Review recent changes."
        )

    def test_accuracy_above_baseline(self):
        """Accuracy must meet the minimum threshold."""
        acc = self.metrics.get("accuracy", 0)
        assert acc >= MIN_ACCURACY, (
            f"Accuracy {acc:.4f} is below the baseline {MIN_ACCURACY}. "
            f"Model regression detected!"
        )

    def test_roc_auc_reasonable(self):
        """ROC AUC should be significantly above random (0.5)."""
        auc = self.metrics.get("roc_auc", 0)
        assert auc >= 0.85, (
            f"ROC AUC {auc:.4f} is too low. "
            f"Model may not be discriminating well."
        )

    def test_precision_recall_balance(self):
        """Neither precision nor recall should be drastically low."""
        precision = self.metrics.get("precision", 0)
        recall = self.metrics.get("recall", 0)

        assert precision >= 0.80, f"Precision {precision:.4f} is too low."
        assert recall >= 0.80, f"Recall {recall:.4f} is too low."

        # Check that they're roughly balanced (not a trivial classifier)
        balance = abs(precision - recall)
        assert balance < 0.20, (
            f"Precision ({precision:.4f}) and recall ({recall:.4f}) are too "
            f"imbalanced (diff={balance:.4f}). The model may be biased."
        )

    def test_no_metric_degradation_from_previous(self):
        """
        If a previous metrics file exists, ensure no key metric
        has degraded by more than 2 percentage points.
        """
        prev_path = METRICS_PATH.parent / "metrics_previous.json"
        if not prev_path.exists():
            pytest.skip("No previous metrics to compare against.")

        with open(prev_path) as f:
            prev_metrics = json.load(f)

        for key in ["f1", "accuracy", "roc_auc"]:
            current = self.metrics.get(key, 0)
            previous = prev_metrics.get(key, 0)
            degradation = previous - current

            assert degradation < 0.02, (
                f"{key} degraded from {previous:.4f} to {current:.4f} "
                f"(delta={degradation:.4f}). This exceeds the 2pp tolerance."
            )
