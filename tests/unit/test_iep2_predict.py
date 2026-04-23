"""Unit tests for IEP2 LeakClassifier predict output contract."""

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pytest

from iep2.app.classifier import LeakClassifier, LABEL_MAP

FEAT_DIM = 39  # must match IEP1 output


def _dummy_features():
    rng = np.random.default_rng(7)
    return rng.standard_normal(FEAT_DIM).astype(np.float32)


@pytest.fixture(scope="module")
def clf():
    c = LeakClassifier()
    if not c._is_loaded:
        pytest.skip("IEP2 models not present — skipping predict tests")
    return c


class TestPredictContract:
    def test_returns_dict(self, clf):
        result = clf.predict(_dummy_features())
        assert isinstance(result, dict)

    def test_keys_exist(self, clf):
        result = clf.predict(_dummy_features())
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_label_is_valid(self, clf):
        result = clf.predict(_dummy_features())
        assert result["label"] in LABEL_MAP.values()

    def test_confidence_in_range(self, clf):
        result = clf.predict(_dummy_features())
        conf = result["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self, clf):
        result = clf.predict(_dummy_features())
        probs = result["probabilities"]

        assert isinstance(probs, dict)
        assert set(probs.keys()) == set(LABEL_MAP.values())

        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4

    def test_pipe_materials(self, clf):
        for mat in ("PVC", "Steel", "Cast_Iron"):
            result = clf.predict(_dummy_features(), pipe_material=mat)
            assert result["label"] in LABEL_MAP.values()


class TestLabelMap:
    def test_label_map_has_two_classes(self):
        assert len(LABEL_MAP) == 2

    def test_label_map_contains_leak_and_no_leak(self):
        values = set(LABEL_MAP.values())
        assert "Leak" in values
        assert "No_Leak" in values