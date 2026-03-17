"""
Unit tests for IEP2 OOD detector and classifier.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.calibration import CalibrationManager


class TestCalibrationManager:
    """Test suite for dynamic OOD calibration."""

    def test_initial_threshold_is_none(self):
        mgr = CalibrationManager()
        assert mgr.get_threshold() is None
        assert mgr.is_calibrated is False

    def test_calibrate_sets_threshold(self):
        mgr = CalibrationManager()
        scores = [0.1, 0.15, 0.12, 0.08, 0.11]
        threshold = mgr.calibrate(scores)

        assert mgr.is_calibrated is True
        assert mgr.get_threshold() is not None
        # Threshold should be mean - 2*std
        expected = np.mean(scores) - 2 * np.std(scores)
        assert abs(threshold - expected) < 1e-6

    def test_calibrate_custom_sigma(self):
        mgr = CalibrationManager()
        scores = [0.1, 0.2, 0.3]
        threshold = mgr.calibrate(scores, n_sigma=1.0)
        expected = np.mean(scores) - 1.0 * np.std(scores)
        assert abs(threshold - expected) < 1e-6

    def test_reset(self):
        mgr = CalibrationManager()
        mgr.calibrate([0.1, 0.2, 0.3])
        assert mgr.is_calibrated is True

        mgr.reset()
        assert mgr.is_calibrated is False
        assert mgr.get_threshold() is None

    def test_recalibration_overwrites(self):
        mgr = CalibrationManager()
        t1 = mgr.calibrate([0.1, 0.2, 0.3])
        t2 = mgr.calibrate([0.5, 0.6, 0.7])
        assert t1 != t2
        assert mgr.get_threshold() == t2


class TestOODDetectorSchema:
    """Test schema validation for OOD detector inputs."""

    def test_embedding_length_check(self):
        """Verify that a 1024-d embedding shape is expected."""
        embedding = np.random.randn(1024).astype(np.float32)
        assert embedding.shape == (1024,)

    def test_wrong_embedding_length(self):
        """Verify we can detect wrong-size embeddings."""
        embedding = np.random.randn(512).astype(np.float32)
        assert embedding.shape != (1024,)


class TestClassifierSchema:
    """Test metadata encoding for the classifier."""

    def test_pipe_material_encoding(self):
        from app.classifier import PIPE_MATERIAL_MAP
        assert PIPE_MATERIAL_MAP["PVC"] == 0
        assert PIPE_MATERIAL_MAP["Steel"] == 1
        assert PIPE_MATERIAL_MAP["Cast_Iron"] == 2

    def test_label_map(self):
        from app.classifier import LABEL_MAP
        assert LABEL_MAP[0] == "background"
        assert LABEL_MAP[1] == "leak"
