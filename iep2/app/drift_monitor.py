"""
Input Drift Monitor
====================
Tracks embedding distribution statistics over a sliding window
to detect input drift — a key ML-specific observability signal.

Exposes Prometheus metrics for:
  - Embedding mean/std (per sliding window)
  - Cosine similarity to reference centroid
  - Distribution shift indicator (PSI-like)
"""

import threading

import numpy as np
from prometheus_client import Gauge, Histogram

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
EMBEDDING_MEAN = Gauge(
    "iep2_embedding_mean",
    "Mean of incoming embedding vector (drift signal)",
)
EMBEDDING_STD = Gauge(
    "iep2_embedding_std",
    "Std deviation of incoming embedding vector (drift signal)",
)
EMBEDDING_NORM = Histogram(
    "iep2_embedding_l2_norm",
    "L2 norm of incoming feature vectors (208-d vibration features)",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
)
COSINE_SIM_TO_CENTROID = Histogram(
    "iep2_cosine_similarity_to_centroid",
    "Cosine similarity of input embedding to training centroid",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
CONFIDENCE_BUCKET = Histogram(
    "iep2_prediction_confidence_detailed",
    "Detailed prediction confidence distribution",
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)


class DriftMonitor:
    """
    Tracks embedding distribution statistics for drift detection.

    Maintains a sliding window of recent embeddings and compares
    incoming data to a reference centroid computed at calibration time.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._window: list[np.ndarray] = []
        self._reference_centroid: np.ndarray | None = None
        self._lock = threading.Lock()

    def set_reference_centroid(self, centroid: np.ndarray):
        """Set the reference centroid (from training data or calibration)."""
        with self._lock:
            self._reference_centroid = centroid.astype(np.float32)

    def observe(self, embedding: np.ndarray, confidence: float | None = None):
        """
        Record an embedding observation and update metrics.

        Args:
            embedding: 1024-d float32 embedding vector
            confidence: Optional prediction confidence to track
        """
        emb = embedding.astype(np.float32)

        # L2 norm
        l2_norm = float(np.linalg.norm(emb))
        EMBEDDING_NORM.observe(l2_norm)

        # Embedding stats
        EMBEDDING_MEAN.set(float(np.mean(emb)))
        EMBEDDING_STD.set(float(np.std(emb)))

        # Cosine similarity to reference centroid
        if self._reference_centroid is not None:
            cos_sim = float(
                np.dot(emb, self._reference_centroid)
                / (np.linalg.norm(emb) * np.linalg.norm(self._reference_centroid) + 1e-8)
            )
            COSINE_SIM_TO_CENTROID.observe(cos_sim)

        # Confidence tracking
        if confidence is not None:
            CONFIDENCE_BUCKET.observe(confidence)

        # Update sliding window
        with self._lock:
            self._window.append(emb)
            if len(self._window) > self._window_size:
                self._window.pop(0)


# Singleton instance
drift_monitor = DriftMonitor()
