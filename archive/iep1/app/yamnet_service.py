"""
YAMNet Model Service
=====================
Encapsulates YAMNet model loading and inference.
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class YAMNetService:
    """Singleton service wrapping YAMNet from TensorFlow Hub."""

    YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

    def __init__(self):
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self):
        """Load YAMNet model from TF Hub (call once at startup)."""
        if self._is_loaded:
            return
        self._model = hub.load(self.YAMNET_URL)
        self._is_loaded = True

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract mean-pooled 1024-d embedding from a waveform.

        Args:
            waveform: 1D float32 array at 16kHz

        Returns:
            1024-d float32 embedding vector
        """
        if not self._is_loaded:
            raise RuntimeError("YAMNet model not loaded. Call load() first.")

        waveform_tf = tf.cast(waveform, tf.float32)
        scores, embeddings, log_mel = self._model(waveform_tf)

        # Mean-pool across all time frames → single 1024-d vector
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return mean_embedding.astype(np.float32)

    def extract_frame_embeddings(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract per-frame embeddings (useful for advanced analysis).

        Returns:
            (N_frames, 1024) float32 array
        """
        if not self._is_loaded:
            raise RuntimeError("YAMNet model not loaded. Call load() first.")

        waveform_tf = tf.cast(waveform, tf.float32)
        scores, embeddings, log_mel = self._model(waveform_tf)

        return embeddings.numpy().astype(np.float32)
