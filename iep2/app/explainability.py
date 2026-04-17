"""
SHAP Explainability Module for IEP2 FastAPI Service

Provides per-prediction SHAP waterfall explanations using lazy-loaded TreeExplainer.

Usage:
    from iep2.app.explainability import ShapExplainer

    explainer = ShapExplainer(model_path='iep2/models/xgboost_classifier.joblib')
    explanation = explainer.explain(features)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    Lazy-loading SHAP explainer for XGBoost models.

    Provides per-prediction SHAP waterfall explanations with graceful
    degradation if SHAP is not installed.
    """

    def __init__(self, model_path: str = 'iep2/models/xgboost_classifier.joblib',
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the SHAP explainer with lazy loading.

        Args:
            model_path: Path to the saved XGBoost model (joblib format)
            feature_names: Optional list of feature names. If not provided, will use generic names.
        """
        self.model_path = model_path
        self.feature_names = feature_names
        self._explainer = None
        self._model = None
        self._shap_available = self._check_shap_availability()

    def _check_shap_availability(self) -> bool:
        """Check if SHAP package is installed."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP package not installed. Per-prediction explanations will be unavailable.")
            logger.info("Install SHAP with: pip install shap")
            return False

    def _load_model(self):
        """Load the XGBoost model from disk (lazy loading)."""
        if self._model is not None:
            return

        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return

        try:
            import joblib
            self._model = joblib.load(self.model_path)
            logger.info(f"Loaded XGBoost model from {self.model_path}")
        except ImportError:
            logger.error("joblib not installed. Cannot load model.")
            return
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

    def _create_explainer(self):
        """Create SHAP TreeExplainer (lazy initialization)."""
        if self._explainer is not None:
            return

        if not self._shap_available:
            return

        self._load_model()

        if self._model is None:
            return

        try:
            import shap
            self._explainer = shap.TreeExplainer(self._model)
            logger.info("Initialized SHAP TreeExplainer")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._shap_available = False

    def _generate_feature_names(self, n_features: int) -> List[str]:
        """
        Generate feature names if not provided.

        Maps the first (n_features - 2) features to physics-based names from a lookup table,
        with pipe_material and pressure_bar always as the last 2 features.

        Args:
            n_features: Total number of features

        Returns:
            List of feature names
        """
        if self.feature_names and len(self.feature_names) >= n_features:
            return self.feature_names[:n_features]

        # Metadata features (always last 2)
        metadata_features = ['pipe_material', 'pressure_bar']

        # If we have fewer than 2 features, return what we can
        if n_features <= 0:
            return []
        if n_features == 1:
            return ['pipe_material']

        # Reserve last 2 slots for metadata
        num_embedding_features = n_features - 2

        # Physics-based feature names for the first 100 features
        physics_features = [
            # Time domain statistics
            'rms', 'kurtosis', 'crest_factor', 'skewness', 'shape_factor',
            'impulse_factor', 'variance', 'peak_to_peak', 'zcr', 'log_energy',
            'diff_rms', 'diff_kurtosis',

            # Envelope analysis
            'env_rms', 'env_mean', 'env_kurtosis', 'env_crest_factor',
            'env_skewness', 'env_peak', 'env_entropy', 'env_zcr',

            # Octave band energy and kurtosis (8 bands: 0-7)
        ] + [f'octave_energy_{i}' for i in range(8)] + [f'octave_kurtosis_{i}' for i in range(8)]

        # Spectral features
        physics_features.extend([
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'spectral_entropy', 'spectral_flatness', 'dominant_freq',
            'spectral_kurtosis',
        ])

        # Autocorrelation features
        physics_features.extend([
            'autocorr_lag10', 'autocorr_lag50', 'autocorr_lag100',
            'autocorr_lag200', 'autocorr_lag500', 'autocorr_lag1000',
            'autocorr_lag2000', 'autocorr_lag5000',
        ])

        # Teager-Kaiser features
        physics_features.extend([
            'tk_rms', 'tk_mean', 'tk_kurtosis', 'tk_crest',
        ])

        # Frame statistics (6 frames × 3 stats = 18 features)
        for frame_idx in range(6):
            for stat in ['rms', 'kurtosis', 'crest']:
                physics_features.append(f'frame_{stat}_{frame_idx}')

        # Subband modulation
        physics_features.extend([f'sub_band_mod_{i}' for i in range(7)])

        # Bandpass filters
        for bpf_idx in range(4):
            for stat in ['rms', 'kurtosis', 'crest', 'skewness']:
                physics_features.append(f'bandpass_{stat}_{bpf_idx}')

        # Build feature names list
        feature_names = []

        # Add physics features (up to 96 available)
        physics_count = min(num_embedding_features, len(physics_features))
        feature_names.extend(physics_features[:physics_count])

        # If we need more features than available physics, fill with embedding names
        if num_embedding_features > len(physics_features):
            embedding_count = num_embedding_features - len(physics_features)
            feature_names.extend([f'embedding_{i}' for i in range(len(physics_features), len(physics_features) + embedding_count)])

        # Add metadata features at the end (always last 2)
        feature_names.extend(metadata_features)

        return feature_names[:n_features]

    def explain(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP waterfall explanation for a single prediction.

        Args:
            features: Input feature vector (1D array of shape (n_features,))

        Returns:
            JSON-serializable dictionary with top-10 contributing features and their SHAP values.
            Returns empty explanation with message if SHAP is not available.

        Example return format:
            {
                "available": True,
                "top_features": [
                    {
                        "name": "kurtosis",
                        "shap_value": 0.342,
                        "direction": "increases_leak_probability"
                    },
                    ...
                ]
            }
        """
        if not self._shap_available:
            return {
                "available": False,
                "message": "SHAP package not installed",
                "top_features": []
            }

        self._create_explainer()

        if self._explainer is None:
            return {
                "available": False,
                "message": "Failed to initialize SHAP explainer",
                "top_features": []
            }

        try:
            # Ensure features is 2D for prediction
            if features.ndim == 1:
                features_2d = features.reshape(1, -1)
            else:
                features_2d = features

            # Calculate SHAP values for this instance
            shap_values = self._explainer.shap_values(features_2d)

            # Handle multi-class output (list of arrays per class)
            if isinstance(shap_values, list):
                # For multi-class, use the first class
                shap_values_instance = shap_values[0][0]
            else:
                # Binary classification or regression
                shap_values_instance = shap_values[0]

            # Get feature names
            n_features = features.shape[0]
            feature_names = self._generate_feature_names(n_features)

            # Get top-10 features by absolute SHAP value
            abs_shap = np.abs(shap_values_instance)
            top_indices = np.argsort(-abs_shap)[:10]

            # Build explanation
            top_features = []
            for rank, idx in enumerate(top_indices):
                shap_val = float(shap_values_instance[idx])
                direction = "increases_leak_probability" if shap_val > 0 else "decreases_leak_probability"

                top_features.append({
                    "rank": rank + 1,
                    "name": feature_names[idx],
                    "shap_value": round(shap_val, 6),
                    "direction": direction,
                    "abs_shap_value": round(float(abs_shap[idx]), 6)
                })

            return {
                "available": True,
                "top_features": top_features
            }

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {
                "available": True,
                "message": f"Error generating explanation: {str(e)}",
                "top_features": []
            }

    def explain_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for multiple predictions (batch).

        Args:
            features: Input feature matrix (2D array of shape (n_samples, n_features))

        Returns:
            List of explanation dictionaries, one per sample
        """
        if not self._shap_available:
            return [
                {
                    "available": False,
                    "message": "SHAP package not installed",
                    "top_features": []
                }
                for _ in range(features.shape[0])
            ]

        self._create_explainer()

        if self._explainer is None:
            return [
                {
                    "available": False,
                    "message": "Failed to initialize SHAP explainer",
                    "top_features": []
                }
                for _ in range(features.shape[0])
            ]

        try:
            # Calculate SHAP values for all instances
            shap_values = self._explainer.shap_values(features)

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values_main = shap_values[0]
            else:
                shap_values_main = shap_values

            # Get feature names
            n_features = features.shape[1]
            feature_names = self._generate_feature_names(n_features)

            # Generate explanation for each sample
            explanations = []
            for sample_idx in range(features.shape[0]):
                shap_values_instance = shap_values_main[sample_idx]
                abs_shap = np.abs(shap_values_instance)
                top_indices = np.argsort(-abs_shap)[:10]

                top_features = []
                for rank, idx in enumerate(top_indices):
                    shap_val = float(shap_values_instance[idx])
                    direction = "increases_leak_probability" if shap_val > 0 else "decreases_leak_probability"

                    top_features.append({
                        "rank": rank + 1,
                        "name": feature_names[idx],
                        "shap_value": round(shap_val, 6),
                        "direction": direction,
                        "abs_shap_value": round(float(abs_shap[idx]), 6)
                    })

                explanations.append({
                    "available": True,
                    "top_features": top_features
                })

            return explanations

        except Exception as e:
            logger.error(f"Error generating batch SHAP explanations: {e}")
            return [
                {
                    "available": True,
                    "message": f"Error generating explanation: {str(e)}",
                    "top_features": []
                }
                for _ in range(features.shape[0])
            ]
