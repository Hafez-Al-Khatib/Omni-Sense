#!/usr/bin/env python3
"""
SHAP Explainability Module for IEP2 XGBoost Classifier

Generates SHAP-based explanations and feature importance analysis for the trained
XGBoost model used in the IEP2 (Intermittent Equipment Prognostics) service.

Usage:
    python scripts/explain_iep2.py [--model-dir iep2/models] [--embeddings data/synthesized/embeddings.parquet]

Outputs:
    - iep2/explainability/shap_summary.png: SHAP summary (beeswarm) plot
    - iep2/explainability/shap_importance.png: SHAP feature importance bar plot
    - iep2/explainability/feature_importance.json: JSON file with rankings
"""

import argparse
import json
import logging
import os
import sys

# Set matplotlib backend before importing matplotlib
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_feature_names(n_features: int) -> list:
    """
    Generate feature names for the embeddings.

    Maps the first (n_features - 2) features to physics-based names from a lookup table,
    with pipe_material and pressure_bar always as the last 2 features.

    Args:
        n_features: Total number of features in the dataset

    Returns:
        List of feature names
    """
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


def load_data(model_dir: str, embeddings_path: str):
    """
    Load the XGBoost model, label map, and embeddings.

    Args:
        model_dir: Path to directory containing model and label_map.json
        embeddings_path: Path to embeddings.parquet file

    Returns:
        Tuple of (model, X_data, feature_names, label_map) or (None, None, None, None) if files missing
    """
    try:
        import joblib
    except ImportError:
        logger.error("joblib not installed. Please install it with: pip install joblib")
        return None, None, None, None

    model_path = os.path.join(model_dir, 'xgboost_classifier.joblib')
    label_map_path = os.path.join(model_dir, 'label_map.json')

    # Check if model files exist
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None, None, None, None

    if not os.path.exists(label_map_path):
        logger.warning(f"Label map not found: {label_map_path}")
        return None, None, None, None

    if not os.path.exists(embeddings_path):
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return None, None, None, None

    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None, None

    # Load label map
    try:
        with open(label_map_path) as f:
            label_map = json.load(f)
        logger.info(f"Loaded label map with {len(label_map)} classes")
    except Exception as e:
        logger.error(f"Failed to load label map: {e}")
        return None, None, None, None

    # Load embeddings
    try:
        import pandas as pd
        df = pd.read_parquet(embeddings_path)
        logger.info(f"Loaded embeddings with shape {df.shape}")

        # Extract features (all columns except metadata)
        # Assume last 2 columns are pipe_material and pressure_bar
        X_data = df.iloc[:, :-2].values
        feature_names = generate_feature_names(X_data.shape[1])

        logger.info(f"Using {len(feature_names)} features for explanation")
        return model, X_data, feature_names, label_map
    except ImportError:
        logger.error("pandas not installed. Please install it with: pip install pandas")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return None, None, None, None


def generate_shap_explanations(model, X_data: np.ndarray, feature_names: list, output_dir: str):
    """
    Generate SHAP explanations for the model.

    Args:
        model: Trained XGBoost model
        X_data: Feature matrix
        feature_names: List of feature names
        output_dir: Directory to save outputs
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Please install it with: pip install shap")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create SHAP explainer for tree models
        logger.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for all data
        logger.info("Calculating SHAP values (this may take a moment)...")
        shap_values = explainer.shap_values(X_data)

        # Handle multi-class output (list of arrays per class)
        if isinstance(shap_values, list):
            # For multi-class, use the first class for summary
            shap_values_main = shap_values[0]
            logger.info(f"Multi-class model detected ({len(shap_values)} classes)")
        else:
            shap_values_main = shap_values
            logger.info("Binary classification model detected")

        # Generate summary plot (beeswarm with top-20 features)
        logger.info("Generating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_main, X_data, feature_names=feature_names,
                         plot_type="beeswarm", show=False, max_display=20)
        summary_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(summary_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary plot to {summary_path}")

        # Generate bar plot (feature importance)
        logger.info("Generating SHAP importance bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_main, X_data, feature_names=feature_names,
                         plot_type="bar", show=False, max_display=20)
        importance_path = os.path.join(output_dir, 'shap_importance.png')
        plt.savefig(importance_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved importance plot to {importance_path}")

        # Calculate feature importance (mean absolute SHAP values)
        mean_abs_shap = np.abs(shap_values_main).mean(axis=0)
        importance_ranking = np.argsort(-mean_abs_shap)

        # Create JSON output with rankings
        importance_data = {
            "total_features": len(feature_names),
            "top_20_features": [
                {
                    "rank": idx + 1,
                    "feature_name": feature_names[importance_ranking[idx]],
                    "mean_abs_shap_value": float(mean_abs_shap[importance_ranking[idx]]),
                    "feature_index": int(importance_ranking[idx])
                }
                for idx in range(min(20, len(importance_ranking)))
            ],
            "all_features": [
                {
                    "feature_name": feature_names[i],
                    "mean_abs_shap_value": float(mean_abs_shap[i]),
                    "feature_index": i
                }
                for i in importance_ranking
            ]
        }

        json_path = os.path.join(output_dir, 'feature_importance.json')
        with open(json_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        logger.info(f"Saved feature importance rankings to {json_path}")

        logger.info("SHAP explanations generated successfully!")
        return True

    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate SHAP explanations for IEP2 XGBoost classifier'
    )
    parser.add_argument(
        '--model-dir',
        default='iep2/models',
        help='Directory containing xgboost_classifier.joblib and label_map.json'
    )
    parser.add_argument(
        '--embeddings',
        default='data/synthesized/embeddings.parquet',
        help='Path to embeddings.parquet file'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SHAP Explainability Module for IEP2")
    logger.info("=" * 60)

    # Load data
    model, X_data, feature_names, label_map = load_data(args.model_dir, args.embeddings)

    if model is None:
        logger.warning("Could not load model data. Exiting gracefully.")
        sys.exit(0)

    # Generate SHAP explanations
    output_dir = os.path.join(os.path.dirname(args.model_dir), 'explainability')
    success = generate_shap_explanations(model, X_data, feature_names, output_dir)

    if success:
        logger.info(f"All outputs saved to {output_dir}")
        sys.exit(0)
    else:
        logger.error("Failed to generate SHAP explanations")
        sys.exit(1)


if __name__ == '__main__':
    main()
