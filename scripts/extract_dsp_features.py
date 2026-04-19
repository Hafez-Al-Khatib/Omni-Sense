"""
Omni-Sense DSP Feature Extractor

Replaces YAMNet. Extracts explicit physical and statistical
signal processing features from augmented audio clips.

Features extracted (approx 48 dimensions):
  - Time-Domain: RMS, Zero-Crossing Rate, Kurtosis, Skewness, Crest Factor
  - Frequency-Domain: Spectral Centroid, Bandwidth, Rolloff, Flatness
  - Envelope: MFCCs (Mean and Std Dev)
"""

import argparse
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore", category=FutureWarning)

def extract_dsp_features(filepath: Path) -> np.ndarray:
    """Extracts explicit physical characteristics from the vibration signal."""
    # 1. Load Audio
    y, sr = sf.read(str(filepath))
    if y.ndim > 1:
        y = y.mean(axis=1)

    # 2. Time-Domain Statistical Physics
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Structural anomalies (cracks) cause "spiky" vibrations.
    # Kurtosis and Crest Factor perfectly capture this physical impulsiveness.
    kurt = kurtosis(y)
    skw = skew(y)
    crest_factor = np.max(np.abs(y)) / (np.mean(rms) + 1e-8)

    # 3. Frequency-Domain Physics
    # Centroid: The "center of mass" of the frequencies (distinguishes hiss from rumble)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # Rolloff: The frequency below which 85% of the spectral energy lies
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    # Flatness: How "noise-like" vs "tone-like" the vibration is
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # 4. Envelopes (MFCCs)
    # We use 13 MFCCs to capture the broad shape of the spectrum
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 5. Aggregate everything into a 1D Feature Vector
    features = [
        np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
        kurt, skw, crest_factor,
        np.mean(cent), np.std(cent),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness)
    ]

    # Add Mean and Std of each MFCC
    for i in range(mfccs.shape[0]):
        features.append(np.mean(mfccs[i]))
        features.append(np.std(mfccs[i]))

    return np.array(features, dtype=np.float32)

def process_all_clips(input_dir: Path, metadata_path: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_path)
    total = len(metadata_df)
    print(f"  Extracting DSP Physics features from {total} clips...")

    features_list = []
    valid_rows = []

    for idx, row in metadata_df.iterrows():
        filepath = input_dir / row["filename"]
        if not filepath.exists():
            continue

        try:
            feat_vector = extract_dsp_features(filepath)
            features_list.append(feat_vector)
            valid_rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {row['filename']}: {e}")
            continue

        if (idx + 1) % 100 == 0:
            print(f"  → {idx + 1}/{total} processed...")

    n_features = len(features_list[0])
    feature_cols = [f"embedding_{i}" for i in range(n_features)]

    features_df = pd.DataFrame(np.array(features_list), columns=feature_cols)
    metadata_valid = pd.DataFrame(valid_rows).reset_index(drop=True)

    return pd.concat([metadata_valid, features_df], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/synthesized")
    parser.add_argument("--output", default="data/synthesized/embeddings.parquet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.csv"

    result_df = process_all_clips(input_dir, metadata_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(str(output_path), index=False)

    print(f"Done! Extracted {result_df.shape[1] - len(metadata_path.read_text().split(',')[0])} DSP features per clip.")

if __name__ == "__main__":
    main()
