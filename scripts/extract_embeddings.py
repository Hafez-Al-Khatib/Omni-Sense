"""
Omni-Sense YAMNet Embedding Extractor
=======================================
Extracts 1024-dimensional embeddings from synthesized audio clips
using the pre-trained YAMNet model from TensorFlow Hub.

Usage:
    python scripts/extract_embeddings.py \
        --input-dir data/synthesized \
        --output data/synthesized/embeddings.parquet

Output:
    Parquet file with columns:
        - filename (str)
        - label (str): "leak" or "background"
        - snr_db (float, nullable)
        - pipe_material (str)
        - pressure_bar (float)
        - embedding_0..embedding_1023 (float32)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TARGET_SR = 16000


def load_yamnet():
    """Load YAMNet model from TensorFlow Hub."""
    import tensorflow_hub as hub
    print("  Loading YAMNet from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("  YAMNet loaded successfully.")
    return model


def extract_embedding(model, audio: np.ndarray) -> np.ndarray:
    """
    Extract the mean-pooled 1024-d embedding from a YAMNet model.

    YAMNet outputs:
        scores: (N_frames, 521) — class probabilities per frame
        embeddings: (N_frames, 1024) — per-frame embeddings
        log_mel_spectrogram: the spectrogram

    We mean-pool the per-frame embeddings into a single 1024-d vector.
    """
    import tensorflow as tf

    waveform = tf.cast(audio, tf.float32)
    scores, embeddings, log_mel = model(waveform)

    # Mean-pool across time frames → single 1024-d vector
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return mean_embedding


def process_all_clips(
    input_dir: Path,
    metadata_path: Path,
    model,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Process all audio clips and extract embeddings."""
    import soundfile as sf

    metadata_df = pd.read_csv(metadata_path)
    total = len(metadata_df)

    print(f"  Processing {total} audio clips...")

    embeddings_list = []
    valid_rows = []

    for idx, row in metadata_df.iterrows():
        filepath = input_dir / row["filename"]

        if not filepath.exists():
            print(f"  [SKIP] {row['filename']}: file not found")
            continue

        try:
            audio, sr = sf.read(str(filepath))
            if sr != TARGET_SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            audio = audio.astype(np.float32)
            embedding = extract_embedding(model, audio)
            embeddings_list.append(embedding)
            valid_rows.append(row)

        except Exception as e:
            print(f"  [SKIP] {row['filename']}: {e}")
            continue

        if (idx + 1) % batch_size == 0:
            print(f"  → {idx + 1}/{total} processed...")

    # Build DataFrame
    embedding_cols = [f"embedding_{i}" for i in range(1024)]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_cols)

    metadata_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    result_df = pd.concat([metadata_valid, embeddings_df], axis=1)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract YAMNet embeddings from synthesized audio clips."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/synthesized",
        help="Directory containing synthesized .wav files and metadata.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthesized/embeddings.parquet",
        help="Output path for the embeddings Parquet file.",
    )
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=50,
        help="Log progress every N clips.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.csv"
    output_path = Path(args.output)

    if not metadata_path.exists():
        print(f"[ERROR] metadata.csv not found at: {metadata_path}")
        print("  Run synthesize_data.py first.")
        return

    print("=" * 60)
    print("Omni-Sense YAMNet Embedding Extractor")
    print("=" * 60)

    model = load_yamnet()

    result_df = process_all_clips(
        input_dir, metadata_path, model,
        batch_size=args.batch_log_interval,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(str(output_path), index=False)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(result_df)} embeddings extracted.")
    print(f"Output: {output_path}")
    print(f"Shape: {result_df.shape}")
    print(f"Labels: {result_df['label'].value_counts().to_dict()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
