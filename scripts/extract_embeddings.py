"""
Omni-Sense YAMNet Embedding Extractor
=======================================
Extracts 1024-dimensional embeddings from augmented audio clips
using the pre-trained YAMNet model.

Two extraction modes:

  LOCAL (default):
    Loads YAMNet from TensorFlow Hub directly in this process.
    Requires a working TensorFlow installation.

      python scripts/extract_embeddings.py \\
          --input-dir data/synthesized \\
          --output data/synthesized/embeddings.parquet

  SERVICE (recommended if TF has DLL/env issues on Windows):
    Delegates to the running IEP1 microservice via HTTP.
    No TensorFlow needed on the host — run `docker-compose up iep1 -d` first.

      python scripts/extract_embeddings.py \\
          --input-dir data/synthesized \\
          --output data/synthesized/embeddings.parquet \\
          --iep1-url http://localhost:8001

Output:
    Parquet file containing all metadata columns from metadata.csv
    plus embedding_0 .. embedding_1023 (float32).
"""

import argparse
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TARGET_SR = 16000


# ─── Local TF Mode ────────────────────────────────────────────────────────────

def _build_local_embedding_fn() -> Callable[[Path], np.ndarray]:
    """
    Load YAMNet from TF Hub and return a callable that extracts
    a mean-pooled 1024-d embedding from a WAV file path.
    """
    import tensorflow as tf
    import tensorflow_hub as hub
    import soundfile as sf

    print("  Loading YAMNet from TensorFlow Hub (local mode)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("  YAMNet loaded.")

    def embed(wav_path: Path) -> np.ndarray:
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        waveform = tf.cast(audio.astype(np.float32), tf.float32)
        _, embeddings, _ = model(waveform)
        return tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)

    return embed


# ─── Service Mode ─────────────────────────────────────────────────────────────

def _build_service_embedding_fn(iep1_url: str) -> Callable[[Path], np.ndarray]:
    """
    Return a callable that POSTs a WAV file to the running IEP1 service
    and returns the 1024-d embedding from the JSON response.

    Requires:  docker-compose up iep1 -d
    Default:   http://localhost:8001
    """
    import httpx

    base = iep1_url.rstrip("/")

    # Verify IEP1 is reachable before starting the loop
    print(f"  Checking IEP1 health at {base}/health ...")
    try:
        resp = httpx.get(f"{base}/health", timeout=10.0)
        resp.raise_for_status()
        print(f"  IEP1 healthy: {resp.json()}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach IEP1 at {base}. "
            "Start it with: docker-compose up iep1 -d\n"
            f"  Error: {e}"
        )

    def embed(wav_path: Path) -> np.ndarray:
        with open(wav_path, "rb") as f:
            resp = httpx.post(
                f"{base}/embed",
                files={"audio": (wav_path.name, f, "audio/wav")},
                timeout=30.0,
            )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)

    return embed


# ─── Processing Loop ──────────────────────────────────────────────────────────

def process_all_clips(
    input_dir: Path,
    metadata_path: Path,
    embedding_fn: Callable[[Path], np.ndarray],
    log_interval: int = 50,
) -> pd.DataFrame:
    """
    Iterate over every row in metadata.csv, call embedding_fn on the
    corresponding WAV file, and return a DataFrame of metadata + embeddings.
    """
    metadata_df = pd.read_csv(metadata_path)
    total = len(metadata_df)
    print(f"  Processing {total} audio clips...")

    embeddings_list: list[np.ndarray] = []
    valid_rows: list = []

    for idx, row in metadata_df.iterrows():
        filepath = input_dir / row["filename"]

        if not filepath.exists():
            print(f"  [SKIP] {row['filename']}: file not found")
            continue

        try:
            embedding = embedding_fn(filepath)
            embeddings_list.append(embedding)
            valid_rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {row['filename']}: {e}")
            continue

        if (idx + 1) % log_interval == 0:
            print(f"  → {idx + 1}/{total} processed...")

    n_features = len(embeddings_list[0]) if embeddings_list else 208
    embedding_cols = [f"embedding_{i}" for i in range(n_features)]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    embeddings_df = pd.DataFrame(embeddings_array, columns=embedding_cols)

    metadata_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    return pd.concat([metadata_valid, embeddings_df], axis=1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract YAMNet embeddings from augmented audio clips. "
            "Use --iep1-url to delegate to the running IEP1 Docker service "
            "instead of loading TensorFlow locally (recommended on Windows)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/synthesized",
        help="Directory containing .wav clips and metadata.csv (from augment_data.py).",
    )
    parser.add_argument(
        "--output",
        default="data/synthesized/embeddings.parquet",
        help="Output path for the embeddings Parquet file.",
    )
    parser.add_argument(
        "--iep1-url",
        default=None,
        help=(
            "IEP1 service URL for service-mode extraction, e.g. http://localhost:8001. "
            "If omitted, YAMNet is loaded locally (requires TensorFlow)."
        ),
    )
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=100,
        help="Log progress every N clips.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata_path = input_dir / "metadata.csv"
    output_path = Path(args.output)

    if not metadata_path.exists():
        print(f"[ERROR] metadata.csv not found at: {metadata_path}")
        print("  Run augment_data.py first:")
        print("    python scripts/augment_data.py --input-dir Processed_audio_16k --output-dir data/synthesized")
        return

    print("=" * 60)
    print("Omni-Sense YAMNet Embedding Extractor")
    mode = "SERVICE" if args.iep1_url else "LOCAL"
    print(f"Mode: {mode}")
    print("=" * 60)

    # Build the embedding callable for the chosen mode
    if args.iep1_url:
        embedding_fn = _build_service_embedding_fn(args.iep1_url)
    else:
        embedding_fn = _build_local_embedding_fn()

    result_df = process_all_clips(
        input_dir, metadata_path, embedding_fn,
        log_interval=args.batch_log_interval,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(str(output_path), index=False)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(result_df)} embeddings saved.")
    print(f"Output: {output_path}")
    print(f"Shape:  {result_df.shape}")
    print(f"Labels: {result_df['label'].value_counts().to_dict()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
