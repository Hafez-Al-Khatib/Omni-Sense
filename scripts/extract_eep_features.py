"""
Extract 39-d EEP-compatible features from the synthesized audio clips.

Uses the same feature extractor as the production EEP service
(omni/eep/features.py — pure NumPy, no librosa dependency) to guarantee
training/inference parity.

Output parquet format matches train_models.py expectations:
  embedding_0 … embedding_38   — 39-d DSP feature vector
  label, source_wav, pipe_material, pressure_bar, filename — metadata

Sample-rate parity with the deployed sensor
-------------------------------------------
The ESP32-S3 + ADXL345 captures at 3200 Hz (max ODR of the part). Models
trained on the 16 kHz LeakDB corpus produce a different feature
distribution than what the device actually sends in production. To close
the gap, pass ``--target-sr 3200`` and the script polyphase-decimates
each clip with a proper anti-alias filter (scipy.signal.resample_poly)
before computing the 39-d vector. Output features are then directly
comparable to what the device computes on-line at 3200 Hz.

Usage
-----
    # 16 kHz baseline (matches training corpus)
    py -3.12 scripts/extract_eep_features.py \\
        --input-dir data/synthesized \\
        --output    data/synthesized/eep_features.parquet

    # 3200 Hz deployment-rate parity
    py -3.12 scripts/extract_eep_features.py \\
        --input-dir data/synthesized \\
        --target-sr 3200 \\
        --output    data/synthesized/eep_features_3200hz.parquet
"""

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))
from omni.eep.features import extract_features  # 39-d, pure NumPy


def _resample_to(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Polyphase resample with anti-alias FIR — equivalent to scipy's recommended
    decimation path. Uses gcd to keep up/down factors small (e.g. 16 000 → 3200
    becomes up=1, down=5; 22 050 → 3200 becomes up=64, down=441).
    """
    if src_sr == dst_sr:
        return pcm
    g = gcd(src_sr, dst_sr)
    up   = dst_sr // g
    down = src_sr // g
    return resample_poly(pcm, up, down).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/synthesized")
    parser.add_argument("--output", default="data/synthesized/eep_features.parquet")
    parser.add_argument(
        "--target-sr", type=int, default=None,
        help="If set, polyphase-resample every clip to this rate before "
             "feature extraction. Use 3200 for ESP32-S3 deployment parity.",
    )
    args = parser.parse_args()

    clips_dir = Path(args.input_dir)
    output_path = Path(args.output)
    meta_path = clips_dir / "metadata.csv"

    if not meta_path.exists():
        print(f"ERROR: metadata.csv not found in {clips_dir}", file=sys.stderr)
        sys.exit(1)

    df_meta = pd.read_csv(meta_path)
    print(f"Clips in metadata: {len(df_meta)}")
    print(df_meta["label"].value_counts().to_string())
    if args.target_sr:
        print(f"Resampling all clips to {args.target_sr} Hz (polyphase + anti-alias).")

    records = []
    skipped = 0
    src_sr_seen = set()

    for idx, row in df_meta.iterrows():
        wav_path = clips_dir / row["filename"]
        if not wav_path.exists():
            skipped += 1
            continue
        try:
            pcm, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"  SKIP {row['filename']}: {exc}")
            skipped += 1
            continue

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        src_sr_seen.add(sr)

        # Resample BEFORE pad/trim so the 5 s window is in the target rate
        if args.target_sr and args.target_sr != sr:
            pcm = _resample_to(pcm, sr, args.target_sr)
            sr = args.target_sr

        # Pad/trim to exactly 5 s at the (possibly new) sample rate
        target = sr * 5
        if len(pcm) < target:
            pcm = np.pad(pcm, (0, target - len(pcm)))
        else:
            pcm = pcm[:target]

        feat = extract_features(pcm, sr=sr)  # (39,) float32

        record = {
            "filename":      row["filename"],
            "label":         row["label"],
            "source_wav":    row.get("source_wav", row["filename"]),
            "pipe_material": str(row.get("pipe_material", "PVC")),
            "pressure_bar":  float(row.get("pressure_bar", 3.0)),
        }
        for i, v in enumerate(feat):
            record[f"embedding_{i}"] = float(v)

        records.append(record)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df_meta)} clips…")

    print(f"Extracted {len(records)} clips, skipped {skipped}.")
    print(f"Source sample rates seen: {sorted(src_sr_seen)} Hz")
    if args.target_sr:
        print(f"All features computed at target rate: {args.target_sr} Hz")
    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    n_feat = len([c for c in out_df.columns if c.startswith("embedding_")])
    print(f"Saved {len(out_df)} rows x {n_feat} embedding dims -> {output_path}")


if __name__ == "__main__":
    main()
