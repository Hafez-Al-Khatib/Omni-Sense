"""
Extract 39-d EEP-compatible features from the synthesized audio clips.

Uses the same feature extractor as the production EEP service
(omni/eep/features.py — pure NumPy, no librosa dependency) to guarantee
training/inference parity.

Output parquet format matches train_models.py expectations:
  embedding_0 … embedding_38   — 39-d DSP feature vector
  label, source_wav, pipe_material, pressure_bar, filename — metadata

Usage
-----
    py -3.12 scripts/extract_eep_features.py \\
        --input-dir data/synthesized \\
        --output    data/synthesized/eep_features.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from omni.eep.features import extract_features  # 39-d, pure NumPy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/synthesized")
    parser.add_argument("--output", default="data/synthesized/eep_features.parquet")
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

    records = []
    skipped = 0

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

        # Pad/trim to exactly 5 s
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
    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    n_feat = len([c for c in out_df.columns if c.startswith("embedding_")])
    print(f"Saved {len(out_df)} rows × {n_feat} embedding dims → {output_path}")


if __name__ == "__main__":
    main()
