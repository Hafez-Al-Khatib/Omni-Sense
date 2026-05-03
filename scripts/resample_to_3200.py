"""Resample all WAV files in data/synthesized from 16 kHz to 3.2 kHz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

SRC_DIR = Path("data/synthesized")
DST_DIR = Path("data/synthesized_3200")
SRC_SR = 16_000
DST_SR = 3_200
DURATION_SEC = 5

wav_files = sorted(SRC_DIR.glob("*.wav"))
total = len(wav_files)
print(f"Found {total} WAV files to resample")

for i, wav_path in enumerate(wav_files, 1):
    pcm, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)

    # Ensure exactly 5 seconds at source rate
    target_src = SRC_SR * DURATION_SEC
    if len(pcm) < target_src:
        pcm = np.pad(pcm, (0, target_src - len(pcm)))
    else:
        pcm = pcm[:target_src]

    # Resample to 3.2 kHz
    target_dst = DST_SR * DURATION_SEC
    pcm_3200 = resample(pcm, target_dst).astype(np.float32)

    out_path = DST_DIR / wav_path.name
    sf.write(str(out_path), pcm_3200, DST_SR, subtype="PCM_16")

    if i % 200 == 0 or i == total:
        print(f"  Resampled {i}/{total} files...")

print("Done resampling.")

# Copy metadata and add sample_rate column
import pandas as pd

meta_src = SRC_DIR / "metadata.csv"
meta_dst = DST_DIR / "metadata.csv"
df = pd.read_csv(meta_src)
df["sample_rate"] = DST_SR
df.to_csv(meta_dst, index=False)
print(f"Copied metadata.csv with sample_rate={DST_SR} column -> {meta_dst}")
print(f"Total rows: {len(df)}")
