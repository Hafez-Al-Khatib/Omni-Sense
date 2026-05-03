#!/usr/bin/env python3
"""Resample all training clips from 16 kHz -> 3.2 kHz using FFT-based sinc interpolation."""
import shutil
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy import signal

SRC_DIR = Path('data/synthesized')
DST_DIR = Path('data/synthesized_3200')
TARGET_SR = 3200

DST_DIR.mkdir(exist_ok=True)
wavs = sorted(SRC_DIR.glob('*.wav'))
print(f"Resampling {len(wavs)} files: 16 kHz -> {TARGET_SR} Hz")

done = 0
for p in wavs:
    try:
        pcm, sr = sf.read(str(p), dtype='float32')
        if pcm.ndim > 1:
            pcm = np.mean(pcm, axis=1)
        n_target = int(len(pcm) * TARGET_SR / sr)
        pcm_3200 = signal.resample(pcm, n_target)
        sf.write(str(DST_DIR / p.name), pcm_3200, TARGET_SR, subtype='PCM_16')
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(wavs)} done")
    except Exception as e:
        print(f"  FAIL {p.name}: {e}")

shutil.copy(SRC_DIR / 'metadata.csv', DST_DIR / 'metadata.csv')
print(f"Done. Output: {DST_DIR} ({done} files)")
