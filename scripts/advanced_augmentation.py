"""
New Augmentation Strategy.
Perform Amplitude scaling, AWGN for electronic noise floor simulation, speed perturbation via time-stretching, and sensor position interpolation.
Should generate much richer samples.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def apply_butterworth_lpf(signal: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter to simulate pipe-material frequency response."""
    nyq = sr / 2.0
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    sos = butter(order, normal_cutoff, btype="low", output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def apply_physics_augmentation(audio_path, output_dir):
    """
    Takes a clean 16kHz WAV file and generates physically-plausible variants.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    file_stem = Path(audio_path).stem

    sf.write(output_dir / f"{file_stem}_base.wav", y, sr)

    y_amp_high = np.clip(y * 1.2, -1.0, 1.0)
    y_amp_low = y * 0.8
    sf.write(output_dir / f"{file_stem}_amp_high.wav", y_amp_high, sr)
    sf.write(output_dir / f"{file_stem}_amp_low.wav", y_amp_low, sr)

    # AWGN
    noise_floor = np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
    y_awgn = np.clip(y + noise_floor, -1.0, 1.0)
    sf.write(output_dir / f"{file_stem}_awgn.wav", y_awgn, sr)

    # Time stretching
    y_stretch_fast = librosa.effects.time_stretch(y, rate=1.08)
    y_stretch_slow = librosa.effects.time_stretch(y, rate=0.92)
    sf.write(output_dir / f"{file_stem}_stretch_fast.wav", y_stretch_fast, sr)
    sf.write(output_dir / f"{file_stem}_stretch_slow.wav", y_stretch_slow, sr)

    # Sensor position interpolation
    y_interp_mid = (y[:-1] + y[1:]) / 2
    sf.write(output_dir / f"{file_stem}_interp_mid.wav", y_interp_mid, sr)

    # Pipe material simulation
    y_pvc = apply_butterworth_lpf(y, 2000, sr)
    y_steel = apply_butterworth_lpf(y, 6000, sr)
    y_cast_iron = apply_butterworth_lpf(y, 3500, sr)
    sf.write(output_dir / f"{file_stem}_pvc.wav", y_pvc, sr)
    sf.write(output_dir / f"{file_stem}_steel.wav", y_steel, sr)
    sf.write(output_dir / f"{file_stem}_cast_iron.wav", y_cast_iron, sr)

    return [f"{file_stem}_base.wav", f"{file_stem}_amp_high.wav", f"{file_stem}_amp_low.wav", f"{file_stem}_awgn.wav", f"{file_stem}_stretch_fast.wav", f"{file_stem}_stretch_slow.wav", f"{file_stem}_interp_mid.wav", f"{file_stem}_pvc.wav", f"{file_stem}_steel.wav", f"{file_stem}_cast_iron.wav"]

if __name__ == "__main__":
    input_dir = Path("./Processed_audio_16k")
    output_dir = Path("./Processed_audio_16k_Augmented")
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in input_dir.glob("*.wav"):
        apply_physics_augmentation(file, output_dir)
