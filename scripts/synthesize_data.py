"""
Omni-Sense Data Synthesizer
============================
Generates realistic training data by overlaying clean leak signatures
with urban noise at various SNR levels and applying pipe-material filters.

Usage:
    python scripts/synthesize_data.py \
        --leak-dir data/raw/leaks \
        --noise-dir data/raw/urbansound8k/audio \
        --output-dir data/synthesized \
        --samples-per-combo 10

Input expectations:
    - Leak signals: WAV or CSV files from Mendeley Water Leakage Dataset
    - Noise signals: WAV files from UrbanSound8K dataset

Output:
    - data/synthesized/*.wav  (16kHz mono, 5-second clips)
    - data/synthesized/metadata.csv  (label, snr, pipe_material, pressure_bar, source_files)
"""

import argparse
import csv
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_SR = 16000  # YAMNet's expected sample rate
CLIP_DURATION_S = 5.0  # seconds per training clip
CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION_S)  # 80,000 samples

# SNR levels (dB) for mixing leak signal with noise
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]

# Pipe material simulation via Butterworth low-pass filter
PIPE_PROFILES = {
    "PVC": {"cutoff_hz": 2000, "order": 4},     # Dampens high-freq hiss
    "Steel": {"cutoff_hz": 6000, "order": 4},    # Preserves sharp transients
    "Cast_Iron": {"cutoff_hz": 3500, "order": 3},  # Middle ground
}

# Simulated pressure levels (bar) — affects leak amplitude scaling
PRESSURE_LEVELS = [1.5, 2.0, 3.0, 4.5, 6.0]


# ─── Audio Utilities ─────────────────────────────────────────────────────────

def load_audio(filepath: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file, resample to target_sr, and convert to mono."""
    filepath = str(filepath)

    if filepath.endswith(".csv"):
        # Mendeley dataset: CSV with a single column of float samples
        import pandas as pd
        df = pd.read_csv(filepath, header=None)
        audio = df.iloc[:, 0].values.astype(np.float32)
        # Assume 8kHz original rate for Mendeley hydrophone data
        if TARGET_SR != 8000:
            audio = librosa.resample(audio, orig_sr=8000, target_sr=target_sr)
    else:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """RMS-normalize audio to a target level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    return audio * (target_rms / rms)


def extract_random_clip(audio: np.ndarray, clip_samples: int = CLIP_SAMPLES) -> np.ndarray:
    """Extract a random clip of fixed length. Pad with zeros if too short."""
    if len(audio) < clip_samples:
        padded = np.zeros(clip_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded

    start = np.random.randint(0, len(audio) - clip_samples)
    return audio[start:start + clip_samples]


def apply_butterworth_lpf(
    audio: np.ndarray,
    cutoff_hz: int,
    sr: int = TARGET_SR,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter to simulate pipe material dampening."""
    from scipy.signal import butter, sosfilt

    nyquist = sr / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    # Clamp to valid range
    normalized_cutoff = min(normalized_cutoff, 0.99)
    sos = butter(order, normalized_cutoff, btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix signal and noise at a specified SNR (dB)."""
    # Ensure same length
    min_len = min(len(signal), len(noise))
    signal = signal[:min_len]
    noise = noise[:min_len]

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return signal

    # Scale noise to achieve target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    mixed = signal + noise_scaled

    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    return mixed.astype(np.float32)


def apply_pressure_scaling(audio: np.ndarray, pressure_bar: float) -> np.ndarray:
    """
    Simulate the effect of pipe pressure on leak amplitude.
    Higher pressure → louder leak signature.
    Normalized around 3.0 bar as reference.
    """
    scale = np.sqrt(pressure_bar / 3.0)  # sqrt relationship (approximate)
    return (audio * scale).astype(np.float32)


# ─── Data Generation ─────────────────────────────────────────────────────────

def collect_audio_files(directory: str | Path, extensions: tuple = (".wav", ".flac", ".ogg", ".csv")) -> list[Path]:
    """Recursively find all audio files in a directory."""
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def generate_leak_samples(
    leak_files: list[Path],
    noise_files: list[Path],
    output_dir: Path,
    samples_per_combo: int = 10,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate synthesized leak training samples."""
    if rng is None:
        rng = np.random.default_rng(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    sample_idx = 0

    print(f"  Leak files: {len(leak_files)}")
    print(f"  Noise files: {len(noise_files)}")

    for leak_path in leak_files:
        try:
            leak_audio = load_audio(leak_path)
            leak_audio = normalize_audio(leak_audio)
        except Exception as e:
            print(f"  [SKIP] {leak_path.name}: {e}")
            continue

        for pipe_name, pipe_cfg in PIPE_PROFILES.items():
            # Apply pipe material filter to leak signal
            filtered_leak = apply_butterworth_lpf(
                leak_audio,
                cutoff_hz=pipe_cfg["cutoff_hz"],
                order=pipe_cfg["order"],
            )

            for snr_db in SNR_LEVELS:
                for _ in range(samples_per_combo):
                    # Pick a random noise file and clip
                    noise_path = rng.choice(noise_files)
                    try:
                        noise_audio = load_audio(noise_path)
                    except Exception:
                        continue

                    # Random pressure
                    pressure = float(rng.choice(PRESSURE_LEVELS))

                    # Apply pressure scaling
                    leak_clip = extract_random_clip(filtered_leak)
                    leak_clip = apply_pressure_scaling(leak_clip, pressure)

                    # Extract noise clip and mix
                    noise_clip = extract_random_clip(noise_audio)
                    mixed = mix_at_snr(leak_clip, noise_clip, snr_db)

                    # Save
                    filename = f"leak_{sample_idx:06d}.wav"
                    filepath = output_dir / filename
                    sf.write(str(filepath), mixed, TARGET_SR)

                    metadata_rows.append({
                        "filename": filename,
                        "label": "leak",
                        "snr_db": snr_db,
                        "pipe_material": pipe_name,
                        "pressure_bar": pressure,
                        "source_leak": leak_path.name,
                        "source_noise": noise_path.name,
                    })
                    sample_idx += 1

    return metadata_rows


def generate_background_samples(
    noise_files: list[Path],
    output_dir: Path,
    num_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate background-only (no leak) training samples."""
    if rng is None:
        rng = np.random.default_rng(42)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    print(f"  Generating {num_samples} background samples...")

    for i in range(num_samples):
        noise_path = rng.choice(noise_files)
        try:
            noise_audio = load_audio(noise_path)
        except Exception:
            continue

        clip = extract_random_clip(noise_audio)
        clip = normalize_audio(clip, target_rms=0.08)

        # Optionally apply a random pipe filter (shouldn't matter for background)
        pipe_name = str(rng.choice(list(PIPE_PROFILES.keys())))
        pipe_cfg = PIPE_PROFILES[pipe_name]
        clip = apply_butterworth_lpf(clip, pipe_cfg["cutoff_hz"], order=pipe_cfg["order"])

        filename = f"background_{i:06d}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), clip, TARGET_SR)

        pressure = float(rng.choice(PRESSURE_LEVELS))
        metadata_rows.append({
            "filename": filename,
            "label": "background",
            "snr_db": None,
            "pipe_material": pipe_name,
            "pressure_bar": pressure,
            "source_leak": None,
            "source_noise": noise_path.name,
        })

    return metadata_rows


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Omni-Sense Data Synthesizer: Generate training data for acoustic diagnostics."
    )
    parser.add_argument(
        "--leak-dir",
        type=str,
        default="data/raw/leaks",
        help="Directory containing clean leak signal files (WAV or CSV).",
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="data/raw/urbansound8k/audio",
        help="Directory containing UrbanSound8K audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthesized",
        help="Output directory for synthesized training data.",
    )
    parser.add_argument(
        "--samples-per-combo",
        type=int,
        default=10,
        help="Number of random samples per (leak_file × pipe × snr) combination.",
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=500,
        help="Number of background-only (no leak) samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect source files ──
    print("=" * 60)
    print("Omni-Sense Data Synthesizer")
    print("=" * 60)

    leak_files = collect_audio_files(args.leak_dir)
    noise_files = collect_audio_files(args.noise_dir, extensions=(".wav", ".flac", ".ogg"))

    if not leak_files:
        print(f"\n[ERROR] No leak files found in: {args.leak_dir}")
        print("  Please download the Mendeley Water Leakage Dataset and place files there.")
        return

    if not noise_files:
        print(f"\n[ERROR] No noise files found in: {args.noise_dir}")
        print("  Please download UrbanSound8K and place audio files there.")
        return

    print(f"\nFound {len(leak_files)} leak files, {len(noise_files)} noise files.")

    # ── Generate leak samples ──
    print("\n[1/2] Generating leak samples...")
    leak_rows = generate_leak_samples(
        leak_files, noise_files, output_dir,
        samples_per_combo=args.samples_per_combo,
        rng=rng,
    )
    print(f"  → {len(leak_rows)} leak samples generated.")

    # ── Generate background samples ──
    print("\n[2/2] Generating background samples...")
    bg_rows = generate_background_samples(
        noise_files, output_dir,
        num_samples=args.background_samples,
        rng=rng,
    )
    print(f"  → {len(bg_rows)} background samples generated.")

    # ── Write metadata CSV ──
    all_rows = leak_rows + bg_rows
    metadata_path = output_dir / "metadata.csv"
    fieldnames = ["filename", "label", "snr_db", "pipe_material", "pressure_bar",
                  "source_leak", "source_noise"]

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_rows)} total samples → {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
