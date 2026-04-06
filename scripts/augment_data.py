"""
Omni-Sense Physics-Valid Data Augmentation Pipeline
=====================================================
Converts processed accelerometer WAV recordings into a diverse training
dataset via sliding-window segmentation and physics-valid augmentation.

Replaces synthesize_data.py, which required UrbanSound8K (airborne
microphone data — rejected by accelerometers, physics-invalid).

Physics-valid augmentation applied:
  1. Sliding-window segmentation  — multiple clips per 35-second recording
  2. Pipe material LPF            — Butterworth filter models structural
                                    transmission characteristics of each material
  3. Amplitude jitter             — models sensor gain variability (±20%)
  4. AWGN                         — models ADC electronic noise floor
  5. Speed perturbation           — models flow velocity variation (±8%),
                                    implemented as time-stretching before re-clipping
  6. Seismic noise mixing         — optional; mixes with real or synthetic
                                    ground-borne vibration at a target SNR

Label taxonomy (derived from filename, matches LeakDB folder structure):
  Circumferential_Crack | Gasket_Leak | Longitudinal_Crack | No_Leak | Orifice_Leak
  + Normal_Operation (if --hard-negatives-dir is provided, e.g. CWRU dataset)

Output:
  data/synthesized/*.wav         (16 kHz mono, 5-second clips)
  data/synthesized/metadata.csv  (label, topology, condition, sensor_id,
                                  pipe_material, pressure_bar, augmentation)

Usage:
    python scripts/augment_data.py \\
        --input-dir Processed_audio_16k \\
        --output-dir data/synthesized \\
        [--window-step 5.0] \\
        [--no-speed-perturb] \\
        [--seismic-dir data/raw/seismic] \\
        [--hard-negatives-dir data/raw/hard_negatives] \\
        [--seed 42]
"""

import argparse
import csv
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_SR = 16_000
WINDOW_S = 5.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_S)  # 80,000

# Butterworth LPF parameters per pipe material.
# Rationale: denser/stiffer materials transmit higher frequencies;
# PVC absorbs high-frequency energy more than steel.
PIPE_PROFILES: dict[str, dict] = {
    "PVC":       {"cutoff_hz": 2000, "order": 4},
    "Steel":     {"cutoff_hz": 6000, "order": 4},
    "Cast_Iron": {"cutoff_hz": 3500, "order": 3},
}

# Condition label found in filename → simulated operating pressure.
# Based on the LeakDB experimental setup documentation.
CONDITION_PRESSURE: dict[str, float] = {
    "0.18_LPS": 2.0,   # low-flow, lower line pressure
    "0.47_LPS": 4.5,   # higher-flow, higher line pressure
    "ND":        3.0,   # normal discharge (steady-state)
    "Transient": 6.0,   # pressure surge / water hammer
    "Unknown":   3.0,
}

# Normalised fault-class names (replace hyphens, keep underscores).
# "No-leak" folder → "No_Leak" label to match Python identifier conventions.
FAULT_CLASS_NORMALISE: dict[str, str] = {
    "No-leak": "No_Leak",
}


# ─── Audio Utilities ──────────────────────────────────────────────────────────

def apply_pipe_lpf(audio: np.ndarray, material: str) -> np.ndarray:
    """Low-pass filter to simulate pipe material's effect on vibration propagation."""
    cfg = PIPE_PROFILES[material]
    nyquist = TARGET_SR / 2.0
    norm_cutoff = min(cfg["cutoff_hz"] / nyquist, 0.99)
    sos = butter(cfg["order"], norm_cutoff, btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def peak_normalise(audio: np.ndarray, headroom: float = 0.99) -> np.ndarray:
    """Normalise to peak amplitude without hard-clipping."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return (audio * (headroom / peak)).astype(np.float32)


def add_awgn(audio: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at amplitude sigma (models ADC noise floor)."""
    noise = rng.normal(0.0, sigma, size=len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def amplitude_jitter(audio: np.ndarray, factor: float) -> np.ndarray:
    """Scale amplitude by factor (models sensor gain variability)."""
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def speed_perturb(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Time-stretch then re-clip to WINDOW_SAMPLES.
    rate > 1.0 → speeds up (simulates higher flow velocity).
    rate < 1.0 → slows down (lower flow velocity).
    """
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) >= WINDOW_SAMPLES:
        return stretched[:WINDOW_SAMPLES].astype(np.float32)
    padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    padded[:len(stretched)] = stretched
    return padded


def mix_seismic_noise(
    audio: np.ndarray,
    seismic_files: list[Path],
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mix audio with a random clip from a seismic/ground-vibration WAV library.

    Seismic files should be real ground-vibration recordings (e.g. downloaded
    from IRIS EarthScope ambient-noise datasets via ObsPy) or broadband
    seismometer channels resampled to 16 kHz. Unlike ESC-50 traffic audio,
    these are true accelerometer-domain signals with physically correct spectra.

    If the seismic library is absent this function is a no-op.
    """
    if not seismic_files:
        return audio

    seismic_path = rng.choice(seismic_files)
    try:
        noise, sr = sf.read(str(seismic_path))
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        noise = noise.astype(np.float32)
        if sr != TARGET_SR:
            noise = librosa.resample(noise, orig_sr=sr, target_sr=TARGET_SR)
    except Exception:
        return audio

    # Extract a random window from the noise recording
    if len(noise) >= WINDOW_SAMPLES:
        start = rng.integers(0, len(noise) - WINDOW_SAMPLES + 1)
        noise_clip = noise[start : start + WINDOW_SAMPLES]
    else:
        noise_clip = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        noise_clip[: len(noise)] = noise

    # Mix at target SNR
    sig_power = np.mean(audio ** 2)
    nse_power = np.mean(noise_clip ** 2)
    if nse_power < 1e-10:
        return audio

    target_nse = sig_power / (10 ** (snr_db / 10))
    noise_clip = noise_clip * np.sqrt(target_nse / nse_power)
    mixed = np.clip(audio + noise_clip, -1.0, 1.0).astype(np.float32)
    return mixed


# ─── Filename Parsing ─────────────────────────────────────────────────────────

def parse_wav_metadata(wav_path: Path) -> dict:
    """
    Extract topology, fault_class, condition, and sensor_id from the WAV
    filename produced by CSVtoWAV.py.

    Expected pattern:
        {Topology}_{FaultClass}_{BR|LO}_{code}_{condition}_{sensor}.wav
    Examples:
        Branched_Circumferential_Crack_BR_CC_0.18_LPS_A1.wav
        Looped_No-leak_LO_NL_ND_A2.wav
        Branched_Orifice_Leak_BR_OL_Transient_A1.wav
    """
    parts = wav_path.stem.split("_")

    # Find the index of the original-stem prefix (BR or LO)
    split_idx = None
    for i, p in enumerate(parts):
        if p in ("BR", "LO"):
            split_idx = i
            break

    if split_idx is None:
        return {
            "topology": "Unknown",
            "fault_class": "Unknown",
            "condition": "Unknown",
            "sensor_id": "Unknown",
        }

    topo_fault_parts = parts[:split_idx]
    stem_parts = parts[split_idx:]

    topology = topo_fault_parts[0] if topo_fault_parts else "Unknown"
    raw_fault = "_".join(topo_fault_parts[1:]) if len(topo_fault_parts) > 1 else "Unknown"
    fault_class = FAULT_CLASS_NORMALISE.get(raw_fault, raw_fault)

    stem_str = "_".join(stem_parts)

    # Condition detection (order matters: check 0.18/0.47 before ND)
    if "0.18" in stem_str:
        condition = "0.18_LPS"
    elif "0.47" in stem_str:
        condition = "0.47_LPS"
    elif "Transient" in stem_str:
        condition = "Transient"
    elif "ND" in stem_str:
        condition = "ND"
    else:
        condition = "Unknown"

    sensor_id = "A2" if stem_str.endswith("A2") else "A1"

    return {
        "topology":    topology,
        "fault_class": fault_class,
        "condition":   condition,
        "sensor_id":   sensor_id,
    }


# ─── Sliding Window Segmentation ──────────────────────────────────────────────

def extract_windows(audio: np.ndarray, step_samples: int) -> list[np.ndarray]:
    """
    Yield non-overlapping (or overlapping) fixed-length windows.
    Drops the trailing partial window to avoid zero-padded boundary effects.
    """
    windows = []
    n = len(audio)
    start = 0
    while start + WINDOW_SAMPLES <= n:
        windows.append(audio[start : start + WINDOW_SAMPLES].astype(np.float32))
        start += step_samples
    return windows


# ─── Core Pipeline ────────────────────────────────────────────────────────────

def process_source_wav(
    wav_path: Path,
    output_dir: Path,
    seismic_files: list[Path],
    rng: np.random.Generator,
    step_s: float,
) -> list[dict]:
    """
    Load one source WAV, segment into windows, and emit all augmented variants.
    Returns a list of metadata dicts (one per output clip).
    """
    try:
        audio, sr = sf.read(str(wav_path))
    except Exception as e:
        print(f"  [SKIP] {wav_path.name}: {e}")
        return []

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    meta = parse_wav_metadata(wav_path)
    label = meta["fault_class"]
    pressure = CONDITION_PRESSURE.get(meta["condition"], 3.0)

    step_samples = int(step_s * TARGET_SR)
    windows = extract_windows(audio, step_samples)

    if not windows:
        print(f"  [SKIP] {wav_path.name}: too short for one window")
        return []

    # Assign ONE pipe material per source recording.
    # Using all 3 materials per window creates intra-class variance > inter-class
    # variance: the dominant LPF signature masks fault-class differences, making
    # all 5 classes appear statistically identical to the classifier.
    material = rng.choice(list(PIPE_PROFILES.keys()))

    rows: list[dict] = []

    for win_idx, window in enumerate(windows):
        # Do NOT peak-normalise: RMS amplitude is a discriminative feature
        # (orifice leaks are louder than hairline cracks). Peak normalisation
        # was previously destroying inter-class amplitude differences and making
        # all feature distributions identical. Clip only to prevent hard distortion.
        window = np.clip(window, -1.0, 1.0).astype(np.float32)
        filtered = apply_pipe_lpf(window, material)

        # ── Variant A: clean (pipe LPF only) ──
        rows.append(_save_clip(
            filtered, output_dir, label, meta, material, pressure,
            win_idx, "clean", rows, rng,
        ))

        # ── Variant B: + AWGN (σ randomly drawn from [0.003, 0.010]) ──
        sigma = rng.uniform(0.003, 0.010)
        noisy = add_awgn(filtered, sigma, rng)
        rows.append(_save_clip(
            noisy, output_dir, label, meta, material, pressure,
            win_idx, "awgn", rows, rng,
        ))

        # ── Variant C: amplitude jitter ──
        jitter_factor = rng.uniform(0.80, 1.20)
        jittered = amplitude_jitter(filtered, jitter_factor)
        rows.append(_save_clip(
            jittered, output_dir, label, meta, material, pressure,
            win_idx, "amp_jitter", rows, rng,
        ))

        # Speed perturbation (Variant D) is intentionally DISABLED for vibration data.
        # Time-stretching shifts all MFCC-derived spectral features by ±8%, adding
        # intra-class spectral noise that overwhelms the subtle inter-class differences.
        # It is valid for speech (pitch/tempo vary naturally) but harmful here because
        # the specific spectral shape of a leak IS the discriminative fingerprint.

        # ── Variant D: seismic noise mix (optional) ──
        if seismic_files:
            snr_db = float(rng.choice([5, 10, 15]))
            seismic_mixed = mix_seismic_noise(filtered, seismic_files, snr_db, rng)
            rows.append(_save_clip(
                seismic_mixed, output_dir, label, meta, material, pressure,
                win_idx, f"seismic_snr{snr_db:.0f}", rows, rng,
            ))

    return rows


def _save_clip(
    audio: np.ndarray,
    output_dir: Path,
    label: str,
    meta: dict,
    material: str,
    pressure: float,
    win_idx: int,
    aug_tag: str,
    rows: list,
    rng: np.random.Generator,
) -> dict:
    """Write a single clip to disk and return its metadata row."""
    clip_id = len(rows)
    filename = f"clip_{clip_id:07d}.wav"
    sf.write(str(output_dir / filename), audio, TARGET_SR)
    return {
        "filename":     filename,
        "label":        label,
        "topology":     meta["topology"],
        "condition":    meta["condition"],
        "sensor_id":    meta["sensor_id"],
        "pipe_material": material,
        "pressure_bar": round(pressure, 2),
        "augmentation": aug_tag,
        "source_wav":   meta.get("source", ""),
    }


def process_hard_negatives(
    neg_dir: Path,
    output_dir: Path,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Process hard-negative recordings (e.g. CWRU bearing dataset, MFPT).
    These are labelled 'Normal_Operation' and are NOT put through pipe-material
    filters (they are machinery vibrations, not pipe-borne signals).
    Augmentation: amplitude jitter + AWGN only.
    """
    wav_files = list(neg_dir.rglob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No WAV files found in hard-negatives dir: {neg_dir}")
        return []

    rows: list[dict] = []
    for wav_path in wav_files:
        try:
            audio, sr = sf.read(str(wav_path))
        except Exception as e:
            print(f"  [SKIP] {wav_path.name}: {e}")
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        step_samples = WINDOW_SAMPLES
        windows = extract_windows(audio, step_samples)

        for win_idx, window in enumerate(windows):
            window = peak_normalise(window)

            for aug_tag, aug_audio in [
                ("clean", window),
                ("awgn",  add_awgn(window, rng.uniform(0.003, 0.010), rng)),
            ]:
                clip_id = len(rows)
                filename = f"neg_{clip_id:07d}.wav"
                sf.write(str(output_dir / filename), aug_audio, TARGET_SR)
                rows.append({
                    "filename":      filename,
                    "label":         "Normal_Operation",
                    "topology":      "N/A",
                    "condition":     "N/A",
                    "sensor_id":     "N/A",
                    "pipe_material": "N/A",
                    "pressure_bar":  0.0,
                    "augmentation":  aug_tag,
                    "source_wav":    wav_path.name,
                })

    return rows


# ─── CLI ──────────────────────────────────────────────────────────────────────

METADATA_FIELDS = [
    "filename", "label", "topology", "condition", "sensor_id",
    "pipe_material", "pressure_bar", "augmentation", "source_wav",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Omni-Sense physics-valid augmentation pipeline. "
            "Reads from Processed_audio_16k/, emits 5-second clips + metadata.csv."
        )
    )
    parser.add_argument(
        "--input-dir", default="Processed_audio_16k",
        help="Directory of 16 kHz WAV files from CSVtoWAV.py (default: Processed_audio_16k).",
    )
    parser.add_argument(
        "--output-dir", default="data/synthesized",
        help="Output directory for augmented clips and metadata.csv.",
    )
    parser.add_argument(
        "--window-step", type=float, default=5.0,
        help="Sliding window step in seconds (default: 5.0 → non-overlapping). "
             "Use 2.5 for 50%% overlap.",
    )
    parser.add_argument(
        "--no-speed-perturb", action="store_true",
        help="Disable speed-perturbation augmentation (saves time/disk).",
    )
    parser.add_argument(
        "--seismic-dir", default=None,
        help="Optional directory of ground-vibration WAV files (e.g. from IRIS/EarthScope) "
             "for seismic-noise mixing augmentation.",
    )
    parser.add_argument(
        "--hard-negatives-dir", default=None,
        help="Optional directory of WAV files for hard-negative (Normal_Operation) samples "
             "(e.g. CWRU bearing dataset converted to WAV).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Omni-Sense Physics-Valid Augmentation Pipeline")
    print("=" * 60)
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Window step: {args.window_step}s")
    print(f"  Speed perturb: {'disabled' if args.no_speed_perturb else 'enabled'}")

    # ── Collect source WAVs ──
    source_wavs = sorted(input_dir.glob("*.wav"))
    if not source_wavs:
        print(f"\n[ERROR] No WAV files found in: {input_dir}")
        print("  Run CSVtoWAV.py first.")
        return
    print(f"\n  Found {len(source_wavs)} source WAV files.")

    # ── Collect optional seismic WAVs ──
    seismic_files: list[Path] = []
    if args.seismic_dir:
        seismic_files = sorted(Path(args.seismic_dir).rglob("*.wav"))
        print(f"  Seismic noise files: {len(seismic_files)}")

    # ── Process source recordings ──
    print(f"\n[1/2] Augmenting source recordings...")
    all_rows: list[dict] = []

    for i, wav_path in enumerate(source_wavs, 1):
        rows = process_source_wav(
            wav_path,
            output_dir,
            seismic_files,
            rng=rng,
            step_s=args.window_step,
        )
        # Attach source filename to each row
        for r in rows:
            r["source_wav"] = wav_path.name
        all_rows.extend(rows)

        if i % 10 == 0 or i == len(source_wavs):
            print(f"  → {i}/{len(source_wavs)} files processed ({len(all_rows)} clips so far)")

    # ── Process hard negatives (optional) ──
    if args.hard_negatives_dir:
        print(f"\n[2/2] Processing hard negatives from: {args.hard_negatives_dir}")
        neg_rows = process_hard_negatives(
            Path(args.hard_negatives_dir), output_dir, rng
        )
        all_rows.extend(neg_rows)
        print(f"  → {len(neg_rows)} hard-negative clips generated.")
    else:
        print("\n[2/2] No hard-negatives dir specified — skipping.")

    # ── Write metadata CSV ──
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Summary ──
    from collections import Counter
    label_counts = Counter(r["label"] for r in all_rows)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_rows)} clips → {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"\nLabel distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl:<25} {cnt:>5}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
