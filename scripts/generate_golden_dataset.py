"""
Omni-Sense Golden Dataset Generator
=====================================
Curates a small, fixed-composition test set from the processed WAV recordings.
This set NEVER changes after creation — it is the regression gate used by
iep2/tests/test_model_regression.py to ensure model updates don't degrade
performance on known, representative samples.

Selection strategy:
  For each (fault_class × topology × condition) combination that exists in
  Processed_audio_16k/, pick one sensor-A1 recording.
  This yields a balanced cross-section: every fault type, both topologies,
  all flow conditions — without leaking training data (these are raw WAVs,
  not augmented clips).

Output:
  data/golden/
    ├── *.wav                        (copied WAV files)
    └── golden_dataset_v1.csv        (filename, label, topology, condition)

Usage:
    python scripts/generate_golden_dataset.py \\
        --input-dir Processed_audio_16k \\
        --output-dir data/golden
"""

import argparse
import csv
import shutil
from pathlib import Path

# Matches parse logic in augment_data.py
_FAULT_CLASS_NORMALISE = {"No-leak": "No_Leak"}

_CONDITION_PRESSURE = {
    "0.18_LPS": 2.0,
    "0.47_LPS": 4.5,
    "ND":        3.0,
    "Transient": 6.0,
    "Unknown":   3.0,
}


def parse_wav_metadata(wav_path: Path) -> dict:
    parts = wav_path.stem.split("_")
    split_idx = next((i for i, p in enumerate(parts) if p in ("BR", "LO")), None)
    if split_idx is None:
        return {"topology": "Unknown", "fault_class": "Unknown",
                "condition": "Unknown", "sensor_id": "Unknown"}

    topo_fault = parts[:split_idx]
    stem_str = "_".join(parts[split_idx:])

    topology   = topo_fault[0] if topo_fault else "Unknown"
    raw_fault  = "_".join(topo_fault[1:]) if len(topo_fault) > 1 else "Unknown"
    fault_class = _FAULT_CLASS_NORMALISE.get(raw_fault, raw_fault)

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
    return {"topology": topology, "fault_class": fault_class,
            "condition": condition, "sensor_id": sensor_id}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fixed golden test set for model regression gating."
    )
    parser.add_argument("--input-dir",  default="Processed_audio_16k")
    parser.add_argument("--output-dir", default="data/golden")
    parser.add_argument("--windows-per-file", type=int, default=3, help="Number of 5s windows to extract per file.")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_wavs = sorted(input_dir.glob("*.wav"))
    if not source_wavs:
        print(f"[ERROR] No WAV files found in {input_dir}")
        return

    import librosa
    import soundfile as sf
    import numpy as np
    from extract_dsp_features import extract_dsp_features

    TARGET_SR = 16000
    DURATION_S = 5.0
    SAMPLES_PER_WINDOW = int(TARGET_SR * DURATION_S)

    selected_rows = []
    print(f"Processing {len(source_wavs)} source recordings...")

    for wav_path in source_wavs:
        meta = parse_wav_metadata(wav_path)
        
        try:
            # We use a temp file logic or just load and slice
            y, sr = librosa.load(wav_path, sr=TARGET_SR)
            total_samples = len(y)
            
            # Calculate non-overlapping windows
            max_windows = total_samples // SAMPLES_PER_WINDOW
            num_windows = min(args.windows_per_file, max_windows)
            
            if num_windows == 0:
                print(f"  [SKIP] {wav_path.name}: too short ({total_samples} samples)")
                continue

            for i in range(num_windows):
                start = i * SAMPLES_PER_WINDOW
                end = start + SAMPLES_PER_WINDOW
                window = y[start:end]
                
                window_filename = f"golden_{wav_path.stem}_win{i}.wav"
                dest = output_dir / window_filename
                sf.write(dest, window, TARGET_SR)
                
                # Extract DSP features for this window
                # (extract_dsp_features expects a path, so we use the newly saved file)
                embeddings = extract_dsp_features(dest)
                
                row = {
                    "filename":    window_filename,
                    "label":       meta["fault_class"],
                    "topology":    meta["topology"],
                    "condition":   meta["condition"],
                    "sensor_id":   meta["sensor_id"],
                    "pipe_material": meta["topology"], # Fallback for feature prep
                    "pressure_bar": _CONDITION_PRESSURE.get(meta["condition"], 3.0),
                }
                
                # Add embedding columns
                for idx, val in enumerate(embeddings):
                    row[f"embedding_{idx}"] = val
                    
                selected_rows.append(row)
        except Exception as e:
            print(f"  [ERROR] Processing {wav_path.name}: {e}")

    if not selected_rows:
        print("[ERROR] No samples processed.")
        return

    csv_path = output_dir / "golden_dataset_v1.csv"
    # Dynamic fieldnames based on number of embeddings
    emb_cols = [f"embedding_{i}" for i in range(len(embeddings))]
    fieldnames = ["filename", "label", "topology", "condition", "sensor_id", "pipe_material", "pressure_bar"] + emb_cols
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    from collections import Counter
    label_counts = Counter(r["label"] for r in selected_rows)

    print(f"\nExpanded Golden dataset: {len(selected_rows)} samples → {output_dir}")
    print(f"Manifest: {csv_path}")
    print("\nDistribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<25} {count:>3}")


if __name__ == "__main__":
    main()
