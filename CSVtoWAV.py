import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile

dataset_root = Path('./Accelerometer')
processed_dir = Path('./Processed_audio_16k')
processed_dir.mkdir(parents=True, exist_ok=True)

metadata_manifest = []

csv_files = list(dataset_root.rglob('*.csv'))
print(f"Found {len(csv_files)} files. Starting automated pipeline...")

for csv_path in csv_files:
    try:
        fault_type = csv_path.parent.name
        topology = csv_path.parent.parent.name
        file_stem = csv_path.stem

        df = pd.read_csv(csv_path)
        raw_signal = df['Value'].values

        raw_signal_16k = librosa.resample(y=raw_signal, orig_sr=25600, target_sr=16000)

        max_val = np.max(np.abs(raw_signal_16k))
        normalized = raw_signal_16k / max_val if max_val > 0 else raw_signal_16k

        audio_data = np.int16(normalized * 32767)

        wav_filename = f"{topology}_{fault_type}_{file_stem}.wav".replace(" ", "_")
        wav_out_path = processed_dir / wav_filename
        wavfile.write(wav_out_path, 16000, audio_data)

        # Build XGBoost Manifest
        metadata_manifest.append({
            "audio_file" : wav_filename,
            "topology" : topology,
            "fault_class" : fault_type,
            "original_sample_rate" : 25600,
            "flow_rate_extracted" : "0.18" if "0.18" in file_stem else "unknown"
        })

    except Exception:
        print(f"Skipping {csv_path.name}: e")


manifest_path = processed_dir/'xgboost_training_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(metadata_manifest, f, indent=4)

