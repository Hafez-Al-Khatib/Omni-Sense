#!/usr/bin/env python3
"""
Extract pump WAV files from MIMII ZIP archives and convert for use as hard negatives.

Processes 8-channel 16kHz WAV files, converts to mono, trims/pads to 5 seconds,
and saves with standardized naming for the Omni-Sense pipeline.
"""

import argparse
import random
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav_from_zip(zip_file, file_path):
    """
    Read WAV file directly from zip stream without extracting to disk.

    Returns:
        tuple: (audio_data, sample_rate) where audio_data is numpy array
               or (None, None) if file is corrupt
    """
    try:
        with zip_file.open(file_path) as f:
            # Read RIFF header
            riff = f.read(4)
            if riff != b'RIFF':
                return None, None

            struct.unpack('<I', f.read(4))[0]
            wave = f.read(4)
            if wave != b'WAVE':
                return None, None

            # Find fmt subchunk
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    return None, None

                chunk_size = struct.unpack('<I', f.read(4))[0]

                if chunk_id == b'fmt ':
                    # Parse format
                    fmt_data = f.read(chunk_size)
                    (audio_format, num_channels, sample_rate, byte_rate,
                     block_align, bits_per_sample) = struct.unpack('<HHIIHH', fmt_data[:16])

                    if audio_format not in [1, 65534]:  # PCM or Extensible PCM
                        return None, None

                    break
                else:
                    # Skip this chunk
                    f.read(chunk_size)
                    if chunk_size % 2:
                        f.read(1)  # Padding

            # Find data subchunk
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    return None, None

                chunk_size = struct.unpack('<I', f.read(4))[0]

                if chunk_id == b'data':
                    # Read audio data
                    audio_bytes = f.read(chunk_size)

                    # Parse based on bit depth
                    if bits_per_sample == 16:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    elif bits_per_sample == 24:
                        # Handle 24-bit audio
                        audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                        audio_data = audio_data.reshape(-1, 3)
                        # Convert 24-bit to 32-bit
                        audio_int32 = np.zeros(len(audio_data), dtype=np.int32)
                        for i in range(len(audio_data)):
                            audio_int32[i] = (audio_data[i, 0] |
                                            (audio_data[i, 1] << 8) |
                                            ((audio_data[i, 2] ^ 0x80) << 16))
                        audio_data = audio_int32
                    else:
                        return None, None

                    # Reshape to (num_samples, num_channels)
                    num_samples = len(audio_data) // num_channels
                    audio_data = audio_data[:num_samples * num_channels].reshape(num_samples, num_channels)

                    return audio_data, sample_rate
                else:
                    # Skip this chunk
                    f.read(chunk_size)
                    if chunk_size % 2:
                        f.read(1)  # Padding

    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return None, None


def convert_to_mono(audio_data):
    """Convert multi-channel audio to mono by averaging all channels."""
    if audio_data.shape[1] == 1:
        return audio_data[:, 0]
    else:
        return np.mean(audio_data.astype(np.float32), axis=1)


def trim_or_pad(audio, target_samples=80000):
    """Trim or pad audio to exact target length."""
    if len(audio) >= target_samples:
        return audio[:target_samples]
    else:
        padding = target_samples - len(audio)
        return np.pad(audio, (0, padding), mode='constant', constant_values=0)


def extract_mimii_negatives(zip_files, output_dir, max_per_zip=200,
                           snr_preference="0_dB", seed=42):
    """
    Extract pump WAV files from MIMII ZIP archives.

    Args:
        zip_files: List of paths to ZIP files
        output_dir: Output directory for converted WAV files
        max_per_zip: Maximum number of files to extract per ZIP
        snr_preference: Preferred SNR ZIP ("0_dB", "-6_dB", or "6_dB")
        seed: Random seed for reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect files by zip and category
    stats = defaultdict(lambda: {"normal": 0, "abnormal": 0})
    target_samples = 80000  # 5 seconds at 16kHz

    # Process preferred SNR ZIP first
    preferred_zip = None
    other_zips = []

    for zip_file in zip_files:
        if snr_preference in str(zip_file):
            preferred_zip = zip_file
        else:
            other_zips.append(zip_file)

    zips_to_process = []
    if preferred_zip:
        zips_to_process.append(preferred_zip)
    zips_to_process.extend(other_zips)

    files_saved = 0

    for zip_path in zips_to_process:
        zip_path = Path(zip_path)
        if not zip_path.exists():
            print(f"Warning: ZIP file not found: {zip_path}")
            continue

        print(f"\nProcessing: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List all WAV files in the ZIP
                wav_files = defaultdict(lambda: {"normal": [], "abnormal": []})

                for file_info in zf.filelist:
                    if file_info.filename.endswith('.wav'):
                        # Parse path: pump/id_XX/normal or abnormal/*.wav
                        parts = file_info.filename.split('/')
                        if len(parts) >= 3 and parts[0] == 'pump':
                            pump_id = parts[1]  # e.g., 'id_00'
                            category = parts[2]  # 'normal' or 'abnormal'

                            if category in ['normal', 'abnormal']:
                                wav_files[pump_id][category].append(file_info.filename)

                # Sample files from preferred SNR, balance across pump IDs
                for pump_id in sorted(wav_files.keys()):
                    for category in ['normal', 'abnormal']:
                        files = wav_files[pump_id][category]

                        # Sample up to max_per_zip files
                        sample_size = min(len(files), max_per_zip)
                        if sample_size == 0:
                            continue

                        sampled_files = random.sample(files, sample_size)

                        for idx, file_path in enumerate(sampled_files):
                            # Read WAV from ZIP
                            audio_data, sample_rate = read_wav_from_zip(zf, file_path)

                            if audio_data is None:
                                continue

                            if sample_rate != 16000:
                                print(f"  Warning: {file_path} has sample rate {sample_rate}, skipping")
                                continue

                            # Convert to mono
                            mono_audio = convert_to_mono(audio_data)

                            # Normalize to prevent clipping
                            if np.max(np.abs(mono_audio)) > 0:
                                mono_audio = mono_audio / np.max(np.abs(mono_audio)) * 32767

                            # Trim/pad to target length
                            processed_audio = trim_or_pad(mono_audio, target_samples)

                            # Save as mono 16kHz WAV
                            output_file = (output_path /
                                         f"mimii_pump_{pump_id}_{category}_{idx:04d}.wav")

                            # Convert to int16 for saving
                            processed_audio = np.clip(processed_audio, -32768, 32767).astype(np.int16)
                            sf.write(str(output_file), processed_audio, 16000, subtype='PCM_16')

                            stats[pump_id][category] += 1
                            files_saved += 1

                        print(f"  {pump_id}/{category}: Sampled {sample_size} files, "
                              f"saved {stats[pump_id][category]} successfully")

        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
            continue

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)

    total_by_category = {"normal": 0, "abnormal": 0}
    for pump_id in sorted(stats.keys()):
        normal_count = stats[pump_id]["normal"]
        abnormal_count = stats[pump_id]["abnormal"]
        total = normal_count + abnormal_count

        total_by_category["normal"] += normal_count
        total_by_category["abnormal"] += abnormal_count

        print(f"{pump_id}: {normal_count:4d} normal, {abnormal_count:4d} abnormal "
              f"(total: {total:4d})")

    print("-"*60)
    print(f"Total: {total_by_category['normal']:4d} normal, "
          f"{total_by_category['abnormal']:4d} abnormal "
          f"(total: {files_saved:4d})")
    print(f"Output directory: {output_path.absolute()}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract pump WAV files from MIMII ZIP archives for hard negatives"
    )
    parser.add_argument(
        "--zip-files",
        nargs="+",
        required=True,
        help="Paths to MIMII ZIP files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/Normal_Operation",
        help="Output directory for converted WAV files"
    )
    parser.add_argument(
        "--max-per-zip",
        type=int,
        default=200,
        help="Maximum files to extract per ZIP file"
    )
    parser.add_argument(
        "--snr-preference",
        default="0_dB",
        choices=["0_dB", "-6_dB", "6_dB"],
        help="Preferred SNR ZIP file to prioritize"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    extract_mimii_negatives(
        zip_files=args.zip_files,
        output_dir=args.output_dir,
        max_per_zip=args.max_per_zip,
        snr_preference=args.snr_preference,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
