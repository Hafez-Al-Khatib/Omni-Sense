"""
Omni-Sense Golden Difficult Dataset Generator
==============================================
Creates a high-challenge regression set by mixing clean LeakDB recordings
with hard negatives (MIMII pumps) and synthetic distortions.

Challenges:
  1. Low SNR (Leak + Machinery Noise)
  2. Hard Negatives (Pump noise labeled as Normal_Operation)
  3. Spectral Shift (Pipe material filters)
  4. Temporal Jitter (Amplitude & Phase shifts)
"""
import random
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# Paths
SOURCE_DIR = Path("Processed_audio_16k")
PUMP_DIR = Path("data/raw/Normal_Operation")
OUTPUT_DIR = Path("data/golden_difficult")
MANIFEST_PATH = OUTPUT_DIR / "golden_difficult_v1.csv"

# Parameters
SAMPLE_RATE = 16000
DURATION_S = 5.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_S)

def mix_with_snr(clean, noise, snr_db):
    """Mix signals at target SNR."""
    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2)
    if p_noise == 0:
        return clean
    
    # Calculate required noise power
    target_p_noise = p_clean / (10 ** (snr_db / 10.0))
    factor = np.sqrt(target_p_noise / p_noise)
    
    mixed = clean + (noise * factor)
    # Peak normalize
    if np.max(np.abs(mixed)) > 1.0:
        mixed /= np.max(np.abs(mixed))
    return mixed

def apply_jitter(audio, factor=0.02):
    """Random amplitude jitter."""
    jitter = 1.0 + factor * (np.random.rand(len(audio)) - 0.5)
    return audio * jitter

def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load some source leaks
    source_files = list(SOURCE_DIR.glob("*.wav"))
    pump_files = list(PUMP_DIR.glob("mimii_pump_*.wav"))
    
    if not source_files or not pump_files:
        print("Error: Missing source or pump files.")
        return

    difficult_samples = []

    # -- CATEGORY 1: Low SNR Leaks (The "Invisible" Leak) --
    # Mix a faint leak with loud machinery
    leaks = [f for f in source_files if "No-leak" not in f.name]
    for i in range(5):
        clean_f = random.choice(leaks)
        noise_f = random.choice(pump_files)
        
        c, _ = sf.read(clean_f)
        n, _ = sf.read(noise_f)
        
        # Ensure mono and same length
        if c.ndim > 1:
            c = c.mean(axis=1)
        if n.ndim > 1:
            n = n.mean(axis=1)
        c = c[:TARGET_SAMPLES]
        n = n[:TARGET_SAMPLES]
        
        # Mix at 3dB SNR (very difficult)
        mixed = mix_with_snr(c, n, snr_db=3.0)
        
        out_name = f"difficult_low_snr_{i:02d}.wav"
        sf.write(OUTPUT_DIR / out_name, mixed, SAMPLE_RATE)
        
        # Extract label from filename
        label = "Leak"  # Simplify for golden difficult
        if "Circumferential" in clean_f.name:
            label = "Circumferential_Crack"
        elif "Gasket" in clean_f.name:
            label = "Gasket_Leak"
        elif "Longitudinal" in clean_f.name:
            label = "Longitudinal_Crack"
        elif "Orifice" in clean_f.name:
            label = "Orifice_Leak"

        difficult_samples.append({
            "filename": out_name,
            "label": label,
            "challenge": "low_snr_machinery",
            "source_leak": clean_f.name,
            "source_noise": noise_f.name
        })

    # -- CATEGORY 2: Hard Negatives (The "False Positive" Trap) --
    # Pure abnormal pump noise labeled as Normal_Operation
    abnormal_pumps = [f for f in pump_files if "abnormal" in f.name]
    for i in range(5):
        noise_f = random.choice(abnormal_pumps)
        n, _ = sf.read(noise_f)
        if n.ndim > 1:
            n = n.mean(axis=1)
        n = n[:TARGET_SAMPLES]
        
        out_name = f"difficult_hard_negative_{i:02d}.wav"
        sf.write(OUTPUT_DIR / out_name, n, SAMPLE_RATE)
        
        difficult_samples.append({
            "filename": out_name,
            "label": "Normal_Operation", # The trap!
            "challenge": "hard_negative_machinery",
            "source_leak": "None",
            "source_noise": noise_f.name
        })

    # -- CATEGORY 3: Jittery No-Leak (The "Sensor Drift" Trap) --
    no_leaks = [f for f in source_files if "No-leak" in f.name]
    for i in range(5):
        clean_f = random.choice(no_leaks)
        c, _ = sf.read(clean_f)
        if c.ndim > 1:
            c = c.mean(axis=1)
        c = c[:TARGET_SAMPLES]
        
        # Apply heavy jitter and volume spikes
        distorted = apply_jitter(c, factor=0.1)
        
        out_name = f"difficult_jitter_{i:02d}.wav"
        sf.write(OUTPUT_DIR / out_name, distorted, SAMPLE_RATE)
        
        difficult_samples.append({
            "filename": out_name,
            "label": "No_Leak",
            "challenge": "sensor_jitter",
            "source_leak": clean_f.name,
            "source_noise": "None"
        })

    # Save Manifest
    df = pd.DataFrame(difficult_samples)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Created {len(df)} golden difficult samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()
