"""
Audio Preprocessing for IEP1
==============================
Handles audio decoding, resampling, mono conversion,
and duration enforcement for YAMNet input.
"""

import io

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
TARGET_DURATION_S = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION_S)


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw audio bytes for YAMNet inference.

    Steps:
        1. Decode audio from bytes (WAV, OGG, FLAC)
        2. Convert to mono
        3. Resample to 16kHz
        4. Enforce 5-second duration (pad or truncate)
        5. Normalize amplitude

    Args:
        audio_bytes: Raw audio file bytes

    Returns:
        1D float32 numpy array at 16kHz, exactly 80,000 samples

    Raises:
        ValueError: If audio cannot be decoded or is invalid
    """
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise ValueError(f"Cannot decode audio: {e}")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Enforce exactly 5 seconds
    if len(audio) > TARGET_SAMPLES:
        # Take the center clip
        start = (len(audio) - TARGET_SAMPLES) // 2
        audio = audio[start:start + TARGET_SAMPLES]
    elif len(audio) < TARGET_SAMPLES:
        # Pad with zeros
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    # Soft-clip to [-1, 1] without rescaling.
    # Peak normalization is intentionally NOT applied: the augmentation pipeline
    # outputs calibrated amplitudes, and RMS energy is a discriminative feature
    # between fault classes (orifice leaks are louder than hairline cracks).
    # Re-normalising each window independently would destroy this information.
    audio = np.clip(audio, -1.0, 1.0)

    return audio
