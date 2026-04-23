"""Audio preprocessing for IEP4 — identical to IEP1 (shared logic)."""

import io

import librosa
import numpy as np
import soundfile as sf

_TARGET_SR = 16000
_TARGET_SAMPLES = _TARGET_SR * 5  # 80 000 samples


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot decode audio: {exc}")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != _TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)

    if len(audio) > _TARGET_SAMPLES:
        start = (len(audio) - _TARGET_SAMPLES) // 2
        audio = audio[start:start + _TARGET_SAMPLES]
    elif len(audio) < _TARGET_SAMPLES:
        audio = np.pad(audio, (0, _TARGET_SAMPLES - len(audio)))

    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak

    return audio
