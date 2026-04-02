"""
Vibration Feature Extractor
=============================
Replaces YAMNet with signal-domain acoustic features appropriate
for piezoelectric accelerometer data.

Feature vector (208 dimensions):
  MFCCs (40 coeff) mean + std         40 × 2 =  80
  MFCC delta mean + std               40 × 2 =  80
  Spectral centroid mean + std              =   2
  Spectral bandwidth mean + std             =   2
  Spectral rolloff mean + std               =   2
  Spectral contrast (7 bands) mean + std  7 × 2 = 14
  Zero-crossing rate mean + std             =   2
  RMS energy mean + std                     =   2
  Chroma (12 bins) mean + std            12 × 2 = 24
                                        Total = 208
"""

import numpy as np
import librosa

N_FEATURES = 208

_SR = 16000
_N_MFCC = 40
_N_FFT = 2048
_HOP = 512


def _ms(x: np.ndarray) -> np.ndarray:
    """Per-row mean and std across the time axis (axis=-1)."""
    return np.concatenate([x.mean(axis=-1), x.std(axis=-1)]).astype(np.float32)


class FeatureExtractor:
    """
    Stateless vibration feature extractor — no model file needed.

    Interface-compatible with the old YAMNetService so call sites
    only need to update the import and method name.
    """

    @property
    def is_loaded(self) -> bool:
        return True  # always ready; no model file to load

    def load(self):
        """No-op — kept for compatibility with startup hook."""
        pass

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract 208-d feature vector from a preprocessed waveform.

        Args:
            waveform: 1D float32 array at 16 kHz

        Returns:
            208-d float32 feature vector
        """
        y = waveform.astype(np.float32)

        mfcc = librosa.feature.mfcc(
            y=y, sr=_SR, n_mfcc=_N_MFCC, n_fft=_N_FFT, hop_length=_HOP
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP
        )
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP
        )
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=_HOP)
        rms = librosa.feature.rms(y=y, hop_length=_HOP)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=_SR, n_fft=_N_FFT, hop_length=_HOP
        )

        vec = np.concatenate([
            _ms(mfcc),       # 80
            _ms(mfcc_delta), # 80
            _ms(centroid),   #  2
            _ms(bandwidth),  #  2
            _ms(rolloff),    #  2
            _ms(contrast),   # 14
            _ms(zcr),        #  2
            _ms(rms),        #  2
            _ms(chroma),     # 24
        ])  # = 208

        return vec


# Singleton instance
feature_extractor = FeatureExtractor()
