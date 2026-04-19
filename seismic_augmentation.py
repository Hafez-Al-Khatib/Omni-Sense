import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    """A Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_seismic_filter(data, fs, cutoff=250.0, order=5):
    """
    Applies the filter to simulate 2 meters of soil.
    Cuts off almost all frequencies above 250Hz
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def mix_with_snr(clean_signal, noise_signal, snr_db):
    """Mixes clean leak data with seismic noise at a target SNR"""
    min_len = min(len(clean_signal), len(noise_signal))
    clean = clean_signal[:min_len]
    noise = noise_signal[:min_len]

    power_clean = np.var(clean)
    power_noise = np.var(noise)

    if power_noise == 0:
        return clean

    target_noise_power = power_clean / (10 ** (snr_db / 10))
    noise_multiplier = np.sqrt(target_noise_power / power_noise)

    mixed_signal = clean + (noise * noise_multiplier)
    max_val = np.max(np.abs(mixed_signal))

    if max_val > 1.0:
        mixed_signal = mixed_signal / max_val

    return mixed_signal

if __name__ == '__main__':
    SAMPLE_RATE = 16000

    clean_audio, _ = librosa.load('procssed_gasket_leak.wav', sr=SAMPLE_RATE)

    raw_traffic_noise, _ = librosa.load('esc50_heavy_traffic.wav', sr=SAMPLE_RATE)

    seismic_rumble = apply_seismic_filter(raw_traffic_noise, SAMPLE_RATE, cutoff=250.0)

    synthetic_lebanese_environment = mix_with_snr(clean_audio, seismic_rumble, snr_db=10)

    final_audio = np.int16(synthetic_lebanese_environment * 32767)
    wavfile.write('augmented_gasket_leak_with_traffic.wav', SAMPLE_RATE, final_audio)
    print("Seismic augmentation complete.")
