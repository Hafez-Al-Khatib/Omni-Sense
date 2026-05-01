/**
 * omni_features.h — On-device 39-d DSP feature extractor
 * ========================================================
 * C++ port of omni/eep/features.py (pure NumPy reference). Bit-for-bit
 * compatible with the cloud extractor when the input sample rate matches:
 * features computed here at 3200 Hz are directly comparable to the cloud
 * features extracted from the same WAV resampled to 3200 Hz, so the
 * tiny on-device autoencoder OOD model trained on the
 * eep_features_3200hz.parquet corpus generalises with no domain shift.
 *
 * Feature layout (39 floats; matches omni/eep/features.py)
 * --------------------------------------------------------
 *   [ 0]  rms_mean       [ 1]  rms_std
 *   [ 2]  zcr_mean       [ 3]  zcr_std
 *   [ 4]  kurtosis       [ 5]  skewness        [ 6]  crest_factor
 *   [ 7]  centroid_mean  [ 8]  centroid_std
 *   [ 9]  rolloff_mean   [10]  rolloff_std
 *   [11]  flatness_mean  [12]  flatness_std
 *   [13..38] interleaved [mfcc0_mean, mfcc0_std, … mfcc12_mean, mfcc12_std]
 *
 * Memory budget (host-side at 3200 Hz, 5 s buffer)
 * -----------------------------------------------
 *   Input PCM (16000 × int16):     32 KB   (caller-owned)
 *   Hanning window:                  2 KB   (one-time)
 *   Mel filterbank (40 × 257):      ~41 KB  (one-time)
 *   DCT matrix (40 × 13):            2 KB   (one-time)
 *   Per-call FFT scratch:            6 KB   (re/im arrays, on stack)
 *   Per-call accumulators:          ~1 KB
 *   ----------------------------------------
 *   Total peak working set:         ~85 KB  (well under ESP32-S3 320 KB SRAM)
 *
 * Latency
 * -------
 *   At 240 MHz on ESP32-S3, ~61 frames per 5 s buffer × ~1 ms FFT/frame
 *   ≈ 60 ms per buffer. Feature publication interval is 1 frame/sec, so
 *   we use <10 % of available CPU.
 */

#pragma once
#include <stddef.h>
#include <stdint.h>

namespace omni {

// Constants — MUST match omni/eep/features.py
constexpr int   FEATURE_DIM = 39;
constexpr int   FRAME_LEN   = 512;
constexpr int   HOP         = 256;
constexpr int   N_FFT       = 512;
constexpr int   N_FREQ_BINS = N_FFT / 2 + 1;   // 257
constexpr int   N_MELS      = 40;
constexpr int   N_MFCC      = 13;
constexpr float ROLL_PCT    = 0.85f;

/**
 * Initialise windows, mel filterbank, and DCT matrix for the given sample
 * rate. Idempotent — calling twice with the same sr is a no-op. Must be
 * called once at startup before compute_features().
 *
 * Allocations happen lazily on first call, so RAM is not consumed until
 * the device actually runs the on-device feature path.
 */
void init(int sr);

/**
 * Compute the 39-d DSP feature vector.
 *
 * @param pcm  int16 mono audio buffer. For deployment-rate parity with
 *             the cloud models, length should be 5 × sr (e.g. 16000
 *             samples for sr=3200).
 * @param n    Number of samples in pcm. Must be ≥ FRAME_LEN.
 * @param sr   Sample rate in Hz. Must match the value passed to init().
 * @param out  Caller-allocated output array of FEATURE_DIM floats.
 *
 * @return     true on success, false on argument error or if init() was
 *             not called with matching sr.
 *
 * Conversion: int16 samples are scaled by 1/32768 to match the cloud
 * extractor which operates on float32 in [-1, 1].
 */
bool compute_features(const int16_t* pcm, int n, int sr, float* out);

}  // namespace omni
