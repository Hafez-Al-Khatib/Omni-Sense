/**
 * omni_features.cpp — On-device 39-d DSP feature extractor (impl)
 *
 * Algorithmic notes
 * -----------------
 * This is a straight port of omni/eep/features.py. Every coefficient is
 * computed with the same formula and ordering, so feature vectors from
 * this code are bit-for-bit comparable (within float32 round-off) to
 * those from the cloud Python implementation when both run on the same
 * input audio at the same sample rate.
 *
 * Why pure C++ (no ESP-DSP)
 * --------------------------
 * Portability and unit-test simplicity. ESP-DSP gives a ~3× speed-up on
 * the FFT, which matters when frame rates exceed a few hundred Hz; we
 * publish one 5-second window per second, leaving ~99 % CPU idle even
 * with the naive radix-2 FFT below. If a future revision needs higher
 * frame rates, swap rfft() for dsps_fft2r_fc32() and dsps_dct() for the
 * matrix multiply — the public ABI in omni_features.h does not change.
 */

#include "omni_features.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

namespace omni {

// ─── Cached state (initialised by init()) ─────────────────────────────
static int     g_sr      = 0;       // 0 = uninitialised
static float   g_window[FRAME_LEN];                      // Hanning, w[i]
static float   g_mel[N_MELS][N_FREQ_BINS];               // mel filterbank
static float   g_dct[N_MFCC][N_MELS];                    // DCT-II rows 0..12
static float   g_freqs[N_FREQ_BINS];                     // bin centre freqs

// ─── One-time tables ─────────────────────────────────────────────────

static void build_window() {
    // Symmetric Hanning to match numpy.hanning() (matches features.py)
    for (int i = 0; i < FRAME_LEN; ++i) {
        g_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (FRAME_LEN - 1)));
    }
}

static void build_freqs(int sr) {
    for (int k = 0; k < N_FREQ_BINS; ++k) {
        g_freqs[k] = (float)k * sr / (float)N_FFT;
    }
}

static inline float hz_to_mel(float f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}
static inline float mel_to_hz(float m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

static void build_mel_filterbank(int sr) {
    // Mirrors omni/eep/features.py::_mel_filterbank() exactly.
    const float f_min = 0.0f;
    const float f_max = sr * 0.5f;
    const float m_min = hz_to_mel(f_min);
    const float m_max = hz_to_mel(f_max);

    int bin_points[N_MELS + 2];
    for (int m = 0; m < N_MELS + 2; ++m) {
        float mel = m_min + (m_max - m_min) * m / (N_MELS + 1);
        float hz  = mel_to_hz(mel);
        bin_points[m] = (int)floorf((N_FFT + 1) * hz / sr);
        if (bin_points[m] < 0) bin_points[m] = 0;
        if (bin_points[m] >= N_FREQ_BINS) bin_points[m] = N_FREQ_BINS - 1;
    }

    memset(g_mel, 0, sizeof(g_mel));
    for (int m = 1; m <= N_MELS; ++m) {
        int f_lo = bin_points[m - 1];
        int f_md = bin_points[m];
        int f_hi = bin_points[m + 1];
        for (int k = f_lo; k < f_md; ++k) {
            if (f_md != f_lo) g_mel[m - 1][k] = (float)(k - f_lo) / (f_md - f_lo);
        }
        for (int k = f_md; k < f_hi; ++k) {
            if (f_hi != f_md) g_mel[m - 1][k] = (float)(f_hi - k) / (f_hi - f_md);
        }
    }
}

static void build_dct() {
    // Type-II DCT, unnormalised — matches omni/eep/features.py::_dct2().
    // We only keep the first N_MFCC rows.
    for (int k = 0; k < N_MFCC; ++k) {
        for (int n = 0; n < N_MELS; ++n) {
            g_dct[k][n] = cosf((float)M_PI * k * (2 * n + 1) / (2.0f * N_MELS));
        }
    }
}

void init(int sr) {
    if (sr == g_sr) return;       // idempotent
    g_sr = sr;
    build_window();
    build_freqs(sr);
    build_mel_filterbank(sr);
    build_dct();
}

// ─── In-place radix-2 complex FFT (Cooley-Tukey) ─────────────────────

static void fft_radix2(float* re, float* im, int n) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
        int k = n >> 1;
        while (j >= k) { j -= k; k >>= 1; }
        j += k;
    }
    // Butterflies
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wre = cosf(ang), wim = sinf(ang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            float cre = 1.0f, cim = 0.0f;
            for (int k = 0; k < half; ++k) {
                int a = i + k;
                int b = a + half;
                float tre = re[b] * cre - im[b] * cim;
                float tim = re[b] * cim + im[b] * cre;
                re[b] = re[a] - tre;
                im[b] = im[a] - tim;
                re[a] += tre;
                im[a] += tim;
                float ncre = cre * wre - cim * wim;
                cim = cre * wim + cim * wre;
                cre = ncre;
            }
        }
    }
}

// ─── Per-frame helpers ───────────────────────────────────────────────

// Windowed magnitude spectrum into mag[N_FREQ_BINS]
static void frame_magnitude(const float* frame, float* mag) {
    float re[N_FFT], im[N_FFT];
    for (int i = 0; i < FRAME_LEN; ++i) {
        re[i] = frame[i] * g_window[i];
        im[i] = 0.0f;
    }
    // FRAME_LEN == N_FFT in our config, so no zero-padding needed.
    fft_radix2(re, im, N_FFT);
    for (int k = 0; k < N_FREQ_BINS; ++k) {
        mag[k] = sqrtf(re[k] * re[k] + im[k] * im[k]);
    }
}

static float frame_rms(const float* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * x[i];
    return sqrtf((float)(s / n)) + 1e-9f;
}

static float frame_zcr(const float* x, int n) {
    int zc = 0;
    for (int i = 1; i < n; ++i) {
        if ((x[i - 1] >= 0.0f) != (x[i] >= 0.0f)) ++zc;
    }
    return (float)zc / (float)(n - 1);
}

static float spectral_centroid(const float* mag) {
    double num = 0.0, den = 1e-9;
    for (int k = 0; k < N_FREQ_BINS; ++k) {
        num += (double)mag[k] * g_freqs[k];
        den += mag[k];
    }
    return (float)(num / den);
}

static float spectral_rolloff(const float* mag) {
    double total = 0.0;
    for (int k = 0; k < N_FREQ_BINS; ++k) total += mag[k];
    double thr = total * ROLL_PCT;
    double cum = 0.0;
    for (int k = 0; k < N_FREQ_BINS; ++k) {
        cum += mag[k];
        if (cum >= thr) return g_freqs[k];
    }
    return g_freqs[N_FREQ_BINS - 1];
}

static float spectral_flatness(const float* mag) {
    const float eps = 1e-9f;
    double log_sum = 0.0, arith = 0.0;
    for (int k = 0; k < N_FREQ_BINS; ++k) {
        log_sum += log((double)mag[k] + eps);
        arith   += mag[k];
    }
    double log_mean = log_sum / N_FREQ_BINS;
    double a = arith / N_FREQ_BINS + eps;
    return (float)(exp(log_mean) / a);
}

// (n_mels) log-mel-power then DCT → (n_mfcc) coefficients
static void frame_mfccs(const float* mag, float* mfcc_out) {
    float mel_power[N_MELS];
    for (int m = 0; m < N_MELS; ++m) {
        double s = 1e-9;
        for (int k = 0; k < N_FREQ_BINS; ++k) {
            s += (double)mag[k] * mag[k] * g_mel[m][k];
        }
        mel_power[m] = logf((float)s);
    }
    for (int c = 0; c < N_MFCC; ++c) {
        double s = 0.0;
        for (int m = 0; m < N_MELS; ++m) s += g_dct[c][m] * mel_power[m];
        mfcc_out[c] = (float)s;
    }
}

// ─── Public entry point ──────────────────────────────────────────────

bool compute_features(const int16_t* pcm, int n, int sr, float* out) {
    if (!pcm || !out || n < FRAME_LEN || sr <= 0) return false;
    if (sr != g_sr) return false;     // init() not called or sr mismatch

    // Convert int16 → float32 in [-1, 1] (matches Python convention).
    // We store the full signal so we can compute kurtosis/skewness/crest
    // factor on it — exactly like the reference does. Stack-allocated
    // for the typical 5 s buffer (16000 × 4 = 64 KB); fall back to
    // heap for unusually large inputs.
    float* sig;
    bool   sig_on_heap = false;
    static thread_local float sig_stack[16384];
    if (n <= (int)(sizeof(sig_stack) / sizeof(sig_stack[0]))) {
        sig = sig_stack;
    } else {
        sig = (float*)malloc((size_t)n * sizeof(float));
        if (!sig) return false;
        sig_on_heap = true;
    }
    const float inv_int16 = 1.0f / 32768.0f;
    for (int i = 0; i < n; ++i) sig[i] = pcm[i] * inv_int16;

    // ── Whole-signal stats ──────────────────────────────────────────
    double mu_acc = 0.0;
    for (int i = 0; i < n; ++i) mu_acc += sig[i];
    float mu = (float)(mu_acc / n);

    double var_acc = 0.0;
    for (int i = 0; i < n; ++i) {
        float d = sig[i] - mu;
        var_acc += (double)d * d;
    }
    float sigma = sqrtf((float)(var_acc / n)) + 1e-9f;

    double k_acc = 0.0, s_acc = 0.0;
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        float z = (sig[i] - mu) / sigma;
        s_acc += (double)z * z * z;
        k_acc += (double)z * z * z * z;
        float a = fabsf(sig[i]);
        if (a > peak) peak = a;
    }
    float kurt = (float)(k_acc / n) - 3.0f;
    float skw  = (float)(s_acc / n);

    // ── Frame loop: time-domain stats + spectra ─────────────────────
    int n_frames = 1 + (n - FRAME_LEN) / HOP;
    if (n_frames < 1) n_frames = 1;

    double rms_sum = 0.0, rms_sq_sum = 0.0;
    double zcr_sum = 0.0, zcr_sq_sum = 0.0;
    double cen_sum = 0.0, cen_sq_sum = 0.0;
    double rol_sum = 0.0, rol_sq_sum = 0.0;
    double flt_sum = 0.0, flt_sq_sum = 0.0;
    double mfcc_sum   [N_MFCC] = {0};
    double mfcc_sq_sum[N_MFCC] = {0};

    float frame[FRAME_LEN];
    float mag  [N_FREQ_BINS];
    float mfcc [N_MFCC];

    for (int f = 0; f < n_frames; ++f) {
        int start = f * HOP;
        // Zero-pad short tail (final frame may exceed n)
        for (int i = 0; i < FRAME_LEN; ++i) {
            int idx = start + i;
            frame[i] = (idx < n) ? sig[idx] : 0.0f;
        }

        float r = frame_rms(frame, FRAME_LEN);
        float z = frame_zcr(frame, FRAME_LEN);
        rms_sum    += r; rms_sq_sum    += (double)r * r;
        zcr_sum    += z; zcr_sq_sum    += (double)z * z;

        frame_magnitude(frame, mag);

        float c = spectral_centroid(mag);
        float ro = spectral_rolloff(mag);
        float fl = spectral_flatness(mag);
        cen_sum += c;  cen_sq_sum += (double)c * c;
        rol_sum += ro; rol_sq_sum += (double)ro * ro;
        flt_sum += fl; flt_sq_sum += (double)fl * fl;

        frame_mfccs(mag, mfcc);
        for (int j = 0; j < N_MFCC; ++j) {
            mfcc_sum   [j] += mfcc[j];
            mfcc_sq_sum[j] += (double)mfcc[j] * mfcc[j];
        }
    }

    auto mean_std = [&](double s, double sq, double& mean, double& std_out) {
        mean = s / n_frames;
        double var = sq / n_frames - mean * mean;
        if (var < 0.0) var = 0.0;
        std_out = sqrt(var);
    };

    double rms_mean, rms_std, zcr_mean, zcr_std;
    double cen_mean, cen_std, rol_mean, rol_std, flt_mean, flt_std;
    mean_std(rms_sum, rms_sq_sum, rms_mean, rms_std);
    mean_std(zcr_sum, zcr_sq_sum, zcr_mean, zcr_std);
    mean_std(cen_sum, cen_sq_sum, cen_mean, cen_std);
    mean_std(rol_sum, rol_sq_sum, rol_mean, rol_std);
    mean_std(flt_sum, flt_sq_sum, flt_mean, flt_std);

    float crest = peak / (float)rms_mean;

    // ── Pack the 39-d output (layout matches omni/eep/features.py) ──
    out[ 0] = (float)rms_mean;  out[ 1] = (float)rms_std;
    out[ 2] = (float)zcr_mean;  out[ 3] = (float)zcr_std;
    out[ 4] = kurt;
    out[ 5] = skw;
    out[ 6] = crest;
    out[ 7] = (float)cen_mean;  out[ 8] = (float)cen_std;
    out[ 9] = (float)rol_mean;  out[10] = (float)rol_std;
    out[11] = (float)flt_mean;  out[12] = (float)flt_std;

    for (int j = 0; j < N_MFCC; ++j) {
        double m  = mfcc_sum[j] / n_frames;
        double v  = mfcc_sq_sum[j] / n_frames - m * m;
        if (v < 0.0) v = 0.0;
        out[13 + 2 * j    ] = (float)m;
        out[13 + 2 * j + 1] = (float)sqrt(v);
    }

    if (sig_on_heap) free(sig);
    return true;
}

}  // namespace omni
