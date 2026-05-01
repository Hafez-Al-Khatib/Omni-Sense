/**
 * omni_inference.h — TFLite-Micro autoencoder OOD wrapper
 * ========================================================
 * Runs the tiny feature-space autoencoder (39 -> 16 -> 8 -> 16 -> 39)
 * exported by scripts/train_edge_autoencoder.py and quantised to int8.
 *
 * Decision contract
 * -----------------
 * On each call, the wrapper:
 *   1. Standardises the 39-d feature vector with the per-feature
 *      mean/std baked into edge_model_data.h.
 *   2. Quantises to int8, runs the autoencoder, dequantises the
 *      reconstruction.
 *   3. Computes the reconstruction MSE (in standardised space).
 *   4. Returns whether the MSE exceeds the calibrated threshold from
 *      training (also in edge_model_data.h).
 *
 * Placeholder fallback
 * --------------------
 * edge_model_data.h defines OMNI_EDGE_MODEL_PLACEHOLDER when the model
 * file at training time was not a real TFLite blob. In that case
 * begin() returns false and the firmware should publish raw PCM
 * (the existing behaviour) — there is no on-device OOD gate.
 */

#pragma once
#include <stddef.h>
#include <stdint.h>

namespace omni {

struct InferenceResult {
    bool   ok;            // true if inference ran (model loaded, no error)
    bool   is_anomaly;    // true if reconstruction MSE > threshold
    float  mse;           // reconstruction MSE in standardised space
    float  threshold;     // copy of the threshold used (for telemetry)
};

// Initialise the TFLite-Micro interpreter and arena. Must be called
// once at startup. Returns true on success; false if the embedded
// model is a placeholder or the interpreter rejected it.
bool inference_begin();

// True after a successful inference_begin().
bool inference_is_ready();

// Run one OOD check on a 39-d feature vector. Safe to call before
// inference_begin() — returns {ok=false, ...} in that case.
InferenceResult inference_score(const float feat39[39]);

// One-line diagnostic string for telemetry (e.g. "tflite-micro ok / arena=8192").
const char* inference_status();

}  // namespace omni
