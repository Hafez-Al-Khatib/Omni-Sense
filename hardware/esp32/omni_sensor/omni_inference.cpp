/**
 * omni_inference.cpp — TFLite-Micro autoencoder OOD wrapper (impl)
 *
 * Build dependencies (PlatformIO lib_deps in platformio.ini)
 *   - tensorflow/tflite-micro    (the on-device runtime)
 *
 * Memory budget (ESP32-S3, int8 39->16->8->16->39 model)
 *   model bytes (.tflite int8):     ~3-5 KB   (in flash, via const array)
 *   tensor arena:                   8 KB      (allocated once, in SRAM)
 *   per-call stack:                 <1 KB
 *
 * Falls back gracefully when the model is a placeholder (training was
 * skipped). In that case inference_begin() returns false and the
 * firmware reverts to the raw-PCM publish path.
 */

#include "omni_inference.h"
#include "omni_features.h"      // for FEATURE_DIM
#include "edge_model_data.h"    // omni_edge_model_data, threshold, mean/std

#include <math.h>
#include <string.h>

#if !defined(OMNI_EDGE_MODEL_PLACEHOLDER) && __has_include(<tensorflow/lite/micro/micro_interpreter.h>)
  #define OMNI_TFLM_AVAILABLE 1
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
  #include <tensorflow/lite/schema/schema_generated.h>
#else
  #define OMNI_TFLM_AVAILABLE 0
#endif

namespace omni {

#if OMNI_TFLM_AVAILABLE
constexpr int kArenaBytes = 8 * 1024;
alignas(16) static uint8_t g_arena[kArenaBytes];

static const tflite::Model*           g_model       = nullptr;
static tflite::MicroInterpreter*      g_interpreter = nullptr;
static TfLiteTensor*                  g_input       = nullptr;
static TfLiteTensor*                  g_output      = nullptr;

// Op resolver — only the ops the autoencoder actually uses, to keep
// flash size minimal. If you change the model architecture, add the
// corresponding ops here.
using OmniResolver = tflite::MicroMutableOpResolver<3>;
static OmniResolver* g_resolver = nullptr;
static uint8_t       g_resolver_storage[sizeof(OmniResolver)];
#endif

static bool        g_ready  = false;
static const char* g_status = "uninitialised";

bool inference_is_ready() { return g_ready; }
const char* inference_status() { return g_status; }

#if !OMNI_TFLM_AVAILABLE
// ── No TFLite-Micro available (placeholder model OR header missing) ────
bool inference_begin() {
    g_status = "edge model is a placeholder; on-device OOD disabled";
    g_ready  = false;
    return false;
}
InferenceResult inference_score(const float* /*feat39*/) {
    return {false, false, 0.0f, 0.0f};
}
#else
// ── Real TFLite-Micro path ─────────────────────────────────────────────
bool inference_begin() {
    g_model = tflite::GetModel(omni_edge_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        g_status = "tflite schema version mismatch";
        return false;
    }

    g_resolver = new (g_resolver_storage) OmniResolver();
    g_resolver->AddFullyConnected();
    g_resolver->AddRelu();
    g_resolver->AddQuantize();   // for input/output type conversion

    static tflite::MicroInterpreter interp(g_model, *g_resolver,
                                           g_arena, kArenaBytes);
    g_interpreter = &interp;
    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        g_status = "AllocateTensors failed (arena too small?)";
        return false;
    }

    g_input  = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    if (g_input->dims->data[1] != FEATURE_DIM ||
        g_output->dims->data[1] != FEATURE_DIM) {
        g_status = "model I/O shape != FEATURE_DIM";
        return false;
    }

    g_ready  = true;
    g_status = "tflite-micro ok";
    return true;
}

InferenceResult inference_score(const float feat39[39]) {
    InferenceResult r{false, false, 0.0f, omni_edge_threshold};
    if (!g_ready || !feat39) return r;

    // Standardise -> int8 quantise into input tensor
    const float in_scale     = g_input->params.scale;
    const int   in_zero_pt   = g_input->params.zero_point;
    const float out_scale    = g_output->params.scale;
    const int   out_zero_pt  = g_output->params.zero_point;

    float standardised[FEATURE_DIM];
    for (int i = 0; i < FEATURE_DIM; ++i) {
        standardised[i] =
            (feat39[i] - omni_edge_feature_mean[i]) / omni_edge_feature_std[i];
        int q = (int)lroundf(standardised[i] / in_scale) + in_zero_pt;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        g_input->data.int8[i] = (int8_t)q;
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        g_status = "Invoke failed";
        return r;
    }

    // Dequantise output and accumulate MSE in standardised space
    double sse = 0.0;
    for (int i = 0; i < FEATURE_DIM; ++i) {
        float recon = (g_output->data.int8[i] - out_zero_pt) * out_scale;
        float diff  = standardised[i] - recon;
        sse += (double)diff * diff;
    }
    r.mse        = (float)(sse / FEATURE_DIM);
    r.is_anomaly = r.mse > omni_edge_threshold;
    r.ok         = true;
    return r;
}
#endif  // OMNI_TFLM_AVAILABLE

}  // namespace omni
