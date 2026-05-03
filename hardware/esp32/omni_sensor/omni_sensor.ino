/**
 * Omni-Sense ESP32 Edge Firmware v3
 * ==================================
 * Hardware: ESP32-S3 DevKitC + ADXL345 MEMS accelerometer (SPI)
 *
 * WHY ACCELEROMETER, NOT MICROPHONE
 * ----------------------------------
 * Water pipe leak detection relies on structure-borne acoustic emission —
 * stress waves that travel through the pipe wall at 1000–5100 m/s depending
 * on material.  An airborne MEMS microphone (e.g. INMP441) picks up
 * ambient noise (traffic, HVAC, voices) that is completely absent from the
 * pipe wall and will saturate the Isolation Forest OOD gate.
 *
 * The ADXL345 is clamped or epoxied directly to the pipe exterior.  It
 * measures only the acceleration of the pipe wall itself, giving a
 * physics-isolated channel that contains exactly the leak signature:
 *   • Turbulent flow noise   (50–500 Hz)
 *   • Orifice leak whistle   (500–1600 Hz)
 *   • Crack propagation tick (transient 100–1600 Hz)
 *
 * ADXL345 KEY SPECS (SPI mode)
 * -----------------------------
 * ODR up to 3200 Hz (BW_RATE register 0x0F) — THIS IS THE HARDWARE MAXIMUM.
 * We are honest: the firmware samples at 3200 Hz, NOT 16 kHz.
 * ±2 g / ±4 g / ±8 g / ±16 g full-scale range
 * 10-bit or 13-bit resolution
 * 32-sample FIFO with watermark interrupt
 * SPI clock up to 5 MHz (single-byte) / 10 MHz (multi-byte)
 *
 * SAMPLING STRATEGY
 * -----------------
 * ODR  = 3200 Hz   → Nyquist = 1600 Hz   (covers full leak band 50–1600 Hz)
 * Frame = 3200 samples = 1.000 s
 * Payload: Z-axis int16 × 3200 = 6400 bytes → ~8537 bytes base64
 *
 * At 3200 Hz the FIFO fills every 32/3200 = 10 ms.  We use polling with
 * a 5 ms sleep — no interrupt routing needed, saving GPIO.
 *
 * SPI WIRING (adjust to match your PCB)
 * ----------------------------------------
 * ADXL345  →  ESP32
 * VCC      →  3.3 V
 * GND      →  GND
 * CS       →  GPIO 5
 * SDO/MISO →  GPIO 16
 * SDA/MOSI →  GPIO 17
 * SCL/SCK  →  GPIO 18
 * INT1     →  not connected (polling mode)
 * INT2     →  not connected
 *
 * MOUNTING
 * --------
 * Mount the breakout board flat on the pipe surface with the Z axis (top face)
 * perpendicular to the pipe wall.  Use thermal epoxy or a hose clamp adapter.
 * Ensure no air gap — acoustic coupling efficiency drops sharply otherwise.
 *
 * PlatformIO build: see ../platformio.ini
 * Arduino IDE: Board = "ESP32 Dev Module", Partition = "Huge APP"
 */

// ─────────────────────────── Configuration ────────────────────────────────────
// Edit hardware/esp32/config.h (not tracked by git) for local credentials.
#include "config.h"

#ifndef WIFI_SSID
  #error "WIFI_SSID not defined — copy config.h.example to config.h and edit"
#endif

// ADXL345 SPI pins
#define ADXL_CS_PIN       5
#define ADXL_MISO_PIN     16
#define ADXL_MOSI_PIN     17
#define ADXL_SCK_PIN      18
#define ADXL_SPI_FREQ     4000000   // 4 MHz — well within 5 MHz single-byte limit

// ADXL345 register addresses
#define ADXL_REG_DEVID        0x00
#define ADXL_REG_BW_RATE      0x2C   // ODR / bandwidth register
#define ADXL_REG_POWER_CTL   0x2D   // Power mode register
#define ADXL_REG_DATA_FORMAT  0x31   // Range / resolution register
#define ADXL_REG_FIFO_CTL    0x38   // FIFO mode register
#define ADXL_REG_FIFO_STATUS 0x39   // FIFO entries available
#define ADXL_REG_DATAX0      0x32   // Start of 6-byte XYZ burst

// ADXL345 ODR codes (BW_RATE register low nibble)
#define ADXL_ODR_3200     0x0F   // 3200 Hz — maximum, captures full leak band
#define ADXL_ODR_1600     0x0E
#define ADXL_ODR_800      0x0D

// Full-scale range (DATA_FORMAT register bits 1:0)
#define ADXL_RANGE_2G     0x00   // ±2 g — highest sensitivity, ideal for wall vibration
#define ADXL_RANGE_4G     0x01
#define ADXL_RANGE_8G     0x02
#define ADXL_RANGE_16G    0x03

// SPI read flag (bit 7 set) and multi-byte flag (bit 6 set)
#define ADXL_SPI_READ     0x80
#define ADXL_SPI_MB       0x40

// Sampling parameters
#define SAMPLE_RATE       3200    // Hz (matches ADXL_ODR_3200)
#define FRAME_SAMPLES     3200    // 1.000 s × 3200 Hz

// VAD: RMS of Z-axis acceleration below this → drop frame (pipe fully at rest)
// 0.002 g ≈ 20 mg — well above thermal noise floor (~0.5 mg RMS of ADXL345)
#define VAD_THRESHOLD_G   0.002f

// Publish telemetry every N frames
#define TELEMETRY_PERIOD  30

// Deep sleep between captures (0 = continuous)
#define DEEP_SLEEP_US     0ULL

// Battery ADC (GPIO 34 through voltage divider, -1 to disable)
#define BATTERY_ADC_PIN   34
#define BATTERY_ADC_REF   3300
#define BATTERY_DIVIDER   2.0f

// ─────────────────────────── Includes ─────────────────────────────────────────

#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <PubSubClient.h>
#include <ArduinoOTA.h>
#include <math.h>
#include <string.h>
#include <time.h>

// On-device 39-d DSP feature extractor + TFLite-Micro autoencoder OOD.
// Compiled in only when -DOMNI_PUBLISH_FEATURES=1 (ESP32-S3 target in
// platformio.ini). The plain ESP32 target keeps the original raw-PCM
// publish path verbatim, so existing call sites are unaffected.
#if defined(OMNI_PUBLISH_FEATURES) && OMNI_PUBLISH_FEATURES
  #include "omni_features.h"
  #include "omni_inference.h"
#endif

// ─────────────────────────── TLS Certificates ─────────────────────────────────
// Generated by scripts/gen_esp32_certs.sh — DO NOT EDIT MANUALLY.
// #include "certs/ca_cert.h"
// #include "certs/client_cert.h"
// #include "certs/client_key.h"

// ─────────────────────────── Globals ──────────────────────────────────────────

static SPIClass          _spi(HSPI);
static WiFiClient  _tlsClient;
static PubSubClient      _mqttClient(_tlsClient);

// Z-axis sample buffer (int16, raw ADXL counts)
// At ±2 g range and 10-bit resolution: 1 LSB = 4 mg, range ±512 counts
static int16_t  _zBuf[FRAME_SAMPLES];

static uint32_t _frameCount    = 0;
static uint32_t _dropCount     = 0;
static uint32_t _publishCount  = 0;

static char _acousticTopic[128];
static char _telemetryTopic[128];
static char _sensorId[24];

static void initSensorId() {
  if (strcmp(SENSOR_ID, "AUTO") == 0) {
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(_sensorId, sizeof(_sensorId), "esp32-%02X%02X", mac[4], mac[5]);
  } else {
    strncpy(_sensorId, SENSOR_ID, sizeof(_sensorId) - 1);
    _sensorId[sizeof(_sensorId) - 1] = '\0';
  }
}

#if defined(OMNI_PUBLISH_FEATURES) && OMNI_PUBLISH_FEATURES
// 5-second rolling window at SAMPLE_RATE for on-device feature extraction.
// 5 s × 3200 Hz × 2 bytes = 32 000 bytes. Lives in PSRAM on ESP32-S3
// when OMNI_PUBLISH_FEATURES is on (see -DBOARD_HAS_PSRAM in platformio.ini).
constexpr int FEAT_WINDOW_S       = 5;
constexpr int FEAT_WINDOW_SAMPLES = FEAT_WINDOW_S * SAMPLE_RATE;   // 16 000
static int16_t* _featBuf          = nullptr;
static int      _featBufFill      = 0;          // samples accumulated so far
static uint32_t _oodDropCount     = 0;          // frames dropped by on-device OOD
#endif

// ─────────────────────────── SPI helpers ──────────────────────────────────────

static inline void adxl_cs_low()  { digitalWrite(ADXL_CS_PIN, LOW);  }
static inline void adxl_cs_high() { digitalWrite(ADXL_CS_PIN, HIGH); }

static void adxl_write(uint8_t reg, uint8_t val) {
  adxl_cs_low();
  _spi.transfer(reg & 0x3F);   // write: bit7=0, bit6=0
  _spi.transfer(val);
  adxl_cs_high();
}

static uint8_t adxl_read(uint8_t reg) {
  adxl_cs_low();
  _spi.transfer(ADXL_SPI_READ | (reg & 0x3F));
  uint8_t val = _spi.transfer(0x00);
  adxl_cs_high();
  return val;
}

/**
 * Burst-read n_samples × 6 bytes (XYZ × 2) from FIFO starting at DATAX0.
 * Discards X and Y; stores only Z in dst.
 * Returns actual number of samples read.
 */
static int adxl_read_fifo(int16_t *dst, int n_samples) {
  // Check how many entries are in FIFO (bits 5:0 of FIFO_STATUS)
  int available = adxl_read(ADXL_REG_FIFO_STATUS) & 0x3F;
  int to_read   = min(n_samples, available);

  if (to_read == 0) return 0;

  adxl_cs_low();
  // Multi-byte read from DATAX0 (0x32): set bits 7 and 6
  _spi.transfer(ADXL_SPI_READ | ADXL_SPI_MB | ADXL_REG_DATAX0);
  for (int i = 0; i < to_read; i++) {
    int16_t x_raw = (int16_t)(_spi.transfer(0) | ((uint16_t)_spi.transfer(0) << 8));
    int16_t y_raw = (int16_t)(_spi.transfer(0) | ((uint16_t)_spi.transfer(0) << 8));
    int16_t z_raw = (int16_t)(_spi.transfer(0) | ((uint16_t)_spi.transfer(0) << 8));
    (void)x_raw; (void)y_raw;   // discard X, Y
    dst[i] = z_raw;
  }
  adxl_cs_high();
  return to_read;
}

// ─────────────────────────── ADXL345 init ─────────────────────────────────────

static bool adxl_init() {
  _spi.begin(ADXL_SCK_PIN, ADXL_MISO_PIN, ADXL_MOSI_PIN, ADXL_CS_PIN);
  _spi.setFrequency(ADXL_SPI_FREQ);
  _spi.setDataMode(SPI_MODE3);   // ADXL345 requires CPOL=1, CPHA=1

  pinMode(ADXL_CS_PIN, OUTPUT);
  adxl_cs_high();
  delay(5);

  // Verify device ID
  uint8_t devid = adxl_read(ADXL_REG_DEVID);
  if (devid != 0xE5) {
    Serial.printf("[ADXL] ERROR: unexpected DEVID=0x%02X (expected 0xE5)\n", devid);
    Serial.println("[ADXL] Check wiring: CS=" + String(ADXL_CS_PIN) +
                   " MISO=" + String(ADXL_MISO_PIN) +
                   " MOSI=" + String(ADXL_MOSI_PIN) +
                   " SCK="  + String(ADXL_SCK_PIN));
    return false;
  }
  Serial.printf("[ADXL] DEVID=0x%02X OK\n", devid);

  // Standby mode before configuration
  adxl_write(ADXL_REG_POWER_CTL, 0x00);

  // Set ODR to 3200 Hz, low-power = 0 (normal power for max noise performance)
  adxl_write(ADXL_REG_BW_RATE, ADXL_ODR_3200);

  // Set range ±2 g, full resolution mode (bit 3 = FULL_RES)
  // In FULL_RES mode the scale factor is always 3.9 mg/LSB regardless of range
  adxl_write(ADXL_REG_DATA_FORMAT, 0x08 | ADXL_RANGE_2G);

  // FIFO stream mode, watermark at 16 samples (half FIFO)
  // Bits 7:6 = 10 (stream), bits 4:0 = 10000 (watermark=16)
  adxl_write(ADXL_REG_FIFO_CTL, 0x90);

  // Enable measurement
  adxl_write(ADXL_REG_POWER_CTL, 0x08);

  delay(10);   // Allow measurement mode to stabilize
  Serial.printf("[ADXL] initialized: ODR=3200 Hz range=±2g FIFO=stream\n");
  return true;
}

// ─────────────────────────── Frame capture ────────────────────────────────────

/**
 * Accumulate FRAME_SAMPLES Z-axis readings into _zBuf by polling the FIFO.
 * At 3200 Hz ODR, the 32-entry FIFO fills every 10 ms.
 * We poll every 5 ms, draining ~16 samples per poll iteration.
 * Total frame duration: FRAME_SAMPLES / 3200 = 1.000 s.
 */
static bool adxl_capture_frame() {
  int collected = 0;
  const unsigned long deadline = millis() + 2000;   // 2 s hard timeout

  while (collected < FRAME_SAMPLES) {
    int got = adxl_read_fifo(_zBuf + collected, FRAME_SAMPLES - collected);
    collected += got;
    if (collected < FRAME_SAMPLES) {
      delay(5);   // ~16 samples will arrive before next poll
    }
    if (millis() > deadline) {
      Serial.printf("[ADXL] capture timeout: got %d/%d samples\n",
                    collected, FRAME_SAMPLES);
      return false;
    }
  }
  return true;
}

// ─────────────────────────── DC removal ──────────────────────────────────────

/**
 * Remove DC offset (gravity component) from Z-axis buffer.
 * Without this, static gravity (≈ ±1 g depending on mounting angle) would
 * dominate the RMS computation and mask the small leak vibrations.
 */
static void remove_dc_offset(int16_t *buf, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; i++) sum += buf[i];
  int16_t mean = (int16_t)(sum / n);
  for (int i = 0; i < n; i++) buf[i] -= mean;
}

// ─────────────────────────── VAD ──────────────────────────────────────────────

/**
 * Compute RMS of Z-axis buffer in physical units (g).
 * ADXL345 in FULL_RES mode: scale = 3.9 mg/LSB = 0.0039 g/LSB
 */
static float compute_rms_g(const int16_t *buf, int n) {
  const float scale = 0.0039f;   // g per LSB in FULL_RES mode
  double sum_sq = 0.0;
  for (int i = 0; i < n; i++) {
    float v = buf[i] * scale;
    sum_sq += (double)v * (double)v;
  }
  return (float)sqrt(sum_sq / n);
}

static float compute_snr_db(float rms_g) {
  if (rms_g <= 0.0f) return 0.0f;
  const float noise_floor_g = 0.0005f;   // ~0.5 mg ADXL345 noise floor
  float ratio = rms_g / noise_floor_g;
  if (ratio < 1.0f) ratio = 1.0f;
  return 20.0f * log10f(ratio);
}

// ─────────────────────────── Base64 ───────────────────────────────────────────

static const char B64_CHARS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t b64_encode(const uint8_t *src, size_t src_len,
                          char *dst, size_t dst_max) {
  size_t out = 0;
  for (size_t i = 0; i < src_len && out + 4 < dst_max; i += 3) {
    uint32_t val = (uint32_t)src[i] << 16;
    if (i + 1 < src_len) val |= (uint32_t)src[i+1] << 8;
    if (i + 2 < src_len) val |= (uint32_t)src[i+2];
    dst[out++] = B64_CHARS[(val >> 18) & 0x3F];
    dst[out++] = B64_CHARS[(val >> 12) & 0x3F];
    dst[out++] = (i + 1 < src_len) ? B64_CHARS[(val >> 6) & 0x3F] : '=';
    dst[out++] = (i + 2 < src_len) ? B64_CHARS[val & 0x3F]        : '=';
  }
  dst[out] = '\0';
  return out;
}

// ─────────────────────────── Battery ──────────────────────────────────────────

static float read_battery_pct() {
#if BATTERY_ADC_PIN >= 0
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) sum += analogRead(BATTERY_ADC_PIN);
  float adc_mv  = (sum / 16.0f) / 4095.0f * BATTERY_ADC_REF;
  float batt_mv = adc_mv * BATTERY_DIVIDER;
  float pct = (batt_mv - 3000.0f) / (4200.0f - 3000.0f) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
#else
  return 100.0f;
#endif
}

// ─────────────────────────── WiFi ─────────────────────────────────────────────

static void wifi_connect() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("[WiFi] connecting to %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(500); Serial.print("."); retries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] IP=%s RSSI=%d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    configTime(0, 0, "pool.ntp.org", "time.google.com");
    time_t now = 0; int tries = 0;
    while (now < 1700000000L && tries < 20) { delay(500); time(&now); tries++; }
  } else {
    Serial.println("\n[WiFi] failed");
  }
}

// ─────────────────────────── MQTT ─────────────────────────────────────────────

  static bool mqtt_connect() {
    if (_mqttClient.connected()) return true;
    // Plaintext MQTT — no TLS certs needed
    _mqttClient.setServer(MQTT_HOST, MQTT_PORT);
    _mqttClient.setBufferSize(16384);
    _mqttClient.setKeepAlive(60);
    int retries = 0;
    while (!_mqttClient.connected() && retries < 5) {
      if (_mqttClient.connect(_sensorId)) { Serial.println("[MQTT] connected"); return true; }
      Serial.printf("[MQTT] failed state=%d retry %d\n", _mqttClient.state(), retries+1);
      delay(2000 * (1 << retries)); retries++;
    }
    return false;
  }

static bool mqtt_publish_raw(const char *topic, const char *payload) {
  if (!_mqttClient.connected() && !mqtt_connect()) return false;
  size_t len = strlen(payload);
  bool ok = _mqttClient.publish(topic, (const uint8_t*)payload, len, false);
  if (!ok) Serial.printf("[MQTT] publish failed len=%u\n", len);
  return ok;
}

// ─────────────────────────── Timestamp ────────────────────────────────────────

static void iso8601_now(char *buf, size_t buf_len) {
  time_t now; time(&now);
  struct tm *t = gmtime(&now);
  strftime(buf, buf_len, "%Y-%m-%dT%H:%M:%SZ", t);
}

// ─────────────────────────── Publish helpers ──────────────────────────────────

// b64 for 3200 × 2 bytes = 6400 bytes → ceil(6400/3)*4 = 8536 chars + NUL
static char _b64buf[8600];
static char _jsonbuf[9800];

static void publish_acoustic(float rms_g, float snr_db) {
  float vad_conf = (rms_g < VAD_THRESHOLD_G * 10.0f)
                 ? (rms_g / (VAD_THRESHOLD_G * 10.0f)) : 1.0f;

  b64_encode((const uint8_t*)_zBuf, FRAME_SAMPLES * sizeof(int16_t),
             _b64buf, sizeof(_b64buf));

  if (mqtt_publish_raw(_acousticTopic, _b64buf)) {
    _publishCount++;
    Serial.printf("[ACCEL] frame=%lu rms=%.4fg snr=%.1fdB vad=%.3f\n",
                  _frameCount, rms_g, snr_db, vad_conf);
  }
}

#if defined(OMNI_PUBLISH_FEATURES) && OMNI_PUBLISH_FEATURES
// ─────────────────────────── Features-mode publish ─────────────────────────────
//
// On-device path (~40x bandwidth reduction vs raw PCM):
//   1. Append the just-captured 1-second frame to a 5-second rolling buffer.
//   2. When the buffer is full, run omni::compute_features() — bit-for-bit
//      identical to omni/eep/features.py at sr=3200, so the model trained
//      on data/synthesized/eep_features_3200hz.parquet generalises with
//      zero domain shift.
//   3. Run the int8 TFLite-Micro autoencoder OOD gate. If the
//      reconstruction MSE is below the calibrated threshold, the frame
//      is normal background — drop it (saves uplink bandwidth and
//      cloud-side classifier load).
//   4. Otherwise publish a small JSON with the 39 floats + OOD score.
//
// Falls back gracefully when the embedded model is a placeholder
// (training was skipped): inference_score().ok == false → publish
// every frame's features but do not gate on OOD.
static void publish_features(float rms_g, float snr_db) {
  if (!_featBuf) return;       // setup() failed to allocate

  // Append 1-second frame to the 5-second window
  int copy = FRAME_SAMPLES;
  if (_featBufFill + copy > FEAT_WINDOW_SAMPLES)
    copy = FEAT_WINDOW_SAMPLES - _featBufFill;
  memcpy(_featBuf + _featBufFill, _zBuf, copy * sizeof(int16_t));
  _featBufFill += copy;

  if (_featBufFill < FEAT_WINDOW_SAMPLES) {
    // Still accumulating — log progress every second so the operator
    // can see the device is alive even before the first publish.
    Serial.printf("[FEAT] accumulating %d/%d samples\n",
                  _featBufFill, FEAT_WINDOW_SAMPLES);
    return;
  }

  // Buffer full — compute features, gate on OOD, publish
  float feat[omni::FEATURE_DIM];
  if (!omni::compute_features(_featBuf, FEAT_WINDOW_SAMPLES,
                              SAMPLE_RATE, feat)) {
    Serial.println("[FEAT] compute_features failed");
    _featBufFill = 0;
    return;
  }

  omni::InferenceResult inf = omni::inference_score(feat);

  // OOD gate: if model is loaded and frame is in-distribution, drop it
  if (inf.ok && !inf.is_anomaly) {
    _oodDropCount++;
    Serial.printf("[OOD] frame=%lu mse=%.5f<thr=%.5f — dropped\n",
                  _frameCount, inf.mse, inf.threshold);
    _featBufFill = 0;
    return;
  }

  // Build feature JSON. ~200 B vs ~8.5 KB for the raw-PCM path.
  char ts[32]; iso8601_now(ts, sizeof(ts));
  int n = snprintf(_jsonbuf, sizeof(_jsonbuf),
    "{\"schema_version\":\"2-features\","
    "\"sensor_id\":\"%s\",\"site_id\":\"%s\","
    "\"captured_at\":\"%s\","
    "\"sample_rate_hz\":%d,\"window_s\":%d,"
    "\"feature_extractor\":\"omni_features_v1_3200hz\","
    "\"edge_snr_db\":%.2f,"
    "\"edge_ood_score\":%.5f,\"edge_ood_threshold\":%.5f,"
    "\"edge_ood_status\":\"%s\","
    "\"firmware_version\":\"%s\","
    "\"features\":[",
    _sensorId, SITE_ID, ts, SAMPLE_RATE, FEAT_WINDOW_S,
    snr_db, inf.mse, inf.threshold,
    inf.ok ? "anomaly" : "model_unavailable",
    FIRMWARE_VERSION);

  for (int i = 0; i < omni::FEATURE_DIM && n < (int)sizeof(_jsonbuf) - 16; i++) {
    n += snprintf(_jsonbuf + n, sizeof(_jsonbuf) - n,
                  "%s%.6f", (i ? "," : ""), feat[i]);
  }
  if (n < (int)sizeof(_jsonbuf) - 2) {
    n += snprintf(_jsonbuf + n, sizeof(_jsonbuf) - n, "]}");
  }

  if (mqtt_publish_raw(_acousticTopic, _jsonbuf)) {
    _publishCount++;
    Serial.printf("[FEAT] frame=%lu published 39d (rms=%.4fg snr=%.1fdB ood=%.5f)\n",
                  _frameCount, rms_g, snr_db, inf.mse);
  }

  _featBufFill = 0;
}
#endif

static void publish_telemetry() {
  char ts[32]; iso8601_now(ts, sizeof(ts));
  float batt    = read_battery_pct();
  uint32_t heap = ESP.getFreeHeap();
  int64_t uptime = esp_timer_get_time() / 1000000LL;
  float temp_c = 0.0f;
#ifdef CONFIG_IDF_TARGET_ESP32
  extern float temperatureRead();
  temp_c = temperatureRead();
#endif
  snprintf(_jsonbuf, sizeof(_jsonbuf),
    "{\"sensor_id\":\"%s\",\"captured_at\":\"%s\","
    "\"battery_pct\":%.1f,\"temperature_c\":%.1f,"
    "\"disk_free_mb\":%.1f,\"rtc_drift_ms\":0,"
    "\"uptime_s\":%lld,\"firmware_version\":\"%s\","
    "\"wifi_rssi\":%d,\"frames_published\":%lu,\"frames_dropped\":%lu}",
    _sensorId, ts, batt, temp_c,
    (float)heap/1024.0f, uptime, FIRMWARE_VERSION,
    WiFi.RSSI(), _publishCount, _dropCount);
  mqtt_publish_raw(_telemetryTopic, _jsonbuf);
}

// ─────────────────────────── OTA ──────────────────────────────────────────────

static void ota_init() {
  ArduinoOTA.setHostname(_sensorId);
  ArduinoOTA.onStart([]() {
    Serial.println("[OTA] starting — stopping SPI");
    _spi.end();
  });
  ArduinoOTA.onEnd([]() { Serial.println("[OTA] done"); });
  ArduinoOTA.onError([](ota_error_t e) { Serial.printf("[OTA] error %u\n", e); });
  ArduinoOTA.begin();
  Serial.println("[OTA] ready");
}

// ─────────────────────────── setup / loop ─────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);
  initSensorId();
  Serial.println("\n=== Omni-Sense ESP32 v3 — Structure-Borne Accelerometer ===");
  Serial.printf("  Sensor  : %s\n", _sensorId);
  Serial.printf("  Site    : %s\n", SITE_ID);
  Serial.printf("  FW      : %s\n", FIRMWARE_VERSION);
  Serial.printf("  Sensor  : ADXL345 SPI @ %d Hz (Z-axis) — honest structure-borne vibration sampling\n", SAMPLE_RATE);
  Serial.printf("  Frame   : %d samples = %.3f s\n", FRAME_SAMPLES,
                (float)FRAME_SAMPLES / SAMPLE_RATE);
  Serial.printf("  Nyquist : %d Hz (leak detection band 50–1600 Hz)\n", SAMPLE_RATE / 2);
  snprintf(_acousticTopic,  sizeof(_acousticTopic),
           "sensors/%s/accel",  _sensorId);
  snprintf(_telemetryTopic, sizeof(_telemetryTopic),
           "omni/sensor/%s/%s/telemetry", SITE_ID, _sensorId);

#if BATTERY_ADC_PIN >= 0
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
#endif

  wifi_connect();
  ota_init();
  mqtt_connect();

  if (!adxl_init()) {
    Serial.println("[SETUP] ADXL345 init FAILED — halting");
    while (true) delay(1000);
  }

#if defined(OMNI_PUBLISH_FEATURES) && OMNI_PUBLISH_FEATURES
  // Allocate the 5-second rolling buffer in PSRAM (ESP32-S3) when
  // available; fall back to internal SRAM otherwise (works at 32 KB
  // but uses ~10 % of the chip's RAM).
  #ifdef BOARD_HAS_PSRAM
    _featBuf = (int16_t*)ps_malloc(FEAT_WINDOW_SAMPLES * sizeof(int16_t));
  #endif
  if (!_featBuf) {
    _featBuf = (int16_t*)malloc(FEAT_WINDOW_SAMPLES * sizeof(int16_t));
  }
  if (!_featBuf) {
    Serial.println("[FEAT] FATAL: could not allocate 32 KB rolling buffer — "
                   "falling back to raw-PCM publish for this boot");
  } else {
    omni::init(SAMPLE_RATE);
    bool ok = omni::inference_begin();
    Serial.printf("[FEAT] on-device features: ENABLED (sr=%d, window=%ds)\n",
                  SAMPLE_RATE, FEAT_WINDOW_S);
    Serial.printf("[OOD] %s — %s\n",
                  ok ? "tflite-micro initialised" : "model unavailable",
                  omni::inference_status());
  }
#endif

  Serial.println("[SETUP] complete — entering capture loop");
}

void loop() {
  // Keep WiFi + OTA + MQTT alive
  if (WiFi.status() != WL_CONNECTED) wifi_connect();
  ArduinoOTA.handle();
  _mqttClient.loop();

  _frameCount++;

  // Capture one 1-second frame via FIFO polling
  if (!adxl_capture_frame()) {
    _dropCount++; return;
  }

  // DC removal — strip gravity component so only pipe vibration remains
  remove_dc_offset(_zBuf, FRAME_SAMPLES);

  // VAD: compute RMS in physical g units
  float rms_g  = compute_rms_g(_zBuf, FRAME_SAMPLES);
  float snr_db = compute_snr_db(rms_g);

  if (rms_g < VAD_THRESHOLD_G) {
    Serial.printf("[VAD] frame=%lu silent (%.5fg < %.5fg) — dropped\n",
                  _frameCount, rms_g, VAD_THRESHOLD_G);
    _dropCount++;
  } else {
#if defined(OMNI_PUBLISH_FEATURES) && OMNI_PUBLISH_FEATURES
    if (_featBuf) {
      publish_features(rms_g, snr_db);
    } else {
      publish_acoustic(rms_g, snr_db);   // PSRAM-alloc fallback
    }
#else
    publish_acoustic(rms_g, snr_db);
#endif
  }

  if (_frameCount % TELEMETRY_PERIOD == 0) {
    publish_telemetry();
  }

#if DEEP_SLEEP_US > 0
  esp_deep_sleep(DEEP_SLEEP_US);
#endif
}
