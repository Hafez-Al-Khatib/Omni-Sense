# Omni-Sense Hardware Stack

Honest, functional hardware for the Omni-Sense pipe leak detection system.

## Architecture

```
┌─────────────────┐      MQTT (TLS)      ┌─────────────────────────┐      HTTPS      ┌─────────────┐
│  ESP32-S3       │  ──────────────────→ │  Raspberry Pi 5         │  ────────────→ │  Cloud EEP  │
│  ADXL345        │   omni/acoustic/+/+  │  (Edge Inference        │                │  (IEP2-4)   │
│  3,200 Hz       │                      │   Gateway)              │                │             │
│  (honest max)   │                      │  ONNX Runtime           │                │             │
└─────────────────┘                      │  scipy resample         │                └─────────────┘
                                         │  39-d DSP features      │
                                         │  OOD → XGBoost          │
                                         └─────────────────────────┘
                                                   │
                                                   ↓
                                         omni/diagnosis/+/+
```

### Why this split?

- **ESP32** is the telemetry node: low power, Wi-Fi, clamps to the pipe.  It
  honestly samples at 3,200 Hz (the ADXL345 maximum) and publishes raw
  structure-borne vibration frames.
- **Raspberry Pi 5** is the edge inference gateway: it has enough CPU to run
  ONNX Runtime, perform honest FFT-based resampling (`scipy.signal.resample`),
  and extract the 39-dimensional DSP feature vector.  It can read from a local
  ADXL345 (200 Hz I2C) **or** subscribe to ESP32 frames over MQTT.
- **Cloud EEP** receives only the *diagnosis* JSON, not massive raw PCM payloads.
  This saves bandwidth and provides sub-second local alerting even if the cloud
  link is down.

## Honest Sampling Rate Disclosure

| Device | Claimed before | Honest rate | Nyquist | Leak band coverage |
|--------|---------------|-------------|---------|-------------------|
| ESP32  | 16 kHz        | **3,200 Hz**| 1,600 Hz| 50–1,600 Hz ✓     |
| RPi local | 16 kHz (fake upsampling) | **200 Hz**| 100 Hz| 50–100 Hz partial |

> **Why 3,200 Hz is sufficient:** Water pipe leak signatures are dominated by
> turbulent flow noise (50–500 Hz) and orifice whistle (500–1,600 Hz).  The
> ADXL345 at 3,200 Hz captures the entire relevant band.  We do **not** claim
> 16 kHz because the hardware cannot deliver it.

> **Why RPi local is 200 Hz:** The RPi reads the ADXL345 over I2C.  200 Hz is
> the practical FIFO polling rate.  We do **not** upsample linearly to fake
> 16 kHz.  Instead, we use FFT-based resampling to 16 kHz (for model
> compatibility), which is mathematically correct and does not invent frequency
> content.

## Bill of Materials

| Item | Qty | Purpose | Est. cost |
|------|-----|---------|-----------|
| ESP32-S3-DevKitC-1 | 1 | Wi-Fi telemetry node, TLS, OTA | $12 |
| ADXL345 breakout (SPI/I2C) | 2 | Structure-borne vibration sensor | $6 × 2 |
| Raspberry Pi 5 (4 GB) | 1 | Edge inference gateway | $60 |
| Micro-SD card 32 GB | 1 | Pi OS storage | $8 |
| Dupont jumper wires (M-F, M-M) | 40 | Breadboard / prototype wiring | $5 |
| 3.3 V regulator / LM1117 | 1 | Clean power for ADXL345 (optional) | $2 |
| Thermal epoxy / hose clamp | 2 | Mechanical coupling to pipe | $5 |
| USB-C power supply 3 A | 1 | Power RPi 5 | $10 |

**Total: ~$100–110**

## Folder Layout

```
hardware/
├── esp32/                    # Telemetry node firmware
│   ├── omni_sensor/
│   │   └── omni_sensor.ino   # Arduino / PlatformIO sketch
│   ├── scripts/
│   │   └── gen_esp32_certs.sh# Self-signed CA + client cert generator
│   ├── config.h              # Wi-Fi + broker credentials (gitignored)
│   ├── platformio.ini        # Build configs for esp32dev & esp32s3
│   └── README.md             # Wiring, build & flash instructions
├── rpi_edge_agent/           # Edge inference gateway
│   ├── agent.py              # Main gateway (ONNX + MQTT)
│   ├── requirements.txt      # Python dependencies
│   ├── install.sh            # Systemd service installer
│   ├── omni-edge.service     # Systemd unit template
│   └── README.md             # Setup & env var reference
├── scripts/
│   └── demo_edge_pipeline.py # Integration test / one-shot demo
└── README.md                 # This file
```

## Quick Start

### 1. Start the Mosquitto broker

From the project root:
```bash
docker compose up mqtt-broker -d
```

The broker listens on:
- `1883` — plain TCP (for local lab / demo)
- `8883` — mTLS (for production ESP32 / RPi)

### 2. Flash the ESP32

```bash
cd hardware/esp32
bash scripts/gen_esp32_certs.sh S-ESP32-001
# Edit config.h with your Wi-Fi credentials
pio run -e esp32s3 --target upload
```

### 3. Run the RPi edge gateway

```bash
cd hardware/rpi_edge_agent
pip install -r requirements.txt
SIMULATE_MODE=1 MQTT_HOST=localhost MQTT_PORT=1883 python agent.py
```

### 4. Run the integration demo

```bash
python hardware/scripts/demo_edge_pipeline.py
```

## Capstone Demo Checklist

- [ ] ESP32-S3 boots and prints `ODR=3200 Hz` (not 16 kHz)
- [ ] ESP32 publishes base64 acoustic frames to `omni/acoustic/+/+`
- [ ] RPi agent loads `isolation_forest.onnx` and `xgboost_classifier.onnx`
- [ ] RPi agent runs VAD + feature extraction + ONNX inference
- [ ] RPi agent publishes JSON with `label`, `confidence`, `is_ood`
- [ ] `demo_edge_pipeline.py` completes with `VALIDATION PASS`
