# Omni-Sense Raspberry Pi 5 Edge Inference Gateway

The RPi 5 acts as the **edge inference gateway** in the Omni-Sense architecture.
It receives (or locally collects) vibration data, extracts a 39-dimensional DSP
feature vector, and runs the IEP2 ONNX models locally:

1. **Isolation Forest** — out-of-distribution (OOD) gate
2. **XGBoost classifier** — leak / no-leak diagnosis

Results are published as JSON to `omni/diagnosis/{site_id}/{sensor_id}`.

## Why the RPi 5?

- **Compute:** The Cortex-A76 CPU runs ONNX Runtime efficiently.
- **Honest signal processing:** We use `scipy.signal.resample` (FFT-based) when
  changing sample rates — never linear interpolation fakery.
- **Flexibility:** Can read from local ADXL345 (200 Hz I2C) **or** subscribe to
  ESP32 acoustic frames (3,200 Hz) over MQTT.

## Honest Sampling Rates

| Source | Raw rate | Notes |
|--------|----------|-------|
| RPi local ADXL345 | 200 Hz | I2C FIFO polling, no upsampling |
| ESP32 remote ADXL345 | 3,200 Hz | SPI streaming, honest hardware max |
| Feature extractor | 16,000 Hz | Models trained at 16 kHz; we FFT-resample to match |

> **Important:** The 200 Hz and 3,200 Hz signals are resampled to 16 kHz using
> FFT-based sinc interpolation (`scipy.signal.resample`).  This does **not**
> invent new frequency content — it is the theoretically correct method for
> bandlimited signals.

---

## Hardware: ADXL345 Wiring to Raspberry Pi 5

```
ADXL345 Pin     →    Raspberry Pi 5 GPIO (40-pin header)
─────────────────────────────────────────────────────────
VCC (3.3 V)     →    Pin 1  (3.3V Power)
GND             →    Pin 6  (Ground)
SDA             →    Pin 3  (GPIO2 / I2C1 SDA)
SCL             →    Pin 5  (GPIO3 / I2C1 SCL)
SDO             →    Pin 6  (GND)  ← I2C address = 0x53
CS              →    Pin 1  (3.3V) ← selects I2C mode
INT1            →    (unconnected — we use FIFO polling)
INT2            →    (unconnected)
```

Confirm I2C address after wiring:
```bash
sudo i2cdetect -y 1
# Should show '53' (or '1d') in the grid
```

---

## Software Setup

### 1. Enable I2C on Raspberry Pi OS

```bash
sudo raspi-config
# Interface Options → I2C → Yes → Finish → Reboot
```

### 2. Install dependencies

```bash
cd omni-sense/hardware/rpi_edge_agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify ONNX models are present

The agent expects `isolation_forest.onnx` and `xgboost_classifier.onnx` in
`../../iep2/models/` (relative to `agent.py`).  You can override with:

```bash
export OMNI_MODEL_PATH=/path/to/iep2/models
```

### 4. Run the gateway

**Simulation mode** (no hardware needed):
```bash
SIMULATE_MODE=1 MQTT_HOST=localhost MQTT_PORT=1883 python agent.py
```

**Hardware mode** (requires ADXL345 on I2C):
```bash
HARDWARE_MODE=1 MQTT_HOST=localhost MQTT_PORT=1883 python agent.py
```

**Auto-detect** (tries hardware, falls back to simulation):
```bash
MQTT_HOST=localhost MQTT_PORT=1883 python agent.py
```

### 5. (Optional) Install as systemd service

```bash
sudo bash install.sh
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENSOR_ID` | `S-RPI5-001` | Sensor identity |
| `SITE_ID` | `beirut/hamra` | Site slug |
| `MQTT_HOST` | `localhost` | Mosquitto broker host |
| `MQTT_PORT` | `1883` | Mosquitto broker port |
| `OMNI_MODEL_PATH` | `../../iep2/models` | Path to ONNX models |
| `HARDWARE_MODE` | `auto` | Force I2C ADXL345 read |
| `SIMULATE_MODE` | `auto` | Force synthetic data |
| `VAD_THRESHOLD` | `0.005` | RMS voice-activity gate |
| `FIRMWARE_VER` | `edge-fw-rpi5-v1` | Firmware version string |

---

## Monitoring

```bash
# Live logs
journalctl -u omni-edge -f

# Service status
systemctl status omni-edge

# I2C bus scan
sudo i2cdetect -y 1
```

---

## Power Consumption

| Mode                     | Approx current |
|--------------------------|----------------|
| ADXL345 measurement mode | ~140 µA @ 3.3V |
| Raspberry Pi 5 (idle)    | ~600 mA @ 5V   |
| Raspberry Pi 5 (load)    | ~1000 mA @ 5V  |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ADXL345 not found — DEVID=0xFF` | I2C not enabled or wiring fault | `raspi-config` → I2C; check wiring |
| `DEVID=0x00` | Pull-ups missing | Add 4.7 kΩ pull-ups on SDA/SCL to 3.3V |
| `Model missing` | ONNX files not found | Check `OMNI_MODEL_PATH` |
| MQTT publish fail | Broker not reachable | Verify `MQTT_HOST` / `MQTT_PORT` |
| `Permission denied /dev/i2c-1` | User not in `i2c` group | `sudo usermod -aG i2c $USER` + re-login |
