# Omni-Sense ESP32 Edge Node

ESP32 DevKit firmware for the INMP441 MEMS microphone.  Captures 0.975 s
acoustic frames at 16 kHz, applies VAD, base64-encodes PCM16 and publishes
to the Omni-Sense MQTT broker over TLS.

---

## Wiring — INMP441 I2S MEMS Microphone to ESP32

```
INMP441 Pin   →   ESP32 GPIO        Notes
──────────────────────────────────────────────────────────────────
VDD           →   3.3V              Do NOT connect to 5V
GND           →   GND
WS  (L/R)     →   GPIO 15           Word Select (frame sync)
SCK (CLK)     →   GPIO 14           Bit clock
SD  (DATA)    →   GPIO 32           Serial data output
L/R           →   GND               LEFT channel (address select)
```

Physical notes:
- Use short wire runs (< 10 cm) to minimise noise at 16 kHz clock
- Decouple VDD with a 100 nF ceramic capacitor between VDD and GND, close to the mic
- The INMP441 outputs 24-bit data left-justified in 32-bit I2S frames; the firmware shifts right to extract 16-bit PCM

### ESP32 DevKit Pinout (38-pin)

```
               USB
         ┌─────────┐
  EN  ── │ 1    38 │ ── GND
  VP  ── │ 2    37 │ ── 3.3V  ← INMP441 VDD
  VN  ── │ 3    36 │ ── 5V
  IO34── │ 4    35 │ ── GND   ← INMP441 GND
  IO35── │ 5    34 │ ── IO23
  IO32── │ 6    33 │ ── IO22  ← also I2C SDA (unused)
  IO33── │ 7    32 │ ── IO1
  IO25── │ 8    31 │ ── IO3
  IO26── │ 9    30 │ ── IO21
  IO27── │10    29 │ ── IO19
  IO14── │11    28 │ ── IO18  INMP441 SCK ← GPIO 14
  IO12── │12    27 │ ── IO5
  GND ── │13    26 │ ── IO17
  IO13── │14    25 │ ── IO16
  IO9 ── │15    24 │ ── IO4
  IO10── │16    23 │ ── IO0
  IO11── │17    22 │ ── IO2
  IO6 ── │18    21 │ ── IO15  INMP441 WS ← GPIO 15
  IO7 ── │19    20 │ ── IO8
         └─────────┘
  INMP441 SD → GPIO 32 (pin 6 on left column)
```

---

## Optional: ADXL345 via SPI (instead of INMP441)

If using ADXL345 for vibration measurement instead of acoustic:

```
ADXL345 Pin   →   ESP32 GPIO
──────────────────────────────
VCC           →   3.3V
GND           →   GND
CS            →   GPIO 5   (SPI CS)
SDO / MISO    →   GPIO 19  (SPI MISO)
SDA / MOSI    →   GPIO 23  (SPI MOSI)
SCL / CLK     →   GPIO 18  (SPI CLK)
INT1          →   (unconnected)
```

Uncomment `#define USE_ADXL345_SPI` at the top of `omni_sensor.ino` and
link in the `SparkFun ADXL345` library via PlatformIO.

---

## Build & Flash Instructions

### Prerequisites

Install [PlatformIO](https://platformio.org/):
```bash
pip install platformio
```

### Build

```bash
cd hardware/esp32
pio run                          # compile for esp32dev
pio run -e esp32s3               # compile for ESP32-S3
```

### Flash via USB

```bash
pio run --target upload          # auto-detect COM port
pio run --target upload --upload-port /dev/ttyUSB0  # explicit port
```

On macOS: port is typically `/dev/cu.usbserial-*`

Hold the **BOOT** button while pressing **EN** if the board does not enter
flash mode automatically.

### Monitor serial output

```bash
pio device monitor --baud 115200
```

### OTA update (wireless)

1. Uncomment `upload_protocol = espota` in `platformio.ini`
2. Set the device's IP address in `upload_port`
3. `pio run --target upload`

---

## Certificates

The firmware embeds TLS certificates as `const char*` strings.

1. Generate a CA and client cert:
   ```bash
   bash ../../omni/scripts/gen_certs.sh S-ESP32-001
   ```

2. Convert PEM to C string format:
   ```bash
   awk 'NF {sub(/\r/, ""); printf "%s\\n", $0}' ca.crt
   ```

3. Paste into the `CA_CERT[]`, `CLIENT_CERT[]`, `CLIENT_KEY[]` arrays in
   `omni_sensor.ino`.

4. For production, store certs in SPIFFS:
   ```bash
   pio run --target uploadfs       # upload SPIFFS filesystem image
   ```

---

## Power Consumption

| Mode                          | Typical current (3.3V) |
|-------------------------------|------------------------|
| Active capture + WiFi TX      | ~200-250 mA            |
| WiFi connected, idle          | ~80 mA                 |
| Light sleep (WiFi modem off)  | ~0.8 mA                |
| Deep sleep (RTC only)         | ~10 µA                 |
| Deep sleep (ULP active)       | ~100 µA                |

### Battery life estimates (1000 mAh LiPo)

| Mode                          | Estimated life |
|-------------------------------|----------------|
| Continuous capture            | ~4-5 hours     |
| 1 frame/6 s + deep sleep      | ~7-8 days      |
| 1 frame/60 s + deep sleep     | ~30-40 days    |

To enable deep sleep between captures, set in `omni_sensor.ino`:
```c
#define DEEP_SLEEP_US  4800000ULL   // 4.8 s sleep → ~1 frame / 6 s
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All zeros in PCM buffer | INMP441 wiring fault | Check WS, SCK, SD pins; add 100nF cap near VDD |
| "malloc failed" | Insufficient heap | Use `huge_app.csv` partition; avoid large static arrays |
| MQTT `state=-2` | TLS handshake failed | Verify CA cert; check clock sync (NTP) |
| OTA not found | Firewall / IP wrong | `ping <sensor_ip>`, check `upload_port` in platformio.ini |
| Brownout reboot | Insufficient power | Use 500 mA+ USB supply; add 100 µF capacitor on 5V rail |
| Continuous capture very loud | Gain too high | INMP441 has fixed gain; move mic away from vibration source |
