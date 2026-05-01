# Omni-Sense ESP32 Edge Node

ESP32-S3 DevKitC firmware for the **ADXL345 MEMS accelerometer**.  Captures
structure-borne vibration frames at **3,200 Hz** — the maximum output data rate
(ODR) of the ADXL345 — applies VAD, base64-encodes the Z-axis data and publishes
to the local Omni-Sense MQTT broker over TLS.

> **Honest disclosure:** The ADXL345 maximum ODR is 3,200 Hz.  We do **not**
> claim 16 kHz.  The 50–1,600 Hz leak detection band fits comfortably within
> the 3,200 Hz Nyquist (1,600 Hz), making this sensor fully adequate for pipe
> leak detection.

---

## Wiring — ADXL345 SPI to ESP32-S3

```
ADXL345 Pin   →   ESP32-S3 GPIO        Notes
──────────────────────────────────────────────────────────────────
VCC           →   3.3V                 Do NOT connect to 5V
GND           →   GND
CS            →   GPIO 5               SPI chip select
SDO / MISO    →   GPIO 19              SPI MISO
SDA / MOSI    →   GPIO 23              SPI MOSI
SCL / CLK     →   GPIO 18              SPI clock
INT1          →   not connected        polling mode
INT2          →   not connected
```

Physical notes:
- Mount the ADXL345 breakout board **flat on the pipe surface** with the Z-axis
  (top face) perpendicular to the pipe wall.
- Use thermal epoxy or a hose-clamp adapter.  Ensure no air gap — acoustic
  coupling efficiency drops sharply otherwise.
- Keep SPI wires under 20 cm to minimise ringing at 4 MHz.

### ESP32-S3 DevKitC-1 Pinout (38-pin)

```
               USB-C
         ┌─────────┐
  3.3V ──│ 1    38│── GND
  GPIO0 ──│ 2    37│── GPIO19  ← ADXL345 SDO/MISO
  GPIO1 ──│ 3    36│── GPIO20
  GPIO2 ──│ 4    35│── GPIO21
  GPIO3 ──│ 5    34│── GPIO47
  GPIO4 ──│ 6    33│── GPIO48
  GPIO5 ──│ 7    32│── GPIO45  ← ADXL345 CS
  GPIO6 ──│ 8    31│── GPIO38
  GPIO7 ──│ 9    30│── GPIO39
  GPIO8 ──│10    29│── GPIO40
  GPIO9 ──│11    28│── GPIO41
  GPIO10 ──│12    27│── GPIO42
  GPIO11 ──│13    26│── GPIO2
  GPIO12 ──│14    25│── GPIO1
  GPIO13 ──│15    24│── GPIO15
  GPIO14 ──│16    23│── GPIO16
  GPIO15 ──│17    22│── GPIO17
  GPIO16 ──│18    21│── GPIO18  ← ADXL345 SCL/CLK
  GPIO17 ──│19    20│── GPIO8
         └─────────┘
  ADXL345 SDA/MOSI → GPIO 23 (pin 37 on right column, or pin 23 via strap)
```

---

## Build & Flash Instructions

### Prerequisites

Install [PlatformIO](https://platformio.org/):
```bash
pip install platformio
```

### 1. Configure Wi-Fi and broker

```bash
cd hardware/esp32
cp config.h config.h          # edit to taste (already has safe defaults)
```

Edit `config.h`:
```c
#define WIFI_SSID     "YourActualSSID"
#define WIFI_PASSWORD "YourActualPassword"
#define MQTT_HOST     "192.168.1.100"   // your Mosquitto broker IP
#define MQTT_PORT     8883
```

`config.h` is in `.gitignore` — credentials will never be committed.

### 2. Generate TLS certificates

```bash
bash scripts/gen_esp32_certs.sh S-ESP32-001
```

This creates:
- `certs/ca.crt`, `certs/client.crt`, `certs/client.key` (PEM)
- `certs/ca_cert.h`, `certs/client_cert.h`, `certs/client_key.h` (C headers)

The `.ino` automatically `#include`s the headers.  Generated `*.h` files are
in `.gitignore`.

### 3. Build

```bash
pio run                          # compile for esp32dev
pio run -e esp32s3               # compile for ESP32-S3 (PSRAM enabled)
```

### 4. Flash via USB

```bash
pio run --target upload          # auto-detect COM port
pio run --target upload --upload-port /dev/ttyUSB0  # explicit port
```

On Windows: port is typically `COM3` or higher.  
On macOS: port is typically `/dev/cu.usbserial-*`.

Hold the **BOOT** button while pressing **RST** if the board does not enter
flash mode automatically.

### 5. Monitor serial output

```bash
pio device monitor --baud 115200
```

### OTA update (wireless)

1. Uncomment `upload_protocol = espota` in `platformio.ini`
2. Set the device's IP address in `upload_port`
3. `pio run --target upload`

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

To enable deep sleep between captures, set in `config.h`:
```c
#define DEEP_SLEEP_US  4800000ULL   // 4.8 s sleep → ~1 frame / 6 s
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All zeros in buffer | ADXL345 wiring fault | Check CS, MISO, MOSI, SCK pins; verify 3.3 V |
| `DEVID=0x00` or `0xFF` | SPI not connected | Check wire continuity; ensure CS has pull-up |
| MQTT `state=-2` | TLS handshake failed | Verify CA cert; check clock sync (NTP) |
| OTA not found | Firewall / IP wrong | `ping <sensor_ip>`, check `upload_port` in platformio.ini |
| Brownout reboot | Insufficient power | Use 500 mA+ USB supply; add 100 µF capacitor on 5V rail |
| `malloc failed` | Insufficient heap | Use `huge_app.csv` partition; enable PSRAM on S3 |
