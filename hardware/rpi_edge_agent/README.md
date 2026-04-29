# Omni-Sense Raspberry Pi Edge Agent

Reads vibration data from an ADXL345 accelerometer attached to a water pipe,
applies VAD, encodes 0.975-second PCM16 frames and publishes them over MQTT
with mTLS to the Omni-Sense cloud broker.

---

## Hardware: ADXL345 Wiring to Raspberry Pi 4

The ADXL345 communicates over I2C (3.3 V logic).

```
ADXL345 Pin     →    Raspberry Pi 4 GPIO (40-pin header)
─────────────────────────────────────────────────────────
VCC (3.3 V)     →    Pin 1  (3.3V Power)
GND             →    Pin 6  (Ground)
SDA             →    Pin 3  (GPIO2 / I2C1 SDA)
SCL             →    Pin 5  (GPIO3 / I2C1 SCL)
SDO             →    Pin 6  (GND)  ← I2C address = 0x53
CS              →    Pin 1  (3.3V) ← selects I2C mode
INT1            →    (unconnected — we use FIFO polling)
INT2            →    (unconnected)

Alternative SDO wiring:
  SDO → 3.3V   →  I2C address becomes 0x1D
                   (change ADXL345_ADDR in agent.py)
```

### Physical installation

Mount the ADXL345 **directly on the pipe** with epoxy or a hose clamp bracket.
Orient the Z-axis perpendicular to the pipe surface to maximise vibration
sensitivity.  Waterproof the assembly with self-amalgamating tape.

```
Side view:
               ┌──────────────┐
pipe ─────────=│  ADXL345 PCB │=───────── pipe
               │  Z↑ out of   │
               │  pipe wall   │
               └──────────────┘
                     │ ribbon cable → Pi
```

---

## Raspberry Pi 4 Pin-out Reference

```
         3.3V ─ [1] [2] ─ 5V
   SDA (GPIO2) ─ [3] [4] ─ 5V
   SCL (GPIO3) ─ [5] [6] ─ GND  ← wire ADXL SDO here for addr 0x53
          GPIO4 ─ [7] [8]
          GND  ─ [9][10]
         ...
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

Or edit `/boot/firmware/config.txt` (Bookworm) / `/boot/config.txt` (Bullseye):
```
dtparam=i2c_arm=on
```

### 2. Install the agent (automated)

```bash
git clone https://github.com/your-org/omni-sense.git
cd omni-sense/hardware/rpi_edge_agent
sudo bash install.sh
```

The installer:
- Creates a `omni` system user
- Installs to `/opt/omni-edge/`
- Creates a Python venv and installs pip packages
- Registers and starts `omni-edge.service`

### 3. Install certificates

Copy your mTLS certificates to `/etc/omni/certs/`:

```bash
sudo mkdir -p /etc/omni/certs
sudo cp ca.crt                /etc/omni/certs/
sudo cp S-HAMRA-001.crt       /etc/omni/certs/
sudo cp S-HAMRA-001.key       /etc/omni/certs/
sudo chmod 600 /etc/omni/certs/S-HAMRA-001.key
sudo chown -R omni:omni /etc/omni/certs
```

Generate certs with `omni/scripts/gen_certs.sh` if you are running your own
Mosquitto broker.

### 4. Configure environment

Edit the service unit:
```bash
sudo systemctl edit omni-edge.service
```

Add overrides:
```ini
[Service]
Environment="SENSOR_ID=S-HAMRA-001"
Environment="SITE_ID=beirut/hamra"
Environment="MQTT_HOST=mqtt.your-broker.io"
Environment="VAD_THRESHOLD=0.005"
```

Then reload:
```bash
sudo systemctl daemon-reload && sudo systemctl restart omni-edge
```

---

## Manual / simulation mode

Run without hardware (useful for CI or desk testing):

```bash
cd omni-sense/hardware/rpi_edge_agent
pip install -r requirements.txt
SIMULATE=1 SENSOR_ID=S-TEST-001 SITE_ID=test/lab \
  MQTT_HOST=localhost MQTT_PORT=1883 \
  python agent.py
```

---

## Monitoring

```bash
# Live logs
journalctl -u omni-edge -f

# Service status
systemctl status omni-edge

# I2C bus scan
sudo i2cdetect -y 1

# Verify ADXL345 power register (should return 0x08 when measuring)
sudo i2cget -y 1 0x53 0x2D
```

---

## Power consumption

| Mode                     | Approx current (3.3 V) |
|--------------------------|------------------------|
| ADXL345 measurement mode | ~140 µA                |
| ADXL345 standby          | ~0.1 µA                |
| Raspberry Pi 4 (idle)    | ~600 mA @ 5V           |
| Raspberry Pi 4 (load)    | ~1000 mA @ 5V          |

For battery-powered deployments consider a Raspberry Pi Zero 2 W instead
(~120 mA idle) or an ESP32 (see `hardware/esp32/`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ADXL345 not found — DEVID=0xFF` | I2C not enabled or wiring fault | `raspi-config` → I2C; check wiring |
| `DEVID=0x00` | Pull-ups missing | Add 4.7 kΩ pull-ups on SDA/SCL to 3.3V |
| `DEVID=0xE5` but always quiet | Z-axis not perpendicular to pipe | Re-orient sensor |
| MQTT TLS error | Cert mismatch | Verify CA, check hostname in broker cert |
| `Permission denied /dev/i2c-1` | User not in `i2c` group | `sudo usermod -aG i2c omni` + re-login |
