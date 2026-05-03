  #ifndef CONFIG_H
  #define CONFIG_H

  #define WIFI_SSID "Hafez-WiFi"
  #define WIFI_PASSWORD "01061956HRMK$"

  // Production MQTT broker (direct IP avoids DNS dependency on constrained networks)
  // Caddy proxies wss://mqtt.omni-sense.online → broker:9001 for browser dashboards.
  // ESP32 connects directly to port 1883 for native MQTT.
  #define MQTT_BROKER "65.109.163.224"
  #define MQTT_HOST MQTT_BROKER
  #define MQTT_PORT 1883
  #define MQTT_USER "omni"
  #define MQTT_PASSWORD "VuopaI9vrdpmtH0svAdjpe48"

  // Sensor ID: specific name (e.g. "esp32-s3-01") or "AUTO" to derive from MAC.
  // AUTO lets you flash the same binary to every sensor without editing.
  #define SENSOR_ID "AUTO"

  #define SITE_ID "demo-site"
  #define FIRMWARE_VERSION "1.0.0"

  #endif