FROM python:3.11-slim

WORKDIR /app

# Install system deps for numpy, scipy, onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY hardware/rpi_edge_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hardware/rpi_edge_agent/agent.py .
COPY omni/eep/features.py ./omni/eep/

ENV PYTHONPATH=/app
ENV SENSOR_ID=S-RPI5-001
ENV SITE_ID=beirut/hamra
ENV MQTT_HOST=localhost
ENV MQTT_PORT=1883

CMD ["python", "agent.py"]
