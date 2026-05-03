FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py .
COPY bridge.py .

ENV PYTHONUNBUFFERED=1
ENV OMNI_MQTT_HOST=mqtt-broker
ENV OMNI_MQTT_PORT=1883
ENV OMNI_EEP_URL=http://eep:8000
ENV BRIDGE_WINDOW_S=5.0

CMD ["python", "-u", "bridge.py"]
