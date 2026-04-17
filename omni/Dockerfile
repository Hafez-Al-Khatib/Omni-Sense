FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY omni/requirements.txt /app/omni/requirements.txt
RUN pip install --no-cache-dir -r omni/requirements.txt && \
    pip install --no-cache-dir paho-mqtt>=2.0 streamlit>=1.33

# Copy source
COPY omni/ /app/omni/
COPY iep2/models/ /app/iep2/models/
COPY Processed_audio_16k/ /app/Processed_audio_16k/

# Health check — verify the platform can import cleanly
RUN python -c "from omni.cmms.rul_model import get_model; get_model(); print('RUL model OK')"

EXPOSE 8501

CMD ["streamlit", "run", "omni/ops_console/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
