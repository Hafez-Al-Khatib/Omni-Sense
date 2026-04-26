FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for torch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY iep2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY iep2/app/ ./app/
COPY iep2/models/ ./models/
COPY omni/ ./omni/

# Ensure omni module is discoverable
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
