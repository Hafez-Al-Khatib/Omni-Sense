# Omni-Sense Live API
# Production-ready FastAPI service

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies if needed (e.g. for pandas / asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py db.py mqtt_client.py ./

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser $APP_HOME
USER appuser

EXPOSE 8000

# Uvicorn with a single worker (scale horizontally via replicas)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
