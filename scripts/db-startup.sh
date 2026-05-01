#!/bin/bash
# Startup script for the GCE e2-small DB box (TimescaleDB + Redis)
set -e

apt-get update
apt-get install -y docker.io docker-compose

# Run TimescaleDB
docker run -d --name timescaledb \
  --restart unless-stopped \
  -e POSTGRES_USER=omni \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=omnisense \
  -p 5432:5432 \
  -v timescale-data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

# Run Redis
docker run -d --name redis \
  --restart unless-stopped \
  -p 6379:6379 \
  redis:7.2-alpine \
  redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
