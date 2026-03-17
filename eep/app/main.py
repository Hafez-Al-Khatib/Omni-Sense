"""
EEP — External Endpoint (API Gateway)
========================================
The system's single entry point. Orchestrates IEP1 and IEP2,
validates payloads, enforces rate limits, and performs signal QA.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.routes.diagnose import router as diagnose_router
from app.routes.calibrate import router as calibrate_router
from app.middleware.rate_limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eep")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Omni-Sense — Acoustic Diagnostics Platform",
    description=(
        "Cloud-native acoustic diagnostic API for urban infrastructure. "
        "Upload audio samples to detect water leaks with OOD safety gating."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (for web UI) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Prometheus Instrumentation ──
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app)

# ── Routes ──
app.include_router(diagnose_router, prefix="/api/v1", tags=["Diagnostics"])
app.include_router(calibrate_router, prefix="/api/v1", tags=["Calibration"])


@app.get("/health")
async def health():
    """System-wide health check."""
    return {
        "status": "healthy",
        "service": "eep",
        "version": "0.1.0",
        "iep1_url": settings.IEP1_URL,
        "iep2_url": settings.IEP2_URL,
    }
