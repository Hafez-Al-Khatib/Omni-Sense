"""MLOps Feedback Watcher — closes the active-learning loop.

Watches ``iep3/feedback_log.csv`` for new technician corrections and triggers
model retraining when a batch threshold is reached.

The loop
--------
IEP3 feedback endpoint
  → appends row to feedback_log.csv
  → FeedbackWatcher detects Δrows ≥ BATCH_SIZE
  → calls scripts/train_omni_heads.py with --feedback-csv arg
  → on success: hot-reloads ONNX sessions in the EEP orchestrator
  → logs retraining event to audit bus

Configuration (environment variables)
--------------------------------------
FEEDBACK_LOG_PATH   — path to IEP3 feedback CSV (default: iep3/feedback_log.csv)
FEEDBACK_BATCH_SIZE — new records needed to trigger retraining (default: 20)
FEEDBACK_POLL_S     — polling interval in seconds (default: 60)
OMNI_CLIPS_DIR      — WAV clips directory for retraining (default: data/synthesized)
OMNI_MODELS_DIR     — output directory for new ONNX models (default: omni/models)
"""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("mlops.feedback_watcher")

FEEDBACK_LOG  = Path(os.getenv("FEEDBACK_LOG_PATH", "iep3/feedback_log.csv"))
BATCH_SIZE    = int(os.getenv("FEEDBACK_BATCH_SIZE", "20"))
POLL_INTERVAL = float(os.getenv("FEEDBACK_POLL_S", "60"))
CLIPS_DIR     = Path(os.getenv("OMNI_CLIPS_DIR", "data/synthesized"))
MODELS_DIR    = Path(os.getenv("OMNI_MODELS_DIR", "omni/models"))

_last_seen_count: int = 0
_retrain_count: int = 0


def _count_feedback_rows() -> int:
    """Return number of data rows in the feedback CSV (header not counted)."""
    if not FEEDBACK_LOG.exists():
        return 0
    try:
        with FEEDBACK_LOG.open(newline="") as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader)
        return max(0, rows - 1)   # subtract header row
    except Exception as exc:
        log.warning("Could not read feedback log: %s", exc)
        return 0


def _trigger_retraining() -> bool:
    """Run train_omni_heads.py synchronously. Returns True on success."""
    global _retrain_count
    script = Path("scripts/train_omni_heads.py")
    if not script.exists():
        log.warning("Training script not found at %s — skipping retrain", script)
        return False

    cmd = [
        sys.executable, str(script),
        "--clips-dir", str(CLIPS_DIR),
        "--output-dir", str(MODELS_DIR),
        "--feedback-csv", str(FEEDBACK_LOG),
    ]

    log.info("Triggering retrain: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,   # 10-minute hard limit
        )
        if result.returncode == 0:
            _retrain_count += 1
            log.info(
                "Retraining #%d complete. stdout tail:\n%s",
                _retrain_count,
                result.stdout[-1000:] if result.stdout else "(no output)",
            )
            return True
        else:
            log.error(
                "Retraining failed (rc=%d). stderr:\n%s",
                result.returncode,
                result.stderr[-1000:] if result.stderr else "(no output)",
            )
            return False
    except subprocess.TimeoutExpired:
        log.error("Retraining timed out after 10 minutes")
        return False
    except Exception as exc:
        log.error("Retraining subprocess error: %s", exc)
        return False


def _hot_reload_onnx() -> None:
    """Signal the EEP orchestrator to reload ONNX sessions from disk.

    We do this by calling _load_models() again on the already-imported
    orchestrator module.  This is safe because _load_models() uses
    module-level globals and onnxruntime sessions are thread-safe for
    read (inference).
    """
    try:
        from omni.eep import orchestrator
        orchestrator._xgb_session = None
        orchestrator._rf_session  = None
        orchestrator._load_models()
        log.info("ONNX sessions hot-reloaded in EEP orchestrator")
    except Exception as exc:
        log.warning("Hot-reload failed (orchestrator may not be in-process): %s", exc)


async def _publish_retrain_event(new_rows: int) -> None:
    """Publish a retrain audit event to the bus."""
    try:
        from omni.common.bus import Topics, get_bus
        from omni.common.schemas import AuditEvent
        event = AuditEvent(
            event_type="mlops.retrain_triggered",
            actor="feedback_watcher",
            details={
                "feedback_rows_processed": new_rows,
                "retrain_count": _retrain_count,
                "models_dir": str(MODELS_DIR),
            },
        )
        await get_bus().publish(Topics.AUDIT, event)
    except Exception as exc:
        log.debug("Could not publish retrain audit event: %s", exc)


async def watch_loop() -> None:
    """Async polling loop — run this as an asyncio task at platform startup."""
    global _last_seen_count

    log.info(
        "Feedback watcher started: path=%s batch=%d poll=%.0fs",
        FEEDBACK_LOG, BATCH_SIZE, POLL_INTERVAL,
    )

    # Initialise baseline so we don't retrain on existing rows at startup
    _last_seen_count = _count_feedback_rows()
    log.info("Baseline feedback count: %d rows", _last_seen_count)

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        current = _count_feedback_rows()
        new_rows = current - _last_seen_count

        if new_rows <= 0:
            continue

        log.info("Feedback watcher: %d new rows (total=%d)", new_rows, current)

        if new_rows >= BATCH_SIZE:
            log.info(
                "Batch threshold reached (%d ≥ %d) — triggering retrain",
                new_rows, BATCH_SIZE,
            )
            success = await asyncio.to_thread(_trigger_retraining)
            if success:
                _last_seen_count = current
                _hot_reload_onnx()
                await _publish_retrain_event(new_rows)
            else:
                log.warning(
                    "Retrain failed — will retry after next %d new rows",
                    BATCH_SIZE,
                )
        else:
            log.debug(
                "Accumulating feedback: %d/%d rows toward next retrain",
                new_rows, BATCH_SIZE,
            )


def start(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
    """Schedule the watcher as a background asyncio task and return the Task."""
    if loop is None:
        loop = asyncio.get_event_loop()
    task = loop.create_task(watch_loop(), name="feedback_watcher")
    log.info("Feedback watcher task created")
    return task
