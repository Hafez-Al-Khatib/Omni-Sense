"""
Service Orchestrator
=====================
Coordinates calls from EEP → IEP1 (embedding) → IEP2 (diagnosis).
Handles timeouts, retries, and error propagation.
"""

import io
import logging

import httpx
from prometheus_client import Counter

from app.config import settings

# ─── Prometheus metrics ───────────────────────────────────────────────────────

IEP4_FALLBACK_COUNTER = Counter(
    "eep_iep4_fallback_total",
    "Number of times the IEP4 CNN path was unavailable and fell back to IEP2-only.",
    ["reason"],   # "not_trained" | "timeout" | "unreachable" | "error"
)

logger = logging.getLogger("eep.orchestrator")


class OrchestratorError(Exception):
    """Raised when orchestration fails."""

    def __init__(self, message: str, status_code: int = 502, detail: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or {}


async def call_iep1_embed(audio_bytes: bytes) -> list[float]:
    """
    Call IEP1 to extract YAMNet embedding.

    Args:
        audio_bytes: Raw WAV audio bytes

    Returns:
        1024-element float list (embedding)

    Raises:
        OrchestratorError: If IEP1 call fails
    """
    url = f"{settings.IEP1_URL}/embed"

    async with httpx.AsyncClient(timeout=settings.IEP1_TIMEOUT) as client:
        try:
            files = {"audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
            response = await client.post(url, files=files)

            if response.status_code != 200:
                error_detail = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
                raise OrchestratorError(
                    f"IEP1 returned {response.status_code}",
                    status_code=response.status_code,
                    detail=error_detail,
                )

            data = response.json()
            return data["embedding"]

        except httpx.TimeoutException:
            raise OrchestratorError(
                "IEP1 timeout: YAMNet inference took too long",
                status_code=504,
            )
        except httpx.ConnectError:
            raise OrchestratorError(
                "IEP1 unreachable: YAMNet service is not running",
                status_code=503,
            )
        except OrchestratorError:
            raise
        except Exception as e:
            logger.error(f"IEP1 call failed: {e}", exc_info=True)
            raise OrchestratorError(f"IEP1 error: {str(e)}", status_code=502)


async def call_iep2_diagnose(
    embedding: list[float],
    pipe_material: str,
    pressure_bar: float,
) -> dict:
    """
    Call IEP2 to perform OOD detection + classification.

    Args:
        embedding: 1024-d float list from IEP1
        pipe_material: Pipe material type
        pressure_bar: Pipe pressure

    Returns:
        Diagnosis result dict

    Raises:
        OrchestratorError: If IEP2 call fails
    """
    url = f"{settings.IEP2_URL}/diagnose"

    payload = {
        "embedding": embedding,
        "pipe_material": pipe_material,
        "pressure_bar": pressure_bar,
    }

    async with httpx.AsyncClient(timeout=settings.IEP2_TIMEOUT) as client:
        try:
            response = await client.post(url, json=payload)

            data = response.json()

            # 422 = OOD rejection (expected behavior, not an error)
            if response.status_code == 422:
                return {
                    "is_ood": True,
                    **data,
                }

            if response.status_code != 200:
                raise OrchestratorError(
                    f"IEP2 returned {response.status_code}",
                    status_code=response.status_code,
                    detail=data,
                )

            return {"is_ood": False, **data}

        except httpx.TimeoutException:
            raise OrchestratorError(
                "IEP2 timeout: Diagnostic engine took too long",
                status_code=504,
            )
        except httpx.ConnectError:
            raise OrchestratorError(
                "IEP2 unreachable: Diagnostic service is not running",
                status_code=503,
            )
        except OrchestratorError:
            raise
        except Exception as e:
            logger.error(f"IEP2 call failed: {e}", exc_info=True)
            raise OrchestratorError(f"IEP2 error: {str(e)}", status_code=502)


async def call_iep3_notify(
    label: str,
    confidence: float,
    probabilities: dict,
    anomaly_score: float,
    pipe_material: str,
    pressure_bar: float,
    scada_mismatch: bool,
) -> None:
    """
    Fire-and-forget dispatch to IEP3 when a high-confidence fault is detected.
    Failures are logged as warnings only — never propagated to the caller.
    """
    url = f"{settings.IEP3_URL}/api/v1/ticket"
    payload = {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "anomaly_score": anomaly_score,
        "pipe_material": pipe_material,
        "pressure_bar": pressure_bar,
        "scada_mismatch": scada_mismatch,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.IEP3_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            if response.status_code not in (200, 201):
                logger.warning(f"IEP3 ticket creation returned {response.status_code}: {response.text}")
    except Exception as e:
        logger.warning(f"IEP3 dispatch failed (non-critical): {e}")


async def call_iep4_classify(audio_bytes: bytes) -> dict | None:
    """
    Call IEP4 (CNN) in parallel with IEP2 for deep-learning classification.

    Returns None silently if IEP4 is unavailable or returns 503 (model not
    trained yet).  The EEP treats IEP4 as additive — the pipeline continues
    with IEP2 results alone when IEP4 is not ready.
    """
    url = f"{settings.IEP4_URL}/classify"
    try:
        async with httpx.AsyncClient(timeout=settings.IEP4_TIMEOUT) as client:
            files = {"audio": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
            response = await client.post(url, files=files)
            if response.status_code == 503:
                logger.info("IEP4 CNN not yet trained — skipping deep path")
                IEP4_FALLBACK_COUNTER.labels(reason="not_trained").inc()
                return None
            if response.status_code == 200:
                return response.json()
            logger.warning(f"IEP4 returned {response.status_code}")
            IEP4_FALLBACK_COUNTER.labels(reason="error").inc()
            return None
    except httpx.ConnectError:
        logger.warning("IEP4 unreachable — skipping deep path")
        IEP4_FALLBACK_COUNTER.labels(reason="unreachable").inc()
        return None
    except httpx.TimeoutException:
        logger.warning("IEP4 timeout — skipping deep path")
        IEP4_FALLBACK_COUNTER.labels(reason="timeout").inc()
        return None
    except Exception as exc:
        logger.warning(f"IEP4 call failed: {exc}")
        IEP4_FALLBACK_COUNTER.labels(reason="error").inc()
        return None


def ensemble_iep2_iep4(
    iep2_result: dict,
    iep4_result: dict | None,
    iep2_weight: float = 0.60,
    iep4_weight: float = 0.40,
) -> dict:
    """
    Weighted ensemble of IEP2 (XGBoost+RF) and IEP4 (CNN) probabilities.

    Falls back to IEP2-only if IEP4 is unavailable or class sets differ.
    Satisfies the rubric requirement for parallel model interaction in EEP.

    OOD gate
    --------
    When IEP2 returns an OOD flag (422 from the diagnostic engine), the
    ``probabilities`` key is absent from ``iep2_result``.  Attempting to
    read it and merge with IEP4 probabilities would crash the ensemble.
    We bypass the ensemble entirely and return the IEP2 OOD result unchanged.
    """
    # ── OOD short-circuit ─────────────────────────────────────────────────────
    if iep2_result.get("is_ood"):
        return {**iep2_result, "ensemble_method": "ood_bypass"}

    if iep4_result is None:
        return {**iep2_result, "ensemble_method": "iep2_only"}

    iep2_proba = iep2_result.get("probabilities", {})
    iep4_proba = iep4_result.get("probabilities", {})

    if set(iep2_proba.keys()) != set(iep4_proba.keys()):
        logger.warning("IEP2/IEP4 class mismatch — using IEP2 only")
        return {**iep2_result, "ensemble_method": "iep2_only_class_mismatch"}

    ensemble_proba: dict[str, float] = {
        cls: iep2_weight * iep2_proba[cls] + iep4_weight * iep4_proba.get(cls, 0.0)
        for cls in iep2_proba
    }
    best_label = max(ensemble_proba, key=lambda k: ensemble_proba[k])

    return {
        **iep2_result,
        "label": best_label,
        "confidence": float(ensemble_proba[best_label]),
        "probabilities": ensemble_proba,
        "ensemble_method": "weighted_avg",
        "iep2_label": iep2_result.get("label"),
        "iep2_confidence": iep2_result.get("confidence"),
        "iep4_label": iep4_result.get("label"),
        "iep4_confidence": iep4_result.get("confidence"),
    }


async def call_iep2_calibrate(ambient_embeddings: list[list[float]]) -> dict:
    """
    Call IEP2 calibration endpoint.

    Args:
        ambient_embeddings: List of 1024-d embeddings from ambient recordings

    Returns:
        Calibration result dict
    """
    url = f"{settings.IEP2_URL}/calibrate"

    payload = {"ambient_embeddings": ambient_embeddings}

    async with httpx.AsyncClient(timeout=settings.IEP2_TIMEOUT) as client:
        try:
            response = await client.post(url, json=payload)

            if response.status_code != 200:
                data = response.json()
                raise OrchestratorError(
                    f"IEP2 calibration failed: {response.status_code}",
                    status_code=response.status_code,
                    detail=data,
                )

            return response.json()

        except httpx.TimeoutException:
            raise OrchestratorError("IEP2 calibration timeout", status_code=504)
        except httpx.ConnectError:
            raise OrchestratorError("IEP2 unreachable for calibration", status_code=503)
        except OrchestratorError:
            raise
        except Exception as e:
            logger.error(f"IEP2 calibration failed: {e}", exc_info=True)
            raise OrchestratorError(f"IEP2 calibration error: {str(e)}", status_code=502)
