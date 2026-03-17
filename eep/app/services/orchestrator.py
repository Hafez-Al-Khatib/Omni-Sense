"""
Service Orchestrator
=====================
Coordinates calls from EEP → IEP1 (embedding) → IEP2 (diagnosis).
Handles timeouts, retries, and error propagation.
"""

import io
import logging

import httpx
import numpy as np

from app.config import settings

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
