from __future__ import annotations

from ..config import Settings
from .base import CAPABILITIES, GenerationError, GenerationRequest, GenerationResult, InputImage, Provider
from .fal import FalProvider
from .local import LocalProvider
from .mock import MockProvider

__all__ = [
    "CAPABILITIES",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "InputImage",
    "Provider",
    "build_provider",
]


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "mock":
        return MockProvider(step_delay=settings.mock_step_delay)
    if settings.provider == "fal":
        return FalProvider(
            api_key=settings.fal_key,
            text_to_image_model=settings.fal_text_to_image_model,
            default_engine=settings.fal_engine,
            depth_model=settings.fal_depth_model,
            timeout_s=settings.fal_timeout_s,
        )
    if settings.provider == "local":
        return LocalProvider(
            url=settings.local_worker_url, token=settings.local_worker_token, timeout_s=settings.local_timeout_s
        )
    raise ValueError(f"Proveedor desconocido: {settings.provider!r} (usa 'mock', 'local' o 'fal')")
