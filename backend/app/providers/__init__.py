from __future__ import annotations

from ..config import Settings
from .base import GenerationError, GenerationRequest, GenerationResult, Provider
from .fal import FalProvider
from .mock import MockProvider

__all__ = ["GenerationError", "GenerationRequest", "GenerationResult", "Provider", "build_provider"]


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "mock":
        return MockProvider(step_delay=settings.mock_step_delay)
    if settings.provider == "fal":
        return FalProvider(
            api_key=settings.fal_key,
            image_to_3d_model=settings.fal_image_to_3d_model,
            text_to_image_model=settings.fal_text_to_image_model,
            timeout_s=settings.fal_timeout_s,
        )
    raise ValueError(f"Proveedor desconocido: {settings.provider!r} (usa 'mock' o 'fal')")
