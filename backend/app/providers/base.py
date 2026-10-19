"""Provider interface: anything that turns a prompt or an image into a 3D mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

# progress (0-100) and a short human readable stage label
ProgressCallback = Callable[[int, str], Awaitable[None]]


@dataclass
class GenerationRequest:
    mode: str  # "text" | "image"
    prompt: str | None
    quality: str  # "draft" | "standard" | "high"
    image_bytes: bytes | None = None
    image_mime: str | None = None


@dataclass
class GenerationResult:
    model_glb: bytes
    # Reference image produced along the way (e.g. text -> image step), if any.
    reference_image: bytes | None = None
    reference_image_ext: str = "png"


class GenerationError(Exception):
    """Raised by providers with a message that is safe to show to the user."""


class Provider(Protocol):
    name: str

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult: ...
