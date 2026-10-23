"""Provider interface: anything that turns prompts or images into 3D meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

# progress (0-100) and a short human readable stage label
ProgressCallback = Callable[[int, str], Awaitable[None]]

# What a provider can do; the UI hides features the active provider lacks.
CAPABILITIES = ("text", "image", "multi_image", "concepts", "retexture", "depth", "upscale")


@dataclass
class InputImage:
    data: bytes
    mime: str = "image/png"


@dataclass
class GenerationRequest:
    mode: str  # "text" | "image" | "multi_image"
    prompt: str | None
    quality: str  # "draft" | "standard" | "high"
    images: list[InputImage] = field(default_factory=list)
    style: str = "auto"
    symmetry: bool = False
    pose: str = "none"  # "none" | "a-pose" | "t-pose"
    seed: int = 0
    engine: str | None = None  # provider-specific 3D engine (None = provider default)
    # which view each image is ("front", "left", "back", "right"); None = that order
    roles: list[str] | None = None
    # position in a set of shape variants of the same input (0 = the default way of building it):
    # providers may build each one differently, not only with another seed
    variant: int = 0
    # production options applied to the finished model (Meshy-style): polygon budget, topology, PBR maps
    target_faces: int | None = None
    topology: str = "triangle"  # "triangle" | "quad"
    pbr: bool = False


@dataclass
class GenerationResult:
    model_glb: bytes
    # Reference image produced along the way (e.g. text -> image step), if any.
    reference_image: bytes | None = None
    reference_image_ext: str = "png"
    # other files to keep next to the model, e.g. {"depth.png": ...}
    extra_files: dict[str, bytes] = field(default_factory=dict)
    # AI-generated views of a single picture by role ("left", "back", "right"), used to paint the model,
    # and the (azimuth, elevation) they were actually seen from when not square on the axes
    views: dict[str, bytes] = field(default_factory=dict)
    view_poses: dict[str, tuple[float, float]] = field(default_factory=dict)
    # the input pictures as clean cut-outs (same order), so painting uses the exact mask the shape saw
    inputs: list[bytes] = field(default_factory=list)


class GenerationError(Exception):
    """Raised by providers with a message that is safe to show to the user."""


class Provider(Protocol):
    name: str
    capabilities: frozenset[str]

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult: ...

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        """Concept images for a prompt (text -> image), used to pick a design before going 3D."""
        ...

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        """Repaint an existing model from a text prompt, keeping its geometry."""
        ...

    async def estimate_depth(self, image: InputImage) -> bytes:
        """Monocular depth map as a grayscale PNG, brighter = closer (scenes and reliefs)."""
        ...

    async def upscale_image(self, png: bytes, scale: int) -> bytes:
        """AI super-resolution of a texture (Real-ESRGAN), returns PNG/JPEG bytes."""
        ...
