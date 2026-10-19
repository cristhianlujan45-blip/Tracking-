"""Art styles and prompt building shared by every provider."""

from __future__ import annotations

from .providers.base import GenerationRequest

# key: (label, prompt modifier for the image model)
STYLES = {
    "auto": ("Automático", ""),
    "realistic": ("Realista", "photorealistic, physically based materials, highly detailed surfaces"),
    "cartoon": ("Cartoon", "stylized cartoon, vibrant saturated colors, smooth clean shapes"),
    "handpainted": ("Pintado a mano", "hand painted textures, stylized fantasy game asset"),
    "lowpoly": ("Low-poly", "low poly game asset, flat shaded facets, simple geometry"),
    "voxel": ("Vóxel", "voxel art, built from small cubes"),
    "sculpture": ("Escultura", "clay sculpture, single matte material, no textures"),
}

POSES = {
    "none": ("Libre", ""),
    "a-pose": ("A-pose", "full body character standing in A-pose, arms angled down away from the body"),
    "t-pose": ("T-pose", "full body character standing in T-pose, arms stretched out horizontally"),
}

IMAGE_PROMPT_SUFFIX = (
    "single object, centered, full view, isometric 3/4 angle, plain white background, "
    "soft studio lighting, high detail 3D render"
)


def build_image_prompt(req: GenerationRequest) -> str:
    parts = [(req.prompt or "").strip(), STYLES.get(req.style, STYLES["auto"])[1], POSES.get(req.pose, POSES["none"])[1]]
    if req.symmetry:
        parts.append("perfectly symmetrical design")
    parts.append(IMAGE_PROMPT_SUFFIX)
    return ", ".join(p for p in parts if p)
