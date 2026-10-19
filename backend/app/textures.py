"""Texture enhancement: HD upscaling and PBR maps derived from the base colour.

* Upscaling uses the provider's AI upscaler (Real-ESRGAN on your GPU or on fal.ai) when available,
  otherwise a Lanczos + unsharp-mask fallback on the CPU.
* PBR maps are estimated from the albedo the way tools like Materialize do it: a height field from
  luminance gives the normal map and ambient occlusion; roughness follows local contrast. They are
  an approximation, labelled as such, for engines that expect a full PBR set.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, sobel

from .geometry import GeometryError, load_scene

TARGETS = {"2k": 2048, "4k": 4096, "8k": 8192}
Upscaler = Callable[[bytes, int], Awaitable[bytes]]


@dataclass
class TextureReport:
    textures: int
    before: list[tuple[int, int]]
    after: list[tuple[int, int]]
    method: str
    pbr: bool


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def cpu_upscale(image: Image.Image, scale: int) -> Image.Image:
    big = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    return big.filter(ImageFilter.UnsharpMask(radius=1.6, percent=80, threshold=2))


def _fit(image: Image.Image, limit: int) -> Image.Image:
    if max(image.size) <= limit:
        return image
    ratio = limit / max(image.size)
    return image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.LANCZOS)


# ---------------------------------------------------------------- PBR from albedo


def pbr_maps(albedo: Image.Image, strength: float = 2.5) -> tuple[Image.Image, Image.Image, Image.Image]:
    """(normal, metallicRoughness, occlusion) maps for a base-colour texture."""
    rgb = np.asarray(albedo.convert("RGB"), dtype=np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    height = gaussian_filter(lum, 1.0)

    dx = sobel(height, axis=1)
    dy = sobel(height, axis=0)
    normal = np.dstack([-dx * strength, dy * strength, np.ones_like(height)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    normal_img = Image.fromarray(((normal * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")

    # cavities (darker than their surroundings) get occlusion
    cavity = np.clip((gaussian_filter(height, 6) - height) * 4.0, 0, 1)
    ao = 1.0 - 0.7 * cavity
    # rough where the surface is busy, smoother on flat, bright areas
    detail = gaussian_filter(np.abs(height - gaussian_filter(height, 2)), 2)
    roughness = np.clip(0.55 + detail * 6.0 - 0.2 * (lum - 0.5), 0.3, 0.95)
    metallic = np.zeros_like(roughness)
    # glTF reads roughness from G and metallic from B; R carries occlusion for engines that use ORM maps
    orm = np.dstack([ao, roughness, metallic])
    orm_img = Image.fromarray((orm * 255).astype(np.uint8), "RGB")
    occlusion_img = orm_img
    return normal_img, orm_img, occlusion_img


# ---------------------------------------------------------------- GLB pass


def textured_images(glb: bytes) -> list[Image.Image]:
    scene = load_scene(glb)
    out = []
    for geom in scene.geometry.values():
        material = getattr(geom.visual, "material", None)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if image is not None and all(image is not o for o in out):
            out.append(image)
    return out


async def enhance(glb: bytes, target: str = "4k", scale: int = 4, add_pbr: bool = True,
                  upscaler: Upscaler | None = None, method_label: str = "IA") -> tuple[bytes, TextureReport]:
    """Upscale every base-colour texture towards `target` and optionally add PBR maps."""
    scene = load_scene(glb)
    limit = TARGETS.get(target, 4096)
    cache: dict[int, tuple[Image.Image, Image.Image | None, Image.Image | None]] = {}
    before, after = [], []
    method = method_label if upscaler else "CPU (Lanczos)"

    for geom in scene.geometry.values():
        visual = geom.visual
        material = getattr(visual, "material", None)
        if visual.kind != "texture" or material is None:
            continue
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if image is None:
            continue
        key = id(image)
        if key not in cache:
            src = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
            before.append(src.size)
            factor = max(1, min(scale, limit // max(src.size)))
            if factor > 1 and upscaler is not None:
                try:
                    big = Image.open(io.BytesIO(await upscaler(_png(src.convert("RGB")), factor))).convert("RGB")
                except Exception:  # noqa: BLE001 - fall back rather than lose the job
                    big, method = cpu_upscale(src.convert("RGB"), factor), "CPU (Lanczos)"
                if src.mode == "RGBA":  # keep transparency: upscale alpha separately
                    alpha = src.getchannel("A").resize(big.size, Image.LANCZOS)
                    big.putalpha(alpha)
            elif factor > 1:
                big = cpu_upscale(src, factor)
            else:
                big = src
            big = _fit(big, limit)
            after.append(big.size)
            normal = orm = None
            if add_pbr:
                normal, orm, _ = pbr_maps(big)
            cache[key] = (big, normal, orm)
        big, normal, orm = cache[key]
        pbr = trimesh.visual.material.PBRMaterial(
            name=getattr(material, "name", None),
            baseColorTexture=big,
            baseColorFactor=getattr(material, "baseColorFactor", None),
            metallicFactor=0.0 if add_pbr else getattr(material, "metallicFactor", 0.0),
            roughnessFactor=1.0 if add_pbr else getattr(material, "roughnessFactor", 0.8),
            normalTexture=normal,
            metallicRoughnessTexture=orm,
            doubleSided=getattr(material, "doubleSided", False),
        )
        geom.visual = trimesh.visual.TextureVisuals(uv=visual.uv, material=pbr)

    if not cache:
        raise GeometryError("Este modelo no tiene texturas (usa colores por vértice). Retexturízalo primero.")
    data = scene.export(file_type="glb", include_normals=True)
    return data, TextureReport(len(cache), before, after, method, add_pbr)
