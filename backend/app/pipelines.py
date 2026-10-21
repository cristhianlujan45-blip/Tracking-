"""Builders for the units of work the job runner executes."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from pathlib import Path
from typing import Callable

from . import creations, geometry, projection, publishing
from .jobs import Work
from .providers import GenerationRequest, GenerationResult, InputImage, Provider
from .providers.base import ProgressCallback

log = logging.getLogger(__name__)

# generated views are plausible, not exact: real pictures win wherever both see the surface
GENERATED_VIEW_TRUST = 0.6


def generation(provider: Provider, req: GenerationRequest, preview: bool = False) -> Work:
    """`preview`: also render a gallery picture (variants are compared before they are opened)."""

    async def work(on_progress: ProgressCallback) -> GenerationResult:
        started = time.monotonic()
        result = await provider.generate(req, on_progress)
        log.info("Forma lista en %.0f s", time.monotonic() - started)
        if getattr(provider, "paints_from_views", False):
            pictures = result.inputs or [img.data for img in req.images] or (
                [result.reference_image] if result.reference_image else [])
            if pictures:
                roles = list(req.roles) if req.roles and len(req.roles) == len(pictures) else projection.ROLES[:len(pictures)]
                trust = {}
                for role, data in result.views.items():  # AI side/back views of a single picture
                    if role not in roles:
                        roles, pictures = [*roles, role], [*pictures, data]
                        trust[role] = GENERATED_VIEW_TRUST
                await on_progress(97, "Pintando el modelo desde las imágenes")
                painting = time.monotonic()
                try:
                    result.model_glb = await asyncio.to_thread(
                        projection.paint_glb, result.model_glb, pictures, roles, trust, result.view_poses)
                    log.info("Pintado en %.0f s", time.monotonic() - painting)
                except projection.ProjectionError as exc:  # keep the shape: an unpainted model beats no model
                    log.warning("No se pudo pintar el modelo: %s", exc)
                except Exception:  # noqa: BLE001 - same, for anything unexpected (logged for the diagnostics)
                    log.exception("Error inesperado al pintar el modelo; se deja sin pintar")
        if req.style in ("lowpoly", "voxel", "sculpture"):
            await on_progress(96, "Aplicando estilo")
            result.model_glb = await asyncio.to_thread(geometry.apply_style, result.model_glb, req.style)
        if preview:
            try:
                views = await asyncio.to_thread(publishing.render_views, result.model_glb, 320, ("tres_cuartos",))
                result.extra_files["preview.png"] = views["tres_cuartos"]
            except Exception:  # noqa: BLE001 - only a picture: the app takes one when the model is opened
                log.exception("No se pudo crear la vista previa")
        return result

    return work


def local(glb_path: Path, label: str, fn: Callable[..., bytes], **kwargs) -> Work:
    """CPU geometry operation on an existing model, off the event loop."""

    async def work(on_progress: ProgressCallback) -> GenerationResult:
        await on_progress(15, label)
        glb = await asyncio.to_thread(partial(fn, glb_path.read_bytes(), **kwargs))
        return GenerationResult(model_glb=glb)

    return work


def retexture(provider: Provider, glb_path: Path, req: GenerationRequest) -> Work:
    async def work(on_progress: ProgressCallback) -> GenerationResult:
        glb = await provider.retexture(glb_path.read_bytes(), req, on_progress)
        return GenerationResult(model_glb=glb)

    return work


def upload(data: bytes, file_type: str) -> Work:
    async def work(on_progress: ProgressCallback) -> GenerationResult:
        await on_progress(30, "Convirtiendo a GLB")
        return GenerationResult(model_glb=await asyncio.to_thread(geometry.convert_upload, data, file_type))

    return work


def scene(provider: Provider, image: InputImage, kind: str, **opts) -> Work:
    """Photo → scene / relief (needs depth) or lithophane (brightness only)."""

    async def work(on_progress: ProgressCallback) -> GenerationResult:
        if kind == "lithophane":
            await on_progress(20, "Calculando grosores")
            glb = await asyncio.to_thread(partial(creations.lithophane, image.data, **opts))
            return GenerationResult(model_glb=glb)
        if "depth" in provider.capabilities:
            await on_progress(10, "Estimando profundidad con IA")
            depth = await provider.estimate_depth(image)
        else:
            await on_progress(10, "Estimando profundidad")
            depth = await asyncio.to_thread(creations.pseudo_depth, image.data)
        await on_progress(60, "Construyendo malla")
        fn = creations.depth_scene if kind == "scene" else creations.relief
        glb = await asyncio.to_thread(partial(fn, image.data, depth, **opts))
        return GenerationResult(model_glb=glb, extra_files={"depth.png": depth})

    return work


def compute(label: str, fn: Callable[..., bytes], **kwargs) -> Work:
    """CPU-only creation from parameters (e.g. 3D text)."""

    async def work(on_progress: ProgressCallback) -> GenerationResult:
        await on_progress(20, label)
        return GenerationResult(model_glb=await asyncio.to_thread(partial(fn, **kwargs)))

    return work
