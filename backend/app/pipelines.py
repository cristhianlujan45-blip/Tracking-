"""Builders for the units of work the job runner executes."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Callable

from . import creations, geometry
from .jobs import Work
from .providers import GenerationRequest, GenerationResult, InputImage, Provider
from .providers.base import ProgressCallback


def generation(provider: Provider, req: GenerationRequest) -> Work:
    async def work(on_progress: ProgressCallback) -> GenerationResult:
        result = await provider.generate(req, on_progress)
        if req.style in ("lowpoly", "voxel", "sculpture"):
            await on_progress(96, "Aplicando estilo")
            result.model_glb = await asyncio.to_thread(geometry.apply_style, result.model_glb, req.style)
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
