"""HTTP API of the local GPU worker.

The GPU runs one thing at a time: every model call goes through a single-thread executor, and
mesh tasks are queued and polled so the main app can show real progress.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import VERSION
from .backends import Backend, MeshJob, png_bytes

log = logging.getLogger(__name__)
TASK_TTL_S = 3600
MAX_IMAGE_BYTES = 15 * 1024 * 1024


class ImagesIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    count: int = Field(1, ge=1, le=4)
    seed: int = Field(0, ge=0, le=2**32 - 1)
    negative: str = ""
    subject: str = ""  # the user's own words inside prompt: translated to English when they are not


class ChatIn(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=40)
    max_tokens: int = Field(400, ge=16, le=1500)


class DepthIn(BaseModel):
    image: str  # base64


class UpscaleIn(BaseModel):
    image: str  # base64
    scale: int = Field(4, ge=2, le=4)


class ViewsIn(BaseModel):
    image: str  # base64, the front picture
    quality: Literal["draft", "standard", "high"] = "standard"
    seed: int = Field(0, ge=0, lt=2**32)  # another seed imagines other sides and back (shape variants)


class TaskIn(BaseModel):
    kind: Literal["mesh", "texture"] = "mesh"
    images: list[str] = Field(min_length=1, max_length=4)  # base64 (PNG/JPG/WEBP)
    # view of each image; default: front, left, back, right in that order
    roles: list[Literal["front", "left", "back", "right"]] | None = None
    seed: int = Field(0, ge=0, le=2**32 - 1)
    quality: Literal["draft", "standard", "high"] = "standard"
    mesh_glb: str | None = None  # base64, required for kind="texture"


@dataclass
class Task:
    id: str
    status: str = "queued"  # queued | running | succeeded | failed
    progress: int = 0
    stage: str = "En cola"
    error: str | None = None
    result: bytes | None = None
    created: float = field(default_factory=time.time)


def friendly_error(exc: Exception) -> str:
    text = str(exc).lower()
    if getattr(exc, "winerror", None) == 1455 or "paginación" in text or "paging file" in text:
        return ("Windows se quedó sin memoria (RAM + archivo de paginación). Cierra programas abiertos "
                "(navegador, juegos) o aumenta el archivo de paginación: Configuración → Sistema → Información → "
                "Configuración avanzada del sistema → Rendimiento → Opciones avanzadas → Memoria virtual → "
                "«Administrar automáticamente».")
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text:
        return "Sin memoria de GPU (VRAM). Prueba calidad Borrador o cierra otros programas que usen la GPU."
    return f"{type(exc).__name__}: {exc}"[:500]


def decode_image(data: str) -> Image.Image:
    raw = base64.b64decode(data.split(",", 1)[-1], validate=False)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Imagen demasiado grande")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Imagen no válida") from exc
    return ImageOps.exif_transpose(img)  # phone photos store their rotation separately


def create_app(backend: Backend, token: str | None = None) -> FastAPI:
    token = token if token is not None else os.getenv("FORJA_WORKER_TOKEN") or None
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
    tasks: dict[str, Task] = {}
    lock = threading.Lock()

    def check_token(request: Request) -> None:
        if token and request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "Token del worker incorrecto")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        gpu.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Forja3D GPU worker", dependencies=[Depends(check_token)], lifespan=lifespan)

    def prune() -> None:
        now = time.time()
        with lock:
            for task_id in [t.id for t in tasks.values() if now - t.created > TASK_TTL_S and t.status in ("succeeded", "failed")]:
                del tasks[task_id]

    def queue_position(task: Task) -> int:
        with lock:
            waiting = sorted((t for t in tasks.values() if t.status == "queued"), key=lambda t: t.created)
        return next((i for i, t in enumerate(waiting) if t.id == task.id), -1) + 1

    def run_task(task: Task, job: MeshJob, kind: str) -> None:
        def progress(pct: int, stage: str) -> None:
            if stage != task.stage:  # how long each stage takes, for the diagnostics
                log.info("[%s] %5.1f s · %s", task.id[:6], time.time() - started, stage)
            task.progress, task.stage = max(task.progress, min(99, pct)), stage

        started = time.time()
        task.status, task.stage, task.progress = "running", "Iniciando", 1
        try:
            fn = backend.texture_mesh if kind == "texture" else backend.image_to_mesh
            task.result = fn(job, progress)
            task.status, task.progress, task.stage = "succeeded", 100, "Completado"
            log.info("Tarea %s lista en %.1f s", task.id, time.time() - started)
        except Exception as exc:  # noqa: BLE001 - report every failure to the caller
            log.exception("Tarea %s falló", task.id)
            task.status, task.error = "failed", friendly_error(exc)

    @app.get("/health")
    async def health():
        with lock:
            busy = sum(t.status in ("queued", "running") for t in tasks.values())
        return {"status": "ok", "backend": backend.name, "capabilities": sorted(backend.capabilities()),
                "busy": busy, "version": VERSION, **backend.info()}

    @app.post("/shutdown")
    async def shutdown():
        """The app replaces a worker left running by an older version (e.g. after an update)."""
        threading.Timer(0.3, os._exit, args=(0,)).start()
        return {"status": "bye"}

    @app.post("/images")
    async def images(body: ImagesIn):
        if "concepts" not in backend.capabilities():
            raise HTTPException(400, "Este worker no genera imágenes")
        loop = asyncio.get_running_loop()
        try:
            prompt = body.prompt
            if body.subject and body.subject in prompt and hasattr(backend, "to_english"):
                english = await loop.run_in_executor(gpu, backend.to_english, body.subject)
                prompt = prompt.replace(body.subject, english, 1)
            result = await loop.run_in_executor(gpu, backend.generate_images, prompt, body.count, body.seed, body.negative)
        except Exception as exc:  # noqa: BLE001
            log.exception("Generación de imágenes falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        return {"images": [base64.b64encode(png_bytes(img)).decode() for img in result]}

    @app.post("/cutout")
    async def cutout(body: DepthIn):
        """Background removal (BiRefNet + clean-up): the app paints with the same mask the shape used."""
        if "cutout" not in backend.capabilities():
            raise HTTPException(400, "Este worker no recorta fondos")
        image = decode_image(body.image)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(gpu, backend.cutout, image)
            result.thumbnail((2048, 2048))  # plenty for the models; keeps the PNG under the upload limit
        except Exception as exc:  # noqa: BLE001
            log.exception("Recorte falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        return {"image": base64.b64encode(png_bytes(result)).decode()}

    @app.post("/chat")
    async def chat(body: ChatIn):
        if "chat" not in backend.capabilities():
            raise HTTPException(400, "Este worker no tiene modelo de texto")
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(gpu, backend.chat, body.messages, body.max_tokens)
        except Exception as exc:  # noqa: BLE001
            log.exception("Chat falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        return {"text": text}

    @app.post("/depth")
    async def depth(body: DepthIn):
        if "depth" not in backend.capabilities():
            raise HTTPException(400, "Este worker no estima profundidad")
        image = decode_image(body.image)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(gpu, backend.estimate_depth, image)
        except Exception as exc:  # noqa: BLE001
            log.exception("Estimación de profundidad falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        return {"depth": base64.b64encode(png_bytes(result)).decode()}

    @app.post("/upscale")
    async def upscale(body: UpscaleIn):
        if "upscale" not in backend.capabilities():
            raise HTTPException(400, "Este worker no amplía texturas")
        image = decode_image(body.image)
        if max(image.size) * body.scale > 8192:
            raise HTTPException(413, "El resultado superaría 8192 px")
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(gpu, backend.upscale, image, body.scale)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ampliación falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        return {"image": base64.b64encode(png_bytes(result)).decode()}

    @app.post("/views")
    async def views(body: ViewsIn):
        """Side and back views of a single front picture, plus the (azimuth, elevation) each was seen from."""
        if "views" not in backend.capabilities():
            raise HTTPException(400, "Este worker no genera vistas laterales")
        image = decode_image(body.image)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(gpu, backend.novel_views, image, body.quality, body.seed)
        except Exception as exc:  # noqa: BLE001
            log.exception("Generación de vistas falló")
            raise HTTPException(500, friendly_error(exc)) from exc
        poses = getattr(backend, "view_poses", {})
        return {"views": {role: base64.b64encode(png_bytes(img)).decode() for role, img in result.items()},
                "poses": {role: list(poses[role]) for role in result if role in poses}}

    @app.post("/tasks", status_code=202)
    async def create_task(body: TaskIn):
        prune()
        if body.kind == "texture":
            if "retexture" not in backend.capabilities():
                raise HTTPException(400, "El texturizado está desactivado en este worker")
            if not body.mesh_glb:
                raise HTTPException(400, "Falta mesh_glb")
        if body.roles is not None and (len(body.roles) != len(body.images) or len(set(body.roles)) != len(body.roles)):
            raise HTTPException(400, "roles debe dar una vista distinta para cada imagen")
        if len(body.images) > 1 and "multi_image" not in backend.capabilities():
            body.images, body.roles = body.images[:1], None
        job = MeshJob(
            images=[decode_image(i) for i in body.images], seed=body.seed, quality=body.quality,
            mesh_glb=base64.b64decode(body.mesh_glb) if body.mesh_glb else None, roles=body.roles,
        )
        task = Task(id=uuid.uuid4().hex)
        with lock:
            tasks[task.id] = task
        gpu.submit(run_task, task, job, body.kind)
        return {"id": task.id}

    def get_task(task_id: str) -> Task:
        with lock:
            task = tasks.get(task_id)
        if not task:
            raise HTTPException(404, "Tarea no encontrada")
        return task

    @app.get("/tasks/{task_id}")
    async def task_status(task_id: str):
        task = get_task(task_id)
        stage = task.stage
        if task.status == "queued":
            stage = f"En cola en la GPU (posición {queue_position(task)})"
        return {"id": task.id, "status": task.status, "progress": task.progress, "stage": stage, "error": task.error}

    @app.get("/tasks/{task_id}/result")
    async def task_result(task_id: str):
        task = get_task(task_id)
        if task.status != "succeeded" or task.result is None:
            raise HTTPException(409, "La tarea no ha terminado")
        return Response(task.result, media_type="model/gltf-binary")

    @app.delete("/tasks/{task_id}", status_code=204)
    async def delete_task(task_id: str):
        task = get_task(task_id)
        if task.status in ("succeeded", "failed"):
            with lock:
                tasks.pop(task_id, None)

    return app
