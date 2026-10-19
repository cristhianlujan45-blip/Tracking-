"""HTTP API + static frontend for the AI 3D generator."""

from __future__ import annotations

import asyncio
import io
import os
import json
from dataclasses import replace
import logging
import random
import re
import shutil
import unicodedata
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import blender, creations, geometry, pipelines, rigging, textures
from .config import Settings, get_settings, save_user_settings, user_settings_path, with_user_settings
from .db import JobStore
from .jobs import JobRunner
from .providers import GenerationError, GenerationRequest, GenerationResult, InputImage, Provider, build_provider
from .providers.fal import ENGINES as FAL_ENGINES
from .styles import POSES, STYLES

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
BLENDER_ADDON = Path(__file__).resolve().parents[2] / "integrations" / "blender" / "forja3d_blender.py"
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
MIME_BY_EXT = {ext: mime for mime, ext in IMAGE_TYPES.items()}
GENERATIVE_MODES = ("text", "image", "multi_image")
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
HEX_ID = re.compile(r"^[0-9a-f]{32}$")

Quality = Literal["draft", "standard", "high"]
Style = Literal[tuple(STYLES)]  # type: ignore[valid-type]
Pose = Literal[tuple(POSES)]  # type: ignore[valid-type]
ExportFormat = Literal[tuple(geometry.EXPORT_FORMATS) + tuple(blender.FORMATS)]  # type: ignore[valid-type]

MODE_LABELS = {
    "text": "Texto a 3D",
    "image": "Imagen a 3D",
    "multi_image": "Multi-imagen a 3D",
    "upload": "Modelo subido",
    "remesh": "Remallado",
    "retexture": "Retexturizado",
    "repair": "Reparado",
    "split": "Dividido",
    "scene": "Escena 3D",
    "relief": "Relieve",
    "lithophane": "Litofanía",
    "text3d": "Texto 3D",
    "rig": "Personaje animado",
    "enhance": "Textura HD",
}
SCENE_KINDS = ("scene", "relief", "lithophane")
HEX_COLOR = r"^#[0-9a-fA-F]{6}$"
REMESH_LABELS = {"decimate": "reducido", "lowpoly": "low-poly", "voxel": "vóxel", "smooth": "suavizado",
                 "quads": "quads"}

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------- schemas


class GenerationOptions(BaseModel):
    quality: Quality = "standard"
    style: Style = "auto"
    symmetry: bool = False
    pose: Pose = "none"
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    variants: int = Field(1, ge=1, le=4)
    engine: str | None = Field(None, max_length=40)


class TextJobIn(GenerationOptions):
    prompt: str = Field(min_length=3, max_length=500)


class ConceptsIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    style: Style = "auto"
    symmetry: bool = False
    pose: Pose = "none"
    count: int = Field(4, ge=1, le=4)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class JobPatch(BaseModel):
    name: str | None = Field(None, max_length=120)
    favorite: bool | None = None


class RemeshIn(BaseModel):
    mode: Literal["decimate", "lowpoly", "voxel", "smooth", "quads"] = "decimate"
    target_faces: int = Field(5000, ge=100, le=500_000)
    resolution: int = Field(32, ge=8, le=96)


class SplitIn(BaseModel):
    mode: Literal["height", "components"] = "height"
    parts: int = Field(2, ge=2, le=8)


class RetextureIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    style: Style = "auto"
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    resolution: Literal["2k", "4k", "8k"] = "4k"


class EnhanceIn(BaseModel):
    target: Literal["2k", "4k", "8k"] = "4k"
    pbr: bool = True


class VariationsIn(BaseModel):
    count: int = Field(3, ge=1, le=4)


class Text3DIn(BaseModel):
    text: str = Field(min_length=1, max_length=80)
    font: Literal["sans", "serif", "mono"] = "sans"
    bold: bool = True
    style: Literal["letters", "plate", "keychain"] = "letters"
    depth: float = Field(0.25, ge=0.02, le=2.0)
    color: str = Field("#7c5cff", pattern=HEX_COLOR)
    plate_color: str = Field("#f0f0f0", pattern=HEX_COLOR)


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


class SettingsIn(BaseModel):
    provider: Literal["mock", "local", "fal"] | None = None
    fal_key: str | None = Field(None, max_length=200)
    fal_engine: str | None = Field(None, max_length=40)
    local_worker_url: str | None = Field(None, max_length=300)
    local_worker_token: str | None = Field(None, max_length=200)
    blender_path: str | None = Field(None, max_length=500)


class RigIn(BaseModel):
    template: Literal["auto", "humanoid", "quadruped"] = "auto"
    markers: dict[str, list[float]] | None = None
    yaw: Literal[0, 90, 180, 270] | None = None
    symmetric: bool = False
    animations: list[str] = Field(default_factory=lambda: list(rigging.DEFAULT_PRESETS), max_length=40)
    in_place: bool = True


class CharacterIn(BaseModel):
    target: Literal["none", "2k", "4k", "8k"] = "4k"
    pbr: bool = True
    template: Literal["auto", "humanoid", "quadruped"] = "auto"
    animations: list[str] = Field(default_factory=lambda: list(rigging.DEFAULT_PRESETS), max_length=40)
    in_place: bool = True


class MarkersIn(BaseModel):
    template: Literal["auto", "humanoid", "quadruped"] = "auto"
    yaw: Literal[0, 90, 180, 270] | None = None
    auto: bool = False  # ignore a rigged character's saved markers and estimate again


class JobOut(BaseModel):
    id: str
    mode: str
    title: str
    name: str | None
    prompt: str | None
    quality: str
    status: str
    progress: int
    stage: str | None
    error: str | None
    provider: str
    params: dict
    parent_id: str | None
    favorite: bool
    created_at: float
    updated_at: float
    model_url: str | None
    image_url: str | None
    thumbnail_url: str | None
    depth_url: str | None = None


def job_title(job: dict) -> str:
    return job.get("name") or job.get("prompt") or MODE_LABELS.get(job["mode"], job["mode"])


def to_out(job: dict) -> JobOut:
    base = f"/api/jobs/{job['id']}"
    version = int(job["updated_at"])
    return JobOut(
        **{k: job.get(k) for k in JobOut.model_fields if k in job},
        title=job_title(job),
        model_url=f"{base}/model.glb?v={version}" if job["model_file"] else None,
        image_url=f"{base}/image" if job["input_image"] else None,
        thumbnail_url=f"{base}/thumbnail?v={version}" if job.get("thumbnail") else None,
        depth_url=f"{base}/depth" if job["params"].get("has_depth") and job["status"] == "succeeded" else None,
    )


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()[:60] or "modelo"


def new_seed() -> int:
    return random.randint(0, 2**31 - 1)


# ---------------------------------------------------------------- app


def create_app(settings: Settings | None = None, provider: Provider | None = None) -> FastAPI:
    settings = with_user_settings(settings) if settings else get_settings()
    concepts_dir = settings.files_dir / "concepts"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        concepts_dir.mkdir(parents=True, exist_ok=True)
        store = JobStore(settings.db_path)
        store.fail_interrupted()
        app.state.store = store
        app.state.settings = settings
        app.state.provider_error = None
        try:
            app.state.provider = provider or build_provider(settings)
        except (GenerationError, ValueError) as exc:  # e.g. cloud mode without a key: start in demo mode
            app.state.provider_error = str(exc)
            app.state.provider = build_provider(replace(settings, provider="mock"))
        if hasattr(app.state.provider, "start"):
            await app.state.provider.start()
        app.state.runner = JobRunner(store, settings.files_dir, settings.max_concurrent_jobs)
        app.state.blender = blender.find_blender(settings.blender_path)
        yield
        await app.state.runner.shutdown()
        store.close()

    app = FastAPI(title="Forja3D", version="0.2.0", lifespan=lifespan)

    # ------------------------------------------------------------ helpers

    def job_dir(job_id: str) -> Path:
        return settings.files_dir / job_id

    def get_job_or_404(request: Request, job_id: str) -> dict:
        job = request.app.state.store.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return job

    def ready_model(request: Request, job_id: str) -> tuple[dict, Path]:
        job = get_job_or_404(request, job_id)
        if job["status"] != "succeeded" or not job["model_file"]:
            raise HTTPException(409, "El modelo todavía no está listo")
        return job, job_dir(job_id) / job["model_file"]

    def require(request: Request, capability: str, label: str) -> None:
        if capability not in request.app.state.provider.capabilities:
            raise HTTPException(400, f"{label} no está disponible con el proveedor «{request.app.state.provider.name}»")

    def load_inputs(job: dict) -> list[InputImage]:
        return [
            InputImage((job_dir(job["id"]) / name).read_bytes(), MIME_BY_EXT.get(name.rsplit(".", 1)[-1], "image/png"))
            for name in job["params"].get("inputs", [])
        ]

    def create_generation_jobs(
        request: Request,
        mode: str,
        prompt: str | None,
        opts: GenerationOptions,
        images: list[InputImage],
        extra: dict | None = None,
        parent_id: str | None = None,
    ) -> list[JobOut]:
        state = request.app.state
        engines = [e["id"] for e in getattr(state.provider, "engines", [])]
        if opts.engine and opts.engine not in engines:
            raise HTTPException(422, f"Motor desconocido: {opts.engine}")
        base_seed = opts.seed if opts.seed is not None else new_seed()
        jobs = []
        for i in range(opts.variants):
            seed = (base_seed + i) % 2**31
            job_id = uuid.uuid4().hex
            inputs = []
            if images:
                job_dir(job_id).mkdir(parents=True)
                for n, img in enumerate(images):
                    name = f"input_{n}.{IMAGE_TYPES.get(img.mime, 'png')}"
                    (job_dir(job_id) / name).write_bytes(img.data)
                    inputs.append(name)
            params = {
                "style": opts.style, "symmetry": opts.symmetry, "pose": opts.pose, "seed": seed, "engine": opts.engine,
                "inputs": inputs, **(extra or {}),
            }
            job = state.store.create(
                id=job_id, mode=mode, prompt=prompt, quality=opts.quality, provider=state.provider.name,
                input_image=inputs[0] if inputs else None, params=params, parent_id=parent_id,
            )
            req = GenerationRequest(
                mode=mode, prompt=prompt, quality=opts.quality, images=images,
                style=opts.style, symmetry=opts.symmetry, pose=opts.pose, seed=seed, engine=opts.engine,
            )
            state.runner.submit(job_id, pipelines.generation(state.provider, req))
            jobs.append(to_out(job))
        return jobs

    def create_derived_job(request: Request, parent: dict, mode: str, work, name: str, params: dict,
                           prompt: str | None = None) -> JobOut:
        state = request.app.state
        job = state.store.create(
            id=uuid.uuid4().hex, mode=mode, prompt=prompt, quality=parent["quality"], provider=state.provider.name,
            params=params, parent_id=parent["id"], name=name[:120],
        )
        state.runner.submit(job["id"], work)
        return to_out(job)

    async def read_image(upload: UploadFile) -> InputImage:
        mime = upload.content_type or ""
        if mime not in IMAGE_TYPES:
            raise HTTPException(415, "Formato no soportado: usa PNG, JPG o WEBP")
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"La imagen supera {settings.max_upload_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "La imagen está vacía")
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001 - PIL raises many types for corrupt files
            raise HTTPException(400, "La imagen está dañada o no es válida") from exc
        return InputImage(data, mime)

    # ------------------------------------------------------------ meta

    @app.get("/api/health")
    async def health(request: Request):
        return {"status": "ok", "provider": request.app.state.provider.name}

    @app.get("/api/config")
    async def config(request: Request):
        provider = request.app.state.provider
        if hasattr(provider, "start") and not provider.info:  # worker was down at startup: retry
            await provider.start()
        return {
            "provider": provider.name,
            "provider_info": getattr(provider, "info", {}),
            "engines": getattr(provider, "engines", []),
            "provider_error": request.app.state.provider_error,
            "default_engine": getattr(provider, "default_engine", None),
            "capabilities": sorted(provider.capabilities),
            "styles": [{"id": k, "label": v[0]} for k, v in STYLES.items()],
            "poses": [{"id": k, "label": v[0]} for k, v in POSES.items()],
            "export_formats": list(geometry.EXPORT_FORMATS) + (list(blender.FORMATS) if app.state.blender else []),
            "blender": bool(app.state.blender),
            "upload_types": sorted(geometry.UPLOAD_TYPES),
            "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
        }

    @app.get("/api/integrations/blender")
    async def blender_addon(request: Request):
        """The Blender add-on, preconfigured with the address this app is being reached at."""
        if not BLENDER_ADDON.is_file():
            raise HTTPException(404, "Complemento no disponible")
        source = BLENDER_ADDON.read_text(encoding="utf-8")
        base = str(request.base_url).rstrip("/").replace('"', "")
        source = source.replace('DEFAULT_SERVER = "http://localhost:8000"', f'DEFAULT_SERVER = "{base}"', 1)
        return Response(source, media_type="text/x-python",
                        headers={"Content-Disposition": 'attachment; filename="forja3d_blender.py"'})

    # ------------------------------------------------------------ settings (desktop app)

    @app.get("/api/settings")
    async def get_app_settings(request: Request):
        s = request.app.state.settings
        return {
            "provider": s.provider,
            "active_provider": request.app.state.provider.name,
            "provider_error": request.app.state.provider_error,
            "fal_key_set": bool(s.fal_key),
            "fal_key_hint": f"…{s.fal_key[-4:]}" if len(s.fal_key) >= 8 else "",
            "fal_engine": s.fal_engine,
            "fal_engines": [e.public() for e in FAL_ENGINES.values()],
            "local_worker_url": s.local_worker_url,
            "blender_path": s.blender_path,
            "blender_found": request.app.state.blender,
            "first_run": not user_settings_path(s).is_file(),
            "locked": settings_locked(),
        }

    def settings_locked() -> bool:
        return os.getenv("FORJA_LOCK_SETTINGS", "") not in ("", "0")

    @app.put("/api/settings")
    async def put_app_settings(request: Request, body: SettingsIn):
        if settings_locked():
            raise HTTPException(403, "Los ajustes están bloqueados en este servidor")
        state = request.app.state
        values = {k: v for k, v in body.model_dump().items() if v is not None}
        if values.get("fal_key") == "":
            values.pop("fal_key")  # empty field = keep the saved key
        candidate = replace(state.settings, **values)
        try:
            new_provider = build_provider(candidate)
        except (GenerationError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if hasattr(new_provider, "start"):
            await new_provider.start()
        state.settings = save_user_settings(state.settings, values)
        state.provider, state.provider_error = new_provider, None  # running jobs keep their own provider
        state.blender = blender.find_blender(state.settings.blender_path)
        return await get_app_settings(request)

    # ------------------------------------------------------------ concepts (text -> image)

    @app.post("/api/concepts")
    async def create_concepts(request: Request, body: ConceptsIn):
        require(request, "concepts", "La generación de conceptos")
        seed = body.seed if body.seed is not None else new_seed()
        req = GenerationRequest(
            mode="text", prompt=body.prompt.strip(), quality="standard",
            style=body.style, symmetry=body.symmetry, pose=body.pose, seed=seed,
        )
        try:
            images = await request.app.state.provider.generate_images(req, body.count)
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc
        concepts = []
        for data in images:
            concept_id = uuid.uuid4().hex
            (concepts_dir / f"{concept_id}.png").write_bytes(data)
            concepts.append({"id": concept_id, "url": f"/api/concepts/{concept_id}"})
        return {"seed": seed, "concepts": concepts}

    @app.get("/api/concepts/{concept_id}")
    async def get_concept(concept_id: str):
        path = concepts_dir / f"{concept_id}.png"
        if not HEX_ID.match(concept_id) or not path.is_file():
            raise HTTPException(404, "Concepto no encontrado")
        return FileResponse(path, media_type="image/png")

    # ------------------------------------------------------------ generation

    @app.post("/api/jobs/text", response_model=list[JobOut], status_code=201)
    async def create_text_job(request: Request, body: TextJobIn):
        require(request, "text", "Texto a 3D")
        return create_generation_jobs(request, "text", body.prompt.strip(), body, [])

    @app.post("/api/jobs/image", response_model=list[JobOut], status_code=201)
    async def create_image_job(
        request: Request,
        images: list[UploadFile] | None = File(None),
        concept_id: str | None = Form(None),
        prompt: str | None = Form(None, max_length=500),
        quality: Quality = Form("standard"),
        style: Style = Form("auto"),
        seed: int | None = Form(None, ge=0, le=2**31 - 1),
        variants: int = Form(1, ge=1, le=4),
        engine: str | None = Form(None, max_length=40),
    ):
        inputs: list[InputImage] = []
        extra = {}
        if concept_id:
            path = concepts_dir / f"{concept_id}.png"
            if not HEX_ID.match(concept_id) or not path.is_file():
                raise HTTPException(404, "Concepto no encontrado")
            inputs.append(InputImage(path.read_bytes(), "image/png"))
            extra["concept_id"] = concept_id
        for upload in images or []:
            inputs.append(await read_image(upload))
        if not inputs:
            raise HTTPException(400, "Sube al menos una imagen")
        if len(inputs) > 4:
            raise HTTPException(400, "Máximo 4 imágenes")
        mode = "multi_image" if len(inputs) > 1 else "image"
        require(request, mode, "Multi-imagen a 3D" if mode == "multi_image" else "Imagen a 3D")
        opts = GenerationOptions(quality=quality, style=style, seed=seed, variants=variants, engine=engine or None)
        return create_generation_jobs(request, mode, (prompt or "").strip() or None, opts, inputs, extra)

    @app.post("/api/jobs/upload", response_model=JobOut, status_code=201)
    async def upload_model(request: Request, model: UploadFile = File(...)):
        file_type = (model.filename or "").rsplit(".", 1)[-1].lower()
        if file_type not in geometry.UPLOAD_TYPES:
            raise HTTPException(415, "Formato no soportado: usa " + ", ".join(sorted(geometry.UPLOAD_TYPES)).upper())
        limit = settings.max_upload_bytes * 5
        data = await model.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"El modelo supera {limit // (1024 * 1024)} MB")
        state = request.app.state
        stem = (model.filename or "modelo").rsplit(".", 1)[0]
        job = state.store.create(
            id=uuid.uuid4().hex, mode="upload", prompt=None, provider="local",
            params={"original_format": file_type}, name=stem[:120],
        )
        state.runner.submit(job["id"], pipelines.upload(data, file_type))
        return to_out(job)

    # ------------------------------------------------------------ scenes, reliefs, lithophanes, 3D text

    @app.post("/api/jobs/scene", response_model=JobOut, status_code=201)
    async def create_scene_job(
        request: Request,
        image: UploadFile = File(...),
        kind: Literal["scene", "relief", "lithophane"] = Form("scene"),
        resolution: int = Form(256, ge=32, le=512),
        strength: float = Form(0.4, ge=0.02, le=2.0),
        cut_edges: bool = Form(True),
        name: str | None = Form(None, max_length=120),
    ):
        img = await read_image(image)
        state = request.app.state
        if kind == "scene":
            opts = {"resolution": resolution, "strength": strength, "cut_edges": cut_edges}
        elif kind == "relief":
            opts = {"resolution": min(resolution, 400), "strength": strength}
        else:
            opts = {"resolution": min(resolution, 400), "thickness_ratio": 0.03 * max(strength / 0.4, 0.25)}
        job_id = uuid.uuid4().hex
        job_dir(job_id).mkdir(parents=True)
        input_name = f"input_0.{IMAGE_TYPES[img.mime]}"
        (job_dir(job_id) / input_name).write_bytes(img.data)
        uses_depth = kind != "lithophane"
        params = {
            **opts, "kind": kind, "inputs": [input_name], "has_depth": uses_depth,
            "depth_source": ("ia" if "depth" in state.provider.capabilities else "heurística") if uses_depth else None,
        }
        job = state.store.create(
            id=job_id, mode=kind, prompt=None, provider=state.provider.name, input_image=input_name,
            params=params, name=(name or "").strip() or None,
        )
        state.runner.submit(job_id, pipelines.scene(state.provider, img, kind, **opts))
        return to_out(job)

    @app.post("/api/jobs/text3d", response_model=JobOut, status_code=201)
    async def create_text3d_job(request: Request, body: Text3DIn):
        if body.text.count("\n") > 3:
            raise HTTPException(422, "Máximo 4 líneas de texto")
        state = request.app.state
        params = body.model_dump()
        work = pipelines.compute(
            "Construyendo letras", creations.text3d, text=body.text, font=body.font, bold=body.bold,
            style=body.style, depth=body.depth, color=hex_rgb(body.color), plate_color=hex_rgb(body.plate_color),
        )
        job = state.store.create(
            id=uuid.uuid4().hex, mode="text3d", prompt=body.text.strip(), provider="local", params=params,
        )
        state.runner.submit(job["id"], work)
        return to_out(job)

    # ------------------------------------------------------------ jobs CRUD

    @app.get("/api/jobs", response_model=list[JobOut])
    async def list_jobs(request: Request, limit: int = 200):
        return [to_out(j) for j in request.app.state.store.list(min(max(limit, 1), 500))]

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    async def get_job(request: Request, job_id: str):
        return to_out(get_job_or_404(request, job_id))

    @app.patch("/api/jobs/{job_id}", response_model=JobOut)
    async def patch_job(request: Request, job_id: str, body: JobPatch):
        get_job_or_404(request, job_id)
        fields = {}
        if body.name is not None:
            fields["name"] = body.name.strip() or None
        if body.favorite is not None:
            fields["favorite"] = int(body.favorite)
        if fields:
            request.app.state.store.update(job_id, **fields)
        return to_out(get_job_or_404(request, job_id))

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        get_job_or_404(request, job_id)
        request.app.state.store.delete(job_id)
        shutil.rmtree(job_dir(job_id), ignore_errors=True)

    # ------------------------------------------------------------ files

    @app.get("/api/jobs/{job_id}/model.glb")
    async def download_model(request: Request, job_id: str):
        job, path = ready_model(request, job_id)
        return FileResponse(path, media_type="model/gltf-binary", filename=f"{slugify(job_title(job))}.glb")

    @app.get("/api/jobs/{job_id}/image")
    async def download_image(request: Request, job_id: str):
        job = get_job_or_404(request, job_id)
        if not job["input_image"]:
            raise HTTPException(404, "Este trabajo no tiene imagen")
        return FileResponse(job_dir(job_id) / job["input_image"])

    @app.get("/api/jobs/{job_id}/depth")
    async def get_depth(request: Request, job_id: str):
        get_job_or_404(request, job_id)
        path = job_dir(job_id) / "depth.png"
        if not path.is_file():
            raise HTTPException(404, "Este trabajo no tiene mapa de profundidad")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/jobs/{job_id}/thumbnail")
    async def get_thumbnail(request: Request, job_id: str):
        job = get_job_or_404(request, job_id)
        if not job.get("thumbnail"):
            raise HTTPException(404, "Sin miniatura")
        return FileResponse(job_dir(job_id) / job["thumbnail"])

    @app.put("/api/jobs/{job_id}/thumbnail", status_code=204)
    async def put_thumbnail(request: Request, job_id: str):
        get_job_or_404(request, job_id)
        ext = IMAGE_TYPES.get(request.headers.get("content-type", ""))
        if not ext:
            raise HTTPException(415, "La miniatura debe ser PNG, JPG o WEBP")
        data = await request.body()
        if not data or len(data) > MAX_THUMBNAIL_BYTES:
            raise HTTPException(413, "Miniatura vacía o demasiado grande")
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        (job_dir(job_id) / f"thumb.{ext}").write_bytes(data)
        request.app.state.store.update(job_id, thumbnail=f"thumb.{ext}")

    @app.get("/api/jobs/{job_id}/analysis")
    async def analysis(request: Request, job_id: str):
        _, path = ready_model(request, job_id)
        cache = path.with_name("analysis.json")
        if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
            return json.loads(cache.read_text())
        try:
            result = (await asyncio.to_thread(geometry.analyze, path.read_bytes())).as_dict()
        except geometry.GeometryError as exc:
            raise HTTPException(422, str(exc)) from exc
        cache.write_text(json.dumps(result))
        return result

    @app.get("/api/jobs/{job_id}/export")
    async def export(
        request: Request,
        job_id: str,
        format: ExportFormat = "glb",
        size_mm: float | None = Query(None, gt=0, le=10_000),
        fit: Literal["height", "width", "depth", "max"] = "height",
        colors: int = Query(4, ge=1, le=16, description="filamentos para 3mf_color / stl_colors"),
        height_mm: float | None = Query(None, gt=0, le=10_000, description="alias de size_mm con fit=height"),
    ):
        job, path = ready_model(request, job_id)
        if size_mm is None and height_mm is not None:
            size_mm, fit = height_mm, "height"
        filename = slugify(job_title(job))
        quads = job_dir(job_id) / "quads.obj"  # true quad topology from a quad remesh
        if format == "obj" and quads.is_file() and size_mm is None:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("model.obj", quads.read_bytes())
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{filename}.zip"'})
        if format in blender.FORMATS:
            if not request.app.state.blender:
                raise HTTPException(400, "Para FBX/.blend instala Blender (o indica BLENDER_PATH en .env) y reinicia la app")
            source, ext = (quads.read_bytes(), "obj") if quads.is_file() else (path.read_bytes(), "glb")
            try:
                data = await asyncio.to_thread(blender.convert, source, format, request.app.state.blender, 300, ext)
            except blender.BlenderError as exc:
                raise HTTPException(422, str(exc)) from exc
            ext, media_type = blender.FORMATS[format]
            return Response(data, media_type=media_type,
                            headers={"Content-Disposition": f'attachment; filename="{filename}.{ext}"'})
        if format == "glb" and job["mode"] == "rig":  # keep the skeleton and animations untouched
            return Response(path.read_bytes(), media_type="model/gltf-binary",
                            headers={"Content-Disposition": f'attachment; filename="{filename}.glb"'})
        try:
            data, ext, media_type = await asyncio.to_thread(
                geometry.export, path.read_bytes(), format, size_mm, fit, colors
            )
        except geometry.GeometryError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(data, media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="{filename}.{ext}"'})

    # ------------------------------------------------------------ derived jobs (tools)

    @app.post("/api/jobs/{job_id}/remesh", response_model=JobOut, status_code=201)
    async def remesh(request: Request, job_id: str, body: RemeshIn):
        parent, path = ready_model(request, job_id)
        if body.mode == "quads":
            async def work(on_progress):
                await on_progress(15, "Remallado en quads (Instant Meshes)")
                glb, obj = await asyncio.to_thread(geometry.quad_remesh, path.read_bytes(), body.target_faces)
                return GenerationResult(model_glb=glb, extra_files={"quads.obj": obj})
        else:
            work = pipelines.local(path, "Remallando", geometry.remesh, mode=body.mode,
                                   target_faces=body.target_faces, resolution=body.resolution)
        name = f"{job_title(parent)} · {REMESH_LABELS[body.mode]}"
        return create_derived_job(request, parent, "remesh", work, name, body.model_dump())

    @app.post("/api/jobs/{job_id}/enhance", response_model=JobOut, status_code=201)
    async def enhance(request: Request, job_id: str, body: EnhanceIn):
        """HD textures: AI upscaling (Real-ESRGAN) up to 2K/4K/8K plus PBR maps from the base colour."""
        parent, path = ready_model(request, job_id)
        if parent["mode"] == "rig":
            raise HTTPException(400, "Mejora las texturas del modelo original y después crea el rig")
        provider = request.app.state.provider
        upscaler = provider.upscale_image if "upscale" in provider.capabilities else None
        label = {"local": "IA local (Real-ESRGAN)", "fal": "IA en la nube (Real-ESRGAN)"}.get(provider.name, "IA")
        job_ref: dict = {}

        async def work(on_progress):
            await on_progress(10, "Ampliando texturas" + (" con IA" if upscaler else ""))
            glb, report = await textures.enhance(path.read_bytes(), body.target, 4, body.pbr, upscaler, label)
            request.app.state.store.update_params(job_ref["id"], {
                "method": report.method, "sizes": [list(s) for s in report.after], "textures": report.textures})
            return GenerationResult(model_glb=glb)

        out = create_derived_job(request, parent, "enhance", work, f"{job_title(parent)} · {body.target.upper()}",
                                 body.model_dump())
        job_ref["id"] = out.id
        return out

    @app.post("/api/jobs/{job_id}/repair", response_model=JobOut, status_code=201)
    async def repair(request: Request, job_id: str):
        parent, path = ready_model(request, job_id)
        work = pipelines.local(path, "Reparando malla", geometry.repair)
        return create_derived_job(request, parent, "repair", work, f"{job_title(parent)} · reparado", {})

    @app.post("/api/jobs/{job_id}/split", response_model=JobOut, status_code=201)
    async def split(request: Request, job_id: str, body: SplitIn):
        parent, path = ready_model(request, job_id)
        work = pipelines.local(path, "Dividiendo en piezas", geometry.split_parts, mode=body.mode, parts=body.parts)
        return create_derived_job(request, parent, "split", work, f"{job_title(parent)} · piezas", body.model_dump())

    @app.post("/api/jobs/{job_id}/retexture", response_model=JobOut, status_code=201)
    async def retexture(request: Request, job_id: str, body: RetextureIn):
        require(request, "retexture", "El retexturizado")
        parent, path = ready_model(request, job_id)
        seed = body.seed if body.seed is not None else new_seed()
        quality = {"2k": "draft", "4k": "standard", "8k": "high"}[body.resolution]
        req = GenerationRequest(mode="text", prompt=body.prompt.strip(), quality=quality, style=body.style, seed=seed)
        work = pipelines.retexture(request.app.state.provider, path, req)
        params = {"style": body.style, "seed": seed, "resolution": body.resolution}
        return create_derived_job(request, parent, "retexture", work, f"{job_title(parent)} · {req.prompt}",
                                  params, prompt=req.prompt)

    # ------------------------------------------------------------ rigging & animation

    @app.get("/api/rig/presets")
    async def rig_presets():
        def listing(presets):
            return [{"id": k, "label": p.label, "duration": p.duration, "loop": p.loop, "tags": p.tags}
                    for k, p in presets.items()]

        return {
            "presets": listing(rigging.PRESETS),
            "default": rigging.DEFAULT_PRESETS,
            "quadruped": {"presets": listing(rigging.quadruped.QUAD_PRESETS),
                          "default": rigging.quadruped.DEFAULT_QUAD_PRESETS,
                          "markers": [{"id": m, "label": rigging.quadruped.MARKER_LABELS[m]}
                                      for m in rigging.quadruped.MARKERS]},
            "library": [{"id": c["id"], "label": c["label"], "category": c["category"], "seconds": c["seconds"]}
                        for c in rigging.mocap.catalogue().values()],
            "markers": [{"id": m, "label": rigging.MARKER_LABELS[m]} for m in rigging.MARKERS],
        }

    def rig_source(request: Request, job: dict) -> tuple[dict, Path]:
        """Rigging always starts from the static model, also when re-rigging a rigged one."""
        if job["mode"] == "rig":
            source = request.app.state.store.get(job["params"].get("source_id", ""))
            if not source or source["status"] != "succeeded":
                raise HTTPException(409, "El modelo original de este personaje ya no existe")
            return source, job_dir(source["id"]) / source["model_file"]
        return ready_model(request, job["id"])

    @app.post("/api/jobs/{job_id}/rig/markers")
    async def rig_markers(request: Request, job_id: str, body: MarkersIn | None = None):
        """Markers to edit. `view_yaw` is how much the model shown in the viewer must turn to match them
        (a rigged model is already stored turned and centred)."""
        body = body or MarkersIn()
        job, _ = ready_model(request, job_id)
        p = job["params"]
        shown_yaw = p.get("yaw", 0) if job["mode"] == "rig" else 0
        same_template = body.template in ("auto", p.get("template", "humanoid"))
        if job["mode"] == "rig" and not body.auto and same_template and body.yaw in (None, shown_yaw):
            tpl = rigging.TEMPLATES[p.get("template", "humanoid")]
            return {"yaw": p["yaw"], "markers": p["markers"], "source_id": p["source_id"], "view_yaw": 0,
                    "template": tpl.name, "view": tpl.view, "labels": tpl.marker_labels}
        source, path = rig_source(request, job)
        try:
            result = await asyncio.to_thread(rigging.auto_markers, path.read_bytes(), body.yaw, body.template)
        except rigging.RigError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**result, "source_id": source["id"], "view_yaw": (result["yaw"] - shown_yaw) % 360}

    def submit_rig(request: Request, parent: dict, body: RigIn, bvh: list[tuple[str, str, bytes]],
                   keep_bvh: list[str], after: str | None = None) -> JobOut:
        """`after`: a job (HD textures) whose model, if it succeeds, is rigged instead of the parent's."""
        known = set(rigging.PRESETS) | set(rigging.quadruped.QUAD_PRESETS) | set(rigging.mocap.catalogue())
        unknown = [a for a in body.animations if a not in known]
        if unknown:
            raise HTTPException(422, f"Animaciones desconocidas: {', '.join(unknown)}")
        source, path = rig_source(request, parent)
        markers = body.markers
        if markers is not None:
            template = body.template if body.template != "auto" else (
                "quadruped" if "withers" in markers else "humanoid")
            body.template = template
            names = rigging.TEMPLATES[template].markers
            missing = [m for m in names if m not in markers or len(markers[m]) < 2]
            if missing:
                raise HTTPException(422, f"Faltan marcadores: {', '.join(missing)}")
            extra = {k: v for k, v in markers.items() if k.startswith("_")}
            markers = {m: [float(markers[m][0]), float(markers[m][1])] for m in names} | extra
            if body.symmetric and template == "humanoid":
                markers = rigging.mirror_markers(markers, "l")
        job_id = uuid.uuid4().hex
        job_dir(job_id).mkdir(parents=True)
        bvh_files = []
        for name, filename, data in bvh:  # (clip name, file name, bytes)
            (job_dir(job_id) / filename).write_bytes(data)
            bvh_files.append({"name": name, "file": filename})
        for item in keep_bvh_items(parent, keep_bvh):
            (job_dir(job_id) / item["file"]).write_bytes((job_dir(parent["id"]) / item["file"]).read_bytes())
            bvh_files.append(item)
        clips = [(item["name"], (job_dir(job_id) / item["file"]).read_text(encoding="utf-8", errors="replace"))
                 for item in bvh_files]

        async def work(on_progress):
            nonlocal source, path
            if after:
                dep = request.app.state.store.get(after)
                if dep and dep["status"] == "succeeded":
                    source, path = dep, job_dir(after) / dep["model_file"]
                    request.app.state.store.update_params(job_id, {"source_id": after})
            await on_progress(10, "Colocando esqueleto y calculando pesos")
            try:
                result = await asyncio.to_thread(rigging.rig, path.read_bytes(), markers, body.yaw, body.animations,
                                                 clips, body.in_place, body.template)
            except rigging.RigError as exc:
                raise geometry.GeometryError(str(exc)) from exc
            request.app.state.store.update_params(job_id, {"markers": result.markers, "yaw": result.yaw,
                                                           "clips": result.clips, "skeleton": result.skeleton,
                                                           "template": result.template})
            return GenerationResult(model_glb=result.glb)

        params = {"source_id": source["id"], "markers": markers, "yaw": body.yaw, "animations": body.animations,
                  "template": body.template,
                  "bvh": bvh_files, "in_place": body.in_place, "symmetric": body.symmetric}
        state = request.app.state
        title = job_title(source)
        job = state.store.create(id=job_id, mode="rig", prompt=source.get("prompt"), quality=source["quality"],
                                 provider="local", params=params, parent_id=after or parent["id"],
                                 name=f"{title} · rig"[:120])
        state.runner.submit(job_id, work, after=after)
        return to_out(job)

    @app.post("/api/jobs/{job_id}/character", response_model=list[JobOut], status_code=201)
    async def full_character(request: Request, job_id: str, body: CharacterIn):
        """One click, like a studio pipeline: HD/PBR textures (when the model has textures), then the
        skeleton, skin weights and animation clips, ready for Blender, Unity, Unreal or Godot."""
        parent, path = ready_model(request, job_id)
        if parent["mode"] == "rig":
            raise HTTPException(400, "Este personaje ya tiene rig: usa «Actualizar animaciones»")
        out, after = [], None
        if body.target != "none":
            has_textures = bool(await asyncio.to_thread(textures.textured_images, path.read_bytes()))
            if has_textures:
                hd = await enhance(request, job_id, EnhanceIn(target=body.target, pbr=body.pbr))
                out.append(hd)
                after = hd.id
        rig_body = RigIn(template=body.template, animations=body.animations, in_place=body.in_place)
        out.append(submit_rig(request, parent, rig_body, [], [], after))
        return out

    def keep_bvh_items(parent: dict, names: list[str]) -> list[dict]:
        if parent["mode"] != "rig":
            return []
        return [item for item in parent["params"].get("bvh", []) if item["name"] in names]

    @app.post("/api/jobs/{job_id}/rig", response_model=JobOut, status_code=201)
    async def rig_job(request: Request, job_id: str, body: RigIn):
        parent, _ = ready_model(request, job_id)
        keep = [b["name"] for b in parent["params"].get("bvh", [])] if parent["mode"] == "rig" else []
        return submit_rig(request, parent, body, [], keep)

    @app.post("/api/jobs/{job_id}/animate", response_model=JobOut, status_code=201)
    async def animate_job(
        request: Request,
        job_id: str,
        animations: str = Form(""),
        keep_bvh: str = Form(""),
        in_place: bool = Form(True),
        bvh: list[UploadFile] | None = File(None),
    ):
        """Re-bake a rigged character with another set of clips and/or new BVH motion capture."""
        parent, _ = ready_model(request, job_id)
        if parent["mode"] != "rig":
            raise HTTPException(400, "Primero crea el rig del personaje")
        uploads = []
        for i, upload in enumerate(bvh or []):
            data = await upload.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(413, "El BVH supera 20 MB")
            stem = re.sub(r"[^\w\-]+", "_", (upload.filename or f"mocap{i}").rsplit(".", 1)[0])[:40] or f"mocap{i}"
            uploads.append((stem, f"mocap_{i}_{stem}.bvh", data))
        p = parent["params"]
        body = RigIn(markers=p.get("markers"), yaw=p.get("yaw"), in_place=in_place,
                     template=p.get("template", "humanoid"), animations=[a for a in animations.split(",") if a])
        return submit_rig(request, parent, body, uploads, [n for n in keep_bvh.split(",") if n])

    def generative_parent(request: Request, job_id: str) -> dict:
        parent = get_job_or_404(request, job_id)
        if parent["mode"] not in GENERATIVE_MODES:
            raise HTTPException(400, "Sólo se pueden refinar o variar modelos generados por IA")
        return parent

    @app.post("/api/jobs/{job_id}/refine", response_model=list[JobOut], status_code=201)
    async def refine(request: Request, job_id: str):
        """Same input and seed at high quality. Text jobs reuse their concept image when they have one."""
        parent = generative_parent(request, job_id)
        p = parent["params"]
        opts = GenerationOptions(quality="high", style=p.get("style", "auto"), symmetry=p.get("symmetry", False),
                                 pose=p.get("pose", "none"), seed=p.get("seed"), engine=p.get("engine"))
        images = load_inputs(parent)
        mode = parent["mode"]
        if mode == "text" and parent["input_image"]:  # reference image from the text -> image step
            ref = job_dir(job_id) / parent["input_image"]
            images, mode = [InputImage(ref.read_bytes(), MIME_BY_EXT.get(ref.suffix[1:], "image/png"))], "image"
        return create_generation_jobs(request, mode, parent["prompt"], opts, images, {"refined": True}, parent["id"])

    @app.post("/api/jobs/{job_id}/variations", response_model=list[JobOut], status_code=201)
    async def variations(request: Request, job_id: str, body: VariationsIn):
        parent = generative_parent(request, job_id)
        p = parent["params"]
        opts = GenerationOptions(quality=parent["quality"], style=p.get("style", "auto"),
                                 symmetry=p.get("symmetry", False), pose=p.get("pose", "none"), variants=body.count,
                                 engine=p.get("engine"))
        return create_generation_jobs(request, parent["mode"], parent["prompt"], opts, load_inputs(parent),
                                      {"variation_of": parent["id"]}, parent["id"])

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
