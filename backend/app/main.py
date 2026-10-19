"""HTTP API + static frontend for the AI 3D generator."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import shutil
import unicodedata
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import creations, geometry, pipelines
from .config import Settings, get_settings
from .db import JobStore
from .jobs import JobRunner
from .providers import GenerationError, GenerationRequest, InputImage, Provider, build_provider
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
ExportFormat = Literal[tuple(geometry.EXPORT_FORMATS)]  # type: ignore[valid-type]

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
}
SCENE_KINDS = ("scene", "relief", "lithophane")
HEX_COLOR = r"^#[0-9a-fA-F]{6}$"
REMESH_LABELS = {"decimate": "reducido", "lowpoly": "low-poly", "voxel": "vóxel", "smooth": "suavizado"}

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------- schemas


class GenerationOptions(BaseModel):
    quality: Quality = "standard"
    style: Style = "auto"
    symmetry: bool = False
    pose: Pose = "none"
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    variants: int = Field(1, ge=1, le=4)


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
    mode: Literal["decimate", "lowpoly", "voxel", "smooth"] = "decimate"
    target_faces: int = Field(5000, ge=100, le=500_000)
    resolution: int = Field(32, ge=8, le=96)


class SplitIn(BaseModel):
    mode: Literal["height", "components"] = "height"
    parts: int = Field(2, ge=2, le=8)


class RetextureIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    style: Style = "auto"
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


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
    settings = settings or get_settings()
    concepts_dir = settings.files_dir / "concepts"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        concepts_dir.mkdir(parents=True, exist_ok=True)
        store = JobStore(settings.db_path)
        store.fail_interrupted()
        app.state.store = store
        app.state.provider = provider or build_provider(settings)
        if hasattr(app.state.provider, "start"):
            await app.state.provider.start()
        app.state.runner = JobRunner(store, settings.files_dir, settings.max_concurrent_jobs)
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
                "style": opts.style, "symmetry": opts.symmetry, "pose": opts.pose, "seed": seed,
                "inputs": inputs, **(extra or {}),
            }
            job = state.store.create(
                id=job_id, mode=mode, prompt=prompt, quality=opts.quality, provider=state.provider.name,
                input_image=inputs[0] if inputs else None, params=params, parent_id=parent_id,
            )
            req = GenerationRequest(
                mode=mode, prompt=prompt, quality=opts.quality, images=images,
                style=opts.style, symmetry=opts.symmetry, pose=opts.pose, seed=seed,
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
            "capabilities": sorted(provider.capabilities),
            "styles": [{"id": k, "label": v[0]} for k, v in STYLES.items()],
            "poses": [{"id": k, "label": v[0]} for k, v in POSES.items()],
            "export_formats": list(geometry.EXPORT_FORMATS),
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
        opts = GenerationOptions(quality=quality, style=style, seed=seed, variants=variants)
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
        height_mm: float | None = Query(None, gt=0, le=10_000, description="alias de size_mm con fit=height"),
    ):
        job, path = ready_model(request, job_id)
        if size_mm is None and height_mm is not None:
            size_mm, fit = height_mm, "height"
        try:
            data, ext, media_type = await asyncio.to_thread(
                geometry.export, path.read_bytes(), format, size_mm, fit
            )
        except geometry.GeometryError as exc:
            raise HTTPException(422, str(exc)) from exc
        filename = f"{slugify(job_title(job))}.{ext}"
        return Response(data, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    # ------------------------------------------------------------ derived jobs (tools)

    @app.post("/api/jobs/{job_id}/remesh", response_model=JobOut, status_code=201)
    async def remesh(request: Request, job_id: str, body: RemeshIn):
        parent, path = ready_model(request, job_id)
        work = pipelines.local(path, "Remallando", geometry.remesh, mode=body.mode,
                               target_faces=body.target_faces, resolution=body.resolution)
        name = f"{job_title(parent)} · {REMESH_LABELS[body.mode]}"
        return create_derived_job(request, parent, "remesh", work, name, body.model_dump())

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
        req = GenerationRequest(mode="text", prompt=body.prompt.strip(), quality=parent["quality"],
                                style=body.style, seed=seed)
        work = pipelines.retexture(request.app.state.provider, path, req)
        params = {"style": body.style, "seed": seed}
        return create_derived_job(request, parent, "retexture", work, f"{job_title(parent)} · {req.prompt}",
                                  params, prompt=req.prompt)

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
                                 pose=p.get("pose", "none"), seed=p.get("seed"))
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
                                 symmetry=p.get("symmetry", False), pose=p.get("pose", "none"), variants=body.count)
        return create_generation_jobs(request, parent["mode"], parent["prompt"], opts, load_inputs(parent),
                                      {"variation_of": parent["id"]}, parent["id"])

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
