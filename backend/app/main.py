"""HTTP API + static frontend for the AI 3D generator."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import json
from dataclasses import replace
import logging
import random
import re
import shutil
import subprocess
import unicodedata
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import agent as ai_agent
from ._version import VERSION
from . import blender, creations, geometry, pipelines, printing, publishing, quads, rigging, sheet, textures, updates
from .config import Settings, get_settings, save_user_settings, user_settings_path, with_user_settings
from .db import JobStore
from .jobs import JobRunner
from . import local_ai as local_ai_module
from .local_ai import LocalAI, LocalAIError
from .providers import GenerationError, GenerationRequest, GenerationResult, InputImage, Provider, build_provider
from .providers.fal import ENGINES as FAL_ENGINES
from .styles import POSES, STYLES

# repository root, or the unpacked bundle when running as the desktop app (PyInstaller)
RESOURCES = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
FRONTEND_DIR = RESOURCES / "frontend"
BLENDER_ADDON = RESOURCES / "integrations" / "blender" / "forja3d_blender.py"
MCP_SERVER = RESOURCES / "integrations" / "mcp" / "forja3d_mcp.py"
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
MIME_BY_EXT = {ext: mime for mime, ext in IMAGE_TYPES.items()}
GENERATIVE_MODES = ("text", "image", "multi_image")
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
HEX_ID = re.compile(r"^[0-9a-f]{32}$")
VIEW_ROLES = ("front", "left", "back", "right")

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
    mode: Literal["height", "components", "bed"] = "height"
    parts: int = Field(2, ge=2, le=8)
    # mode "bed": cut so every piece fits this printer at the final size
    bed: str = Field("bambu_a1", max_length=40)
    size_mm: float = Field(200, gt=0, le=10_000)
    fit: Literal["height", "width", "depth", "max"] = "height"


class SlicerIn(BaseModel):
    slicer: str | None = Field(None, max_length=40)  # None: the program associated with .3mf
    format: Literal["3mf", "3mf_color"] = "3mf"
    size_mm: float | None = Field(None, gt=0, le=10_000)
    fit: Literal["height", "width", "depth", "max"] = "height"
    colors: int = Field(4, ge=1, le=16)
    bed: str = Field("bambu_a1", max_length=40)


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
    anthropic_key: str | None = Field(None, max_length=300)


class AgentIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(None, max_length=64)
    lang: Literal["es", "en"] = "es"


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
        # auto=1: a picture rendered by the server, replaced by the viewer's own once the model is opened
        thumbnail_url=(f"{base}/thumbnail?v={version}" + ("&auto=1" if job["thumbnail"] == "preview.png" else ""))
        if job.get("thumbnail") else None,
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
        app.state.local_ai = local_ai = LocalAI(settings.data_dir, RESOURCES)
        app.state.updater = updates.Updater(VERSION, settings.data_dir / "actualizaciones")
        try:
            app.state.provider = provider or make_provider(settings)
        except (GenerationError, ValueError) as exc:  # e.g. cloud mode without a key: start in demo mode
            app.state.provider_error = str(exc)
            app.state.provider = build_provider(replace(settings, provider="mock"))
        if getattr(app.state.provider, "launcher", None) is not None:
            warm_up(app.state.provider)
        elif hasattr(app.state.provider, "start"):
            await app.state.provider.start()
        app.state.runner = JobRunner(store, settings.files_dir, settings.max_concurrent_jobs)
        app.state.blender = blender.find_blender(settings.blender_path)
        yield
        await app.state.runner.shutdown()
        local_ai.stop()
        store.close()

    app = FastAPI(title="Forja3D", version="0.2.0", lifespan=lifespan)

    def make_provider(s: Settings) -> Provider:
        """The built-in local AI launcher is used for a worker on this machine, not for a remote one."""
        on_this_pc = s.local_worker_url.rstrip("/") in ("http://127.0.0.1:8001", "http://localhost:8001")
        return build_provider(s, app.state.local_ai if on_this_pc else None)

    def warm_up(local_provider) -> None:
        """Start the local AI in the background (or install it, on the desktop app's first run with an NVIDIA GPU)."""
        local_ai = app.state.local_ai

        async def run():
            try:
                if local_ai.installed:
                    await local_provider._ready()
                elif os.getenv("FORJA_AUTO_INSTALL") and local_ai.status()["gpu"]:
                    local_ai.start_install()
            except (GenerationError, LocalAIError) as exc:
                logging.warning("La IA local no arrancó: %s", exc)

        app.state.warm_up = asyncio.create_task(run())

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
        variants: list[int] | None = None,
        group: str | None = None,
    ) -> list[JobOut]:
        """One job per shape variant. `variants` says how each one is built (see the provider's
        variant_label; default 0, 1, 2…); jobs of one `group` are alternatives to choose from."""
        state = request.app.state
        engines = [e["id"] for e in getattr(state.provider, "engines", [])]
        if opts.engine and opts.engine not in engines:
            raise HTTPException(422, f"Motor desconocido: {opts.engine}")
        base_seed = opts.seed if opts.seed is not None else new_seed()
        variants = variants or list(range(opts.variants))
        ids = [uuid.uuid4().hex for _ in variants]
        if len(ids) > 1 and group is None:
            group = ids[0]
        label_of = getattr(state.provider, "variant_label", None)
        jobs = []
        for i, (job_id, variant) in enumerate(zip(ids, variants)):
            seed = (base_seed + i) % 2**31
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
            if variant:
                params["variant"] = variant
            if group:
                label = label_of(variant, len(images)) if label_of else None
                params.update(group=group, variant_label=label or f"Variante {i + 1}")
            job = state.store.create(
                id=job_id, mode=mode, prompt=prompt, quality=opts.quality, provider=state.provider.name,
                input_image=inputs[0] if inputs else None, params=params, parent_id=parent_id,
            )
            req = GenerationRequest(
                mode=mode, prompt=prompt, quality=opts.quality, images=images,
                style=opts.style, symmetry=opts.symmetry, pose=opts.pose, seed=seed, engine=opts.engine,
                roles=(extra or {}).get("roles"), variant=variant,
            )
            state.runner.submit(job_id, pipelines.generation(state.provider, req, preview=bool(group)))
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
            with Image.open(io.BytesIO(data)) as img:
                if img.getexif().get(0x0112, 1) != 1:  # phone photo stored sideways: apply its rotation
                    buf = io.BytesIO()
                    ImageOps.exif_transpose(img).save(buf, format="PNG")
                    data, mime = buf.getvalue(), "image/png"
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
            "version": VERSION,
            "agent": bool(request.app.state.settings.anthropic_key),
            "local_ai": {k: v for k, v in request.app.state.local_ai.status().items() if k != "log"},
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

    def mcp_script(request: Request) -> str:
        if not MCP_SERVER.is_file():
            raise HTTPException(404, "Servidor MCP no disponible")
        source = MCP_SERVER.read_text(encoding="utf-8")
        base = str(request.base_url).rstrip("/").replace('"', "")
        return source.replace('DEFAULT_URL = "http://127.0.0.1:8000"', f'DEFAULT_URL = "{base}"', 1)

    @app.get("/api/integrations/mcp")
    async def mcp_server(request: Request):
        """MCP server (stdio) for Claude Code, Cursor or Claude Desktop, preconfigured with this app's address."""
        return Response(mcp_script(request), media_type="text/x-python",
                        headers={"Content-Disposition": 'attachment; filename="forja3d_mcp.py"'})

    @app.get("/api/integrations/mcp/setup")
    async def mcp_setup(request: Request):
        """Commands to register the MCP server. On this computer the script is also saved to the data folder,
        so the command works as is; from another machine, download it and use its path."""
        source = mcp_script(request)
        host = request.client.host if request.client else ""
        installed = host in ("127.0.0.1", "::1", "localhost", "testclient")
        if installed:
            target = request.app.state.settings.data_dir / "integrations" / "forja3d_mcp.py"
            if not target.is_file() or target.read_text(encoding="utf-8") != source:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source, encoding="utf-8")
            path = str(target.resolve())
        else:
            path = "/ruta/a/forja3d_mcp.py"
        arg = f'"{path}"' if " " in path else path
        return {
            "path": path,
            "installed": installed,
            "script_url": str(request.base_url).rstrip("/") + "/api/integrations/mcp",
            "requires": "uv: https://docs.astral.sh/uv/",
            "claude_code": f"claude mcp add forja3d -- uv run {arg}",
            "json_config": {"mcpServers": {"forja3d": {"command": "uv", "args": ["run", path]}}},
            "config_files": {"cursor": "~/.cursor/mcp.json", "claude_desktop": "claude_desktop_config.json"},
        }

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
            "anthropic_key_set": bool(s.anthropic_key),
            "anthropic_key_hint": f"…{s.anthropic_key[-4:]}" if len(s.anthropic_key) >= 8 else "",
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
        for key in ("fal_key", "anthropic_key"):
            if values.get(key) == "":
                values.pop(key)  # empty field = keep the saved key
        candidate = replace(state.settings, **values)
        try:
            new_provider = make_provider(candidate)
        except (GenerationError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if getattr(new_provider, "launcher", None) is not None:
            warm_up(new_provider)
        elif hasattr(new_provider, "start"):
            await new_provider.start()
        state.settings = save_user_settings(state.settings, values)
        state.provider, state.provider_error = new_provider, None  # running jobs keep their own provider
        state.blender = blender.find_blender(state.settings.blender_path)
        return await get_app_settings(request)

    # ------------------------------------------------------------ free local AI (installed by the app)

    @app.get("/api/local-ai")
    async def local_ai_status(request: Request):
        status = request.app.state.local_ai.status()
        return {**status, "worker_up": bool(await request.app.state.local_ai.healthy(1))}

    # ------------------------------------------------------------ updates

    @app.get("/api/update")
    async def update_status(request: Request, force: bool = False):
        return await request.app.state.updater.check(force)

    @app.post("/api/update/install")
    async def update_install(request: Request):
        """Windows desktop app: download the new installer, run it silently and reopen the app."""
        updater = request.app.state.updater
        if request.app.state.local_ai.installing:  # quitting would throw away a half-done download
            raise HTTPException(409, "Espera a que termine la instalación de la IA y actualiza después")
        if request.app.state.runner.busy:  # nor a model being generated
            raise HTTPException(409, "Espera a que terminen los trabajos en curso y actualiza después")
        status = await updater.check()
        if not status.get("can_install") or not getattr(sys, "frozen", False):
            raise HTTPException(400, "Esta versión no se puede actualizar sola: descárgala desde la página de versiones")
        loop = asyncio.get_running_loop()

        def ready(installer: Path) -> None:
            request.app.state.local_ai.stop()  # os._exit skips shutdown: don't leave the worker running
            updates.run_windows_installer(installer)
            loop.call_later(1.0, os._exit, 0)  # quit so the installer can replace the app's files

        try:
            updater.start(ready)
        except updates.UpdateError as exc:
            raise HTTPException(400, str(exc)) from exc
        return await updater.check()

    @app.get("/api/diagnostics")
    async def diagnostics(request: Request):
        """Everything needed to understand a problem, as text the user can paste in a message."""
        state = request.app.state
        ai = state.local_ai
        status = ai.status()
        lines = [f"Forja3D {VERSION} · {sys.platform} · Python {sys.version.split()[0]}",
                 f"Motor: {state.settings.provider} (activo: {state.provider.name})"
                 + (f" · error: {state.provider_error}" if state.provider_error else "")]
        smi = local_ai_module.find_nvidia_smi()
        if smi:
            try:
                out = await asyncio.to_thread(
                    subprocess.run, [smi, "--query-gpu=name,driver_version,memory.used,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                lines.append(f"GPU: {out.stdout.strip() or out.stderr.strip()}")
            except (OSError, subprocess.SubprocessError) as exc:
                lines.append(f"GPU: nvidia-smi falló ({exc})")
        else:
            lines.append("GPU: no se encontró nvidia-smi (sin NVIDIA o sin driver)")
        try:
            lines.append(f"Disco libre: {shutil.disk_usage(state.settings.data_dir).free / 1024**3:.0f} GB")
        except OSError:
            pass
        total, free = local_ai_module.ram_gb()
        if total:
            lines.append(f"RAM: {total:.0f} GB ({free:.1f} GB libres)")
        health = await ai.healthy(2)
        lines.append(f"IA local: instalada={status['installed']} instalando={status['installing']} "
                     f"paso={status['step'] or '-'} {status['progress']}% en marcha={bool(health)} "
                     f"modo_ligero={ai.light_mode} url={status['url']}")
        if status["error"]:
            lines.append(f"Error de instalación: {status['error']}")
        if health:
            lines.append(f"Worker: {health.get('gpu')} · {', '.join(health.get('capabilities', []))} · "
                         f"modelos {', '.join(health.get('models', []))} · poca RAM={health.get('low_ram')}")
        failed = [j for j in state.store.list(30) if j["status"] == "failed"][:3]
        lines += [f"Trabajo fallido ({j['mode']}): {j.get('error')}" for j in failed]

        def read(path: Path) -> list[str]:
            try:  # only the end of big logs
                with open(path, "rb") as fh:
                    fh.seek(max(0, fh.seek(0, 2) - 400_000))
                    return fh.read().decode("utf-8", errors="replace").splitlines()
            except OSError:
                return []

        def quiet(text: list[str]) -> list[str]:  # the app polls all the time: drop request lines
            return [line for line in text if '"GET /' not in line and "uvicorn.access" not in line]

        def last_error(text: list[str]) -> list[str]:
            starts = [i for i, line in enumerate(text) if line.startswith("Traceback")]
            if not starts:
                return []
            block = text[max(0, starts[-1] - 2):starts[-1] + 40]
            end = next((i + 1 for i, line in enumerate(block) if i > 2 and line[:1].strip()
                        and not line.startswith(("Traceback", "  ", "During", "The above"))), len(block))
            return block[:end]

        lines += ["", "--- Instalación (últimas líneas) ---", *status["log"]]
        app_log = read(Path(state.settings.data_dir) / "forja3d.log")
        worker_log = read(ai.root / "worker.log")
        for title, text in (("forja3d.log", app_log), ("worker.log", worker_log)):
            error = last_error(text)
            if error:
                lines += ["", f"--- Último error ({title}) ---", *error]
        lines += ["", "--- worker.log ---", *quiet(worker_log)[-30:]]
        lines += ["", "--- forja3d.log ---", *quiet(app_log)[-30:]]
        return {"text": "\n".join(lines)}

    @app.put("/api/local-ai/textures")
    async def local_ai_textures(request: Request, enabled: bool = Query(...)):
        ai = request.app.state.local_ai
        if enabled and not ai.status()["ai_textures_available"]:
            raise HTTPException(400, "El pintor de texturas con IA no está instalado en tu equipo")
        health = await ai.healthy(2)
        if health and health.get("busy"):
            raise HTTPException(409, "Espera a que termine la generación en curso")
        ai.set_ai_textures(enabled)
        request.app.state.provider.info = {}  # capabilities are read again from the restarted worker
        return ai.status()

    @app.post("/api/local-ai/install")
    async def local_ai_install(request: Request):
        if settings_locked():
            raise HTTPException(403, "Los ajustes están bloqueados en este servidor")
        try:
            request.app.state.local_ai.start_install()
        except LocalAIError as exc:
            raise HTTPException(400, str(exc)) from exc
        return request.app.state.local_ai.status()

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

    @app.post("/api/sheet/split")
    async def split_character_sheet(image: UploadFile = File(...)):
        """Find the full-body views on a character sheet and cut them out (transparent background)."""
        upload = await read_image(image)
        try:
            views = await asyncio.to_thread(sheet.split_sheet, upload.data)
        except sheet.SheetError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"views": [{"role": v["role"], "box": v["box"],
                           "url": "data:image/png;base64," + base64.b64encode(v["image"]).decode()} for v in views]}

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
        roles: list[str] | None = Form(None),
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
        if roles and len(inputs) > 1:  # which slot each picture came from (front + back is not front + side)
            if len(roles) != len(inputs) or len(set(roles)) != len(roles) or not set(roles) <= set(VIEW_ROLES):
                raise HTTPException(422, "Cada imagen necesita una vista distinta: front, left, back o right")
            order = sorted(range(len(roles)), key=lambda i: VIEW_ROLES.index(roles[i]))  # front first
            inputs, extra["roles"] = [inputs[i] for i in order], [roles[i] for i in order]
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
    async def analysis(request: Request, job_id: str, size_mm: float | None = Query(None, gt=0, le=10_000),
                       fit: Literal["height", "width", "depth", "max"] = "height", bed: str = "bambu_a1"):
        _, path = ready_model(request, job_id)
        cache = path.with_name("analysis.json")
        if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
            result = json.loads(cache.read_text())
        else:
            try:
                result = (await asyncio.to_thread(geometry.analyze, path.read_bytes())).as_dict()
            except geometry.GeometryError as exc:
                raise HTTPException(422, str(exc)) from exc
            cache.write_text(json.dumps(result))
        # printability at the final size (cheap enough to compute every time)
        report = await asyncio.to_thread(printing.printability, path.read_bytes(), size_mm, fit)
        printer = printing.BEDS.get(bed, printing.BEDS["bambu_a1"])
        return {**result, "print": {**report, "bed": bed, "bed_label": printer[0], "bed_mm": printer[1],
                                    "fits_bed": printing.fits(report["size_mm"], printer[1])}}

    @app.get("/api/jobs/{job_id}/publish")
    async def publish_kit(request: Request, job_id: str, size_mm: float = Query(100, gt=0, le=10_000),
                          bed: str = "bambu_a1"):
        """ZIP for MakerWorld / Printables / Thingiverse: 3MF + STL at size, previews, description."""
        job, path = ready_model(request, job_id)
        title = job_title(job)
        try:
            data = await asyncio.to_thread(publishing.kit, path.read_bytes(), title, size_mm, bed)
        except geometry.GeometryError as exc:
            raise HTTPException(422, str(exc)) from exc
        name = f"forja3d-publicar-{slugify(title)}.zip"
        return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/printers")
    async def printers():
        return {"beds": [{"id": k, "label": v[0], "size_mm": v[1]} for k, v in printing.BEDS.items()],
                "slicers": [{"id": s["id"], "label": s["label"]} for s in printing.find_slicers()]}

    @app.post("/api/jobs/{job_id}/slicer")
    async def send_to_slicer(request: Request, job_id: str, body: SlicerIn):
        """Desktop only: export for printing and open it in an installed slicer."""
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(403, "Abrir el laminador solo funciona en la app de escritorio")
        job, path = ready_model(request, job_id)
        glb = path.read_bytes()
        bed = printing.BEDS.get(body.bed, printing.BEDS["bambu_a1"])[1]
        try:
            if body.format == "3mf_color":
                data, _, _ = await asyncio.to_thread(geometry.export, glb, "3mf_color", body.size_mm, body.fit, body.colors)
            else:
                data = await asyncio.to_thread(printing.plate_3mf, glb, body.size_mm, body.fit, bed)
        except geometry.GeometryError as exc:
            raise HTTPException(422, str(exc)) from exc
        out_dir = settings.data_dir / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{slugify(job_title(job))}.3mf"
        target.write_bytes(data)
        try:
            opened = await asyncio.to_thread(printing.open_in_slicer, target, body.slicer)
        except OSError as exc:
            raise HTTPException(422, f"No se pudo abrir el laminador: {exc}") from exc
        return {"file": str(target), "opened_with": opened}

    @app.get("/api/jobs/{job_id}/export")
    async def export(
        request: Request,
        job_id: str,
        format: ExportFormat = "glb",
        size_mm: float | None = Query(None, gt=0, le=10_000),
        fit: Literal["height", "width", "depth", "max"] = "height",
        colors: int = Query(4, ge=1, le=16, description="filamentos para 3mf_color / stl_colors"),
        height_mm: float | None = Query(None, gt=0, le=10_000, description="alias de size_mm con fit=height"),
        bed: str = Query("bambu_a1", max_length=40, description="impresora: reparte las piezas en su cama (3MF)"),
    ):
        job, path = ready_model(request, job_id)
        if size_mm is None and height_mm is not None:
            size_mm, fit = height_mm, "height"
        filename = slugify(job_title(job))
        quad_obj = job_dir(job_id) / "quads.obj"  # true quad topology from a quad remesh
        quad_files = {n: (job_dir(job_id) / n).read_bytes() for n in ("quads.mtl", "quads.png")
                      if (job_dir(job_id) / n).is_file()}  # its material and texture
        if format == "obj" and quad_obj.is_file() and size_mm is None:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("model.obj", quad_obj.read_bytes().replace(b"mtllib quads.mtl", b"mtllib model.mtl"))
                if "quads.mtl" in quad_files:
                    zf.writestr("model.mtl", quad_files["quads.mtl"].replace(b"quads.png", b"model.png"))
                if "quads.png" in quad_files:
                    zf.writestr("model.png", quad_files["quads.png"])
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{filename}.zip"'})
        if format in blender.FORMATS:
            if not request.app.state.blender:
                raise HTTPException(400, "Para FBX/.blend instala Blender (o indica BLENDER_PATH en .env) y reinicia la app")
            source, ext = (quad_obj.read_bytes(), "obj") if quad_obj.is_file() else (path.read_bytes(), "glb")
            try:
                data = await asyncio.to_thread(blender.convert, source, format, request.app.state.blender, 300, ext,
                                               quad_files if ext == "obj" else None)
            except blender.BlenderError as exc:
                raise HTTPException(422, str(exc)) from exc
            ext, media_type = blender.FORMATS[format]
            return Response(data, media_type=media_type,
                            headers={"Content-Disposition": f'attachment; filename="{filename}.{ext}"'})
        if format == "3mf" and job["mode"] == "split":  # each piece its own object, laid out on the bed
            plate = printing.BEDS.get(bed, printing.BEDS["bambu_a1"])[1]
            try:
                data = await asyncio.to_thread(printing.plate_3mf, path.read_bytes(), size_mm, fit, plate)
            except geometry.GeometryError as exc:
                raise HTTPException(422, str(exc)) from exc
            return Response(data, media_type="model/3mf",
                            headers={"Content-Disposition": f'attachment; filename="{filename}.3mf"'})
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
                await on_progress(15, "Remallado en quads y textura (Instant Meshes)")
                glb, files, _ = await asyncio.to_thread(quads.quad_remesh, path.read_bytes(), body.target_faces,
                                                        name="quads")
                return GenerationResult(model_glb=glb, extra_files=files)
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
        if body.mode == "bed":
            bed = printing.BEDS.get(body.bed)
            if not bed:
                raise HTTPException(422, f"Impresora desconocida: {body.bed}")

            async def work(on_progress):
                await on_progress(15, "Dividiendo para la cama de impresión")
                result = await asyncio.to_thread(printing.split_for_bed, path.read_bytes(), bed[1], body.size_mm, body.fit)
                return GenerationResult(model_glb=result.glb)
        else:
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
            "library": [{"id": c["id"], "label": c["label"], "label_en": c.get("label_en", c["label"]),
                         "category": c["category"], "seconds": c["seconds"]}
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
        kept = {"roles": p["roles"]} if p.get("roles") and mode == parent["mode"] else {}
        return create_generation_jobs(request, mode, parent["prompt"], opts, images, {"refined": True, **kept},
                                      parent["id"], variants=[p.get("variant", 0)])

    @app.post("/api/jobs/{job_id}/variations", response_model=list[JobOut], status_code=201)
    async def variations(request: Request, job_id: str, body: VariationsIn):
        parent = generative_parent(request, job_id)
        p = parent["params"]
        opts = GenerationOptions(quality=parent["quality"], style=p.get("style", "auto"),
                                 symmetry=p.get("symmetry", False), pose=p.get("pose", "none"), variants=body.count,
                                 engine=p.get("engine"))
        kept = {"roles": p["roles"]} if p.get("roles") else {}
        # built differently from the model they vary (the photo alone, or other imagined sides), and
        # grouped with it so you can keep the best one
        variants = [v for v in range(1, body.count + 2) if v != p.get("variant", 0)][:body.count]
        group = p.get("group") or parent["id"]
        if not p.get("group"):
            label = getattr(request.app.state.provider, "variant_label", lambda *_: None)(
                p.get("variant", 0), len(p.get("inputs", [])))
            request.app.state.store.update_params(parent["id"], {"group": group, "variant_label": label or "Original"})
        return create_generation_jobs(request, parent["mode"], parent["prompt"], opts, load_inputs(parent),
                                      {"variation_of": parent["id"], **kept}, parent["id"], variants, group)

    @app.post("/api/jobs/{job_id}/keep", response_model=JobOut)
    async def keep_variant(request: Request, job_id: str):
        """Keep this shape variant and delete the other ones of its group (unfinished ones are stopped)."""
        job = get_job_or_404(request, job_id)
        group = job["params"].get("group")
        if not group:
            raise HTTPException(400, "Este modelo no tiene variantes")
        state = request.app.state
        for other in state.store.list(limit=2000):
            if other["id"] != job_id and other["params"].get("group") == group:
                state.runner.cancel(other["id"])
                state.store.delete(other["id"])
                shutil.rmtree(job_dir(other["id"]), ignore_errors=True)
        state.store.update_params(job_id, {"group": None, "variant_label": None})
        return to_out(state.store.get(job_id))

    # ------------------------------------------------------------ AI agent (chat)

    def agent_tools(request: Request) -> dict[str, ai_agent.ToolFn]:
        """The agent's tools call the same code as the buttons in the app."""

        def job_card(out: JobOut) -> dict:
            return {"type": "job", "job": out.model_dump()}

        async def make_concepts(args):
            result = await create_concepts(request, ConceptsIn(prompt=args["prompt"], style=args["style"]))
            ids = [c["id"] for c in result["concepts"]]
            return {"concept_ids": ids}, [{"type": "concepts", "prompt": args["prompt"], "concepts": result["concepts"]}]

        async def generate_3d(args):
            if args.get("concept_id"):
                jobs = await create_image_job(request, images=None, concept_id=args["concept_id"], prompt=args["prompt"],
                                              quality=args["quality"], style=args["style"], seed=None, variants=1,
                                              engine=None)
            else:
                require(request, "text", "Texto a 3D")
                body = TextJobIn(prompt=args["prompt"], quality=args["quality"], style=args["style"],
                                 pose="a-pose" if args.get("a_pose") else "none")
                jobs = create_generation_jobs(request, "text", body.prompt.strip(), body, [])
            return [{"job_id": j.id, "status": j.status} for j in jobs], [job_card(j) for j in jobs]

        async def find_animations(args):
            words = [w for w in ai_agent.fold(args["query"]).split() if len(w) > 1]
            if args["body"] == "quadruped":
                pool = [(k, p.label, " ".join(p.tags), p.duration)
                        for k, p in rigging.quadruped.QUAD_PRESETS.items()]
            else:
                pool = [(k, p.label, " ".join(p.tags), p.duration) for k, p in rigging.PRESETS.items()]
                pool += [(c["id"], c["label"], f"{c.get('label_en', '')} {c['category']}", c["seconds"])
                         for c in rigging.mocap.catalogue().values()]
            hits = []
            for clip_id, label, extra, seconds in pool:
                text = ai_agent.fold(f"{clip_id} {label} {extra}")
                score = sum(w in text for w in words)
                if score or not words:
                    hits.append((-score, clip_id, label, seconds))
            hits.sort()
            return [{"id": h[1], "label": h[2], "seconds": h[3]} for h in hits[:25]], []

        async def rig_character(args):
            body = CharacterIn(template=args["body"], animations=args["animations"] or list(rigging.DEFAULT_PRESETS))
            jobs = await full_character(request, args["job_id"], body)
            return [{"job_id": j.id, "title": j.title, "status": j.status} for j in jobs], [job_card(j) for j in jobs]

        async def check_print(args):
            _, path = ready_model(request, args["job_id"])
            report = await asyncio.to_thread(printing.printability, path.read_bytes(), args["size_mm"], "height")
            label, bed = printing.BEDS.get(args["printer"], printing.BEDS["bambu_a1"])
            return {**report, "printer": label, "bed_mm": bed, "fits_bed": printing.fits(report["size_mm"], bed),
                    "printers": list(printing.BEDS)}, []

        async def list_models(args):
            jobs = request.app.state.store.list(min(max(int(args["limit"]), 1), 20))
            return [{"job_id": j["id"], "title": job_title(j), "type": j["mode"], "status": j["status"]}
                    for j in jobs], []

        async def job_status(args):
            out = to_out(get_job_or_404(request, args["job_id"]))
            return {"job_id": out.id, "title": out.title, "type": out.mode, "status": out.status,
                    "progress": out.progress, "stage": out.stage, "error": out.error}, [job_card(out)]

        return {"make_concepts": make_concepts, "generate_3d": generate_3d, "find_animations": find_animations,
                "rig_character": rig_character, "check_print": check_print, "list_models": list_models,
                "job_status": job_status}

    @app.post("/api/agent/chat")
    async def agent_chat(request: Request, body: AgentIn):
        state = request.app.state
        key = state.settings.anthropic_key
        agent = getattr(state, "agent", None)
        if not key and agent is None:  # free built-in assistant: no key needed
            if "chat" in getattr(state.provider, "capabilities", ()):  # the text model on the user's GPU
                if getattr(state, "local_model_agent", None) is None:
                    state.local_model_agent = ai_agent.LocalModelAgent(lambda msgs: request.app.state.provider.chat(msgs))
                agent = state.local_model_agent
            else:
                state.offline_agent = getattr(state, "offline_agent", None) or ai_agent.OfflineAgent()
                agent = state.offline_agent
        elif agent is None or getattr(state, "agent_key", None) != key:
            try:
                agent = ai_agent.Agent(key, state.settings.anthropic_model)
            except ai_agent.AgentError as exc:
                raise HTTPException(400, str(exc)) from exc
            state.agent, state.agent_key = agent, key
        try:
            turn = await agent.chat(body.message.strip(), body.conversation_id, agent_tools(request), body.lang)
        except ai_agent.AgentError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"reply": turn.reply, "conversation_id": turn.conversation_id, "artifacts": turn.artifacts}

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
