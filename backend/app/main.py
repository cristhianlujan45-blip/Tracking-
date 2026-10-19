"""HTTP API + static frontend for the AI 3D generator."""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .db import JobStore
from .jobs import JobRunner
from .providers import GenerationRequest, build_provider

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
Quality = Literal["draft", "standard", "high"]

logging.basicConfig(level=logging.INFO)


class TextJobIn(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    quality: Quality = "standard"


class JobOut(BaseModel):
    id: str
    mode: str
    prompt: str | None
    quality: str
    status: str
    progress: int
    stage: str | None
    error: str | None
    provider: str
    created_at: float
    updated_at: float
    model_url: str | None
    image_url: str | None


def to_out(job: dict) -> JobOut:
    base = f"/api/jobs/{job['id']}"
    return JobOut(
        **{k: job[k] for k in JobOut.model_fields if k in job},
        model_url=f"{base}/model.glb" if job["model_file"] else None,
        image_url=f"{base}/image" if job["input_image"] else None,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.files_dir.mkdir(parents=True, exist_ok=True)
        store = JobStore(settings.db_path)
        store.fail_interrupted()
        provider = build_provider(settings)
        app.state.store = store
        app.state.runner = JobRunner(store, provider, settings.files_dir, settings.max_concurrent_jobs)
        app.state.provider_name = provider.name
        yield
        await app.state.runner.shutdown()
        store.close()

    app = FastAPI(title="Forja3D", version="0.1.0", lifespan=lifespan)

    def get_job_or_404(request: Request, job_id: str) -> dict:
        job = request.app.state.store.get(job_id)
        if not job:
            raise HTTPException(404, "Trabajo no encontrado")
        return job

    @app.get("/api/health")
    async def health(request: Request):
        return {"status": "ok", "provider": request.app.state.provider_name}

    @app.post("/api/jobs/text", response_model=JobOut, status_code=201)
    async def create_text_job(request: Request, body: TextJobIn):
        state = request.app.state
        job = state.store.create(
            id=uuid.uuid4().hex, mode="text", prompt=body.prompt.strip(), quality=body.quality,
            provider=state.provider_name,
        )
        state.runner.submit(job["id"], GenerationRequest(mode="text", prompt=job["prompt"], quality=body.quality))
        return to_out(job)

    @app.post("/api/jobs/image", response_model=JobOut, status_code=201)
    async def create_image_job(
        request: Request,
        image: UploadFile = File(...),
        prompt: str | None = Form(None, max_length=500),
        quality: Quality = Form("standard"),
    ):
        ext = IMAGE_TYPES.get(image.content_type or "")
        if not ext:
            raise HTTPException(415, "Formato no soportado: usa PNG, JPG o WEBP")
        data = await image.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"La imagen supera {settings.max_upload_bytes // (1024 * 1024)} MB")
        if not data:
            raise HTTPException(400, "La imagen está vacía")

        state = request.app.state
        job_id = uuid.uuid4().hex
        job_dir = settings.files_dir / job_id
        job_dir.mkdir(parents=True)
        (job_dir / f"input.{ext}").write_bytes(data)
        job = state.store.create(
            id=job_id, mode="image", prompt=(prompt or "").strip() or None, quality=quality,
            provider=state.provider_name, input_image=f"input.{ext}",
        )
        state.runner.submit(
            job_id,
            GenerationRequest(
                mode="image", prompt=job["prompt"], quality=quality, image_bytes=data, image_mime=image.content_type
            ),
        )
        return to_out(job)

    @app.get("/api/jobs", response_model=list[JobOut])
    async def list_jobs(request: Request, limit: int = 100):
        return [to_out(j) for j in request.app.state.store.list(min(max(limit, 1), 500))]

    @app.get("/api/jobs/{job_id}", response_model=JobOut)
    async def get_job(request: Request, job_id: str):
        return to_out(get_job_or_404(request, job_id))

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(request: Request, job_id: str):
        get_job_or_404(request, job_id)
        request.app.state.store.delete(job_id)
        shutil.rmtree(settings.files_dir / job_id, ignore_errors=True)

    @app.get("/api/jobs/{job_id}/model.glb")
    async def download_model(request: Request, job_id: str):
        job = get_job_or_404(request, job_id)
        if not job["model_file"]:
            raise HTTPException(409, "El modelo todavía no está listo")
        return FileResponse(
            settings.files_dir / job_id / job["model_file"],
            media_type="model/gltf-binary",
            filename=f"forja3d-{job_id[:8]}.glb",
        )

    @app.get("/api/jobs/{job_id}/image")
    async def download_image(request: Request, job_id: str):
        job = get_job_or_404(request, job_id)
        if not job["input_image"]:
            raise HTTPException(404, "Este trabajo no tiene imagen")
        return FileResponse(settings.files_dir / job_id / job["input_image"])

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
