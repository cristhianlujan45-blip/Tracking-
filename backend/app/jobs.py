"""Background job runner: limits concurrency and records progress in the store."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable

from .db import JobStore
from .geometry import GeometryError
from .providers import GenerationError, GenerationResult
from .providers.base import ProgressCallback

log = logging.getLogger(__name__)

# A unit of work: receives a progress callback, returns the produced model.
Work = Callable[[ProgressCallback], Awaitable[GenerationResult]]


class JobRunner:
    def __init__(self, store: JobStore, files_dir: Path, max_concurrent: int):
        self.store = store
        self.files_dir = files_dir
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._tasks: set[asyncio.Task] = set()
        self._by_job: dict[str, asyncio.Task] = {}

    def submit(self, job_id: str, work: Work, after: str | None = None) -> None:
        """Run `work` in the background; with `after`, only once that job has finished (either way)."""
        task = asyncio.create_task(self._run(job_id, work, self._by_job.get(after) if after else None))
        self._tasks.add(task)
        self._by_job[job_id] = task
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda _t: self._by_job.pop(job_id, None))

    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    def cancel(self, job_id: str) -> None:
        task = self._by_job.get(job_id)
        if task is not None:
            task.cancel()

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run(self, job_id: str, work: Work, after: asyncio.Task | None = None) -> None:
        if after is not None:  # wait outside the semaphore so a chain never blocks itself
            self.store.update(job_id, stage="Esperando el paso anterior")
            await asyncio.gather(after, return_exceptions=True)
        async with self._semaphore:
            if self.store.get(job_id) is None:  # deleted while queued
                return
            self.store.update(job_id, status="running", stage="Iniciando", progress=1)

            async def on_progress(progress: int, stage: str) -> None:
                if self.store.get(job_id) is not None:
                    self.store.update(job_id, progress=max(1, min(99, progress)), stage=stage)

            try:
                result = await work(on_progress)
            except asyncio.CancelledError:
                self.store.update(job_id, status="failed", error="Cancelado")
                raise
            except (GenerationError, GeometryError) as exc:
                self.store.update(job_id, status="failed", error=str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - never leave a job hanging
                log.exception("Job %s crashed", job_id)
                detail = f"{exc.__class__.__name__}: {exc}".strip().rstrip(":")
                self.store.update(job_id, status="failed",
                                  error=f"Error interno durante el procesamiento ({detail[:300]})")
                return

            if self.store.get(job_id) is None:
                return
            job_dir = self.files_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "model.glb").write_bytes(result.model_glb)
            fields = {"status": "succeeded", "progress": 100, "stage": "Completado", "model_file": "model.glb"}
            for name, data in result.extra_files.items():
                (job_dir / name).write_bytes(data)
            if "preview.png" in result.extra_files:
                fields["thumbnail"] = "preview.png"
            if result.reference_image:
                name = f"reference.{result.reference_image_ext}"
                (job_dir / name).write_bytes(result.reference_image)
                fields["input_image"] = name
            self.store.update(job_id, **fields)
