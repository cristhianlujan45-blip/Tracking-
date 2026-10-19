"""AI generation on your own GPU, through the Forja3D GPU worker (``gpu_worker/``).

The worker runs Hunyuan3D-2mini (shape) and SD-Turbo (concepts) on the local machine. This
provider talks to it over HTTP: it submits a task, polls its real progress and downloads the GLB.
Capabilities come from the worker itself, so the UI only shows what your GPU setup can do.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import httpx

from ..styles import build_image_prompt
from .base import GenerationError, GenerationRequest, GenerationResult, InputImage, ProgressCallback

log = logging.getLogger(__name__)

DEFAULT_CAPABILITIES = frozenset({"text", "image", "concepts"})
NEGATIVE_PROMPT = "blurry, cropped, multiple objects, text, watermark, busy background"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class LocalProvider:
    name = "local"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8001",
        token: str = "",
        timeout_s: float = 1800,
        poll_interval: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout_s = timeout_s
        self.poll_interval = poll_interval
        self.transport = transport  # tests inject the worker app in-process
        self.capabilities = DEFAULT_CAPABILITIES
        self.info: dict = {}

    def _client(self, timeout: float = 120) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.url, headers=self.headers, timeout=timeout, transport=self.transport)

    async def start(self) -> None:
        """Ask the worker what it can do. A worker that is not up yet is not fatal."""
        try:
            async with self._client(timeout=5) as client:
                resp = await client.get("/health")
                resp.raise_for_status()
                self.info = resp.json()
                self.capabilities = frozenset(self.info.get("capabilities", DEFAULT_CAPABILITIES))
                log.info("Worker GPU: %s · %s", self.info.get("gpu"), ", ".join(sorted(self.capabilities)))
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("No se pudo contactar con el worker en %s (%s); se reintentará al generar", self.url, exc)

    # ------------------------------------------------------------ provider API

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        async with self._client() as client:
            reference = None
            images = req.images
            span = (5, 97)
            if req.mode == "text":
                await on_progress(3, "Generando imagen de referencia en tu GPU")
                reference = (await self._images(client, req, 1))[0]
                images = [InputImage(reference, "image/png")]
                span = (15, 97)
            if not images:
                raise GenerationError("No hay imagen de entrada")
            payload = {"images": [_b64(i.data) for i in images], "seed": req.seed % 2**32, "quality": req.quality}
            glb = await self._run_task(client, payload, on_progress, span)
        return GenerationResult(model_glb=glb, reference_image=reference)

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        async with self._client() as client:
            return await self._images(client, req, count)

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        async with self._client() as client:
            await on_progress(3, "Generando imagen de la textura")
            image = (await self._images(client, req, 1))[0]
            payload = {"kind": "texture", "images": [_b64(image)], "mesh_glb": _b64(glb), "seed": req.seed % 2**32}
            return await self._run_task(client, payload, on_progress, (15, 97))

    async def estimate_depth(self, image: InputImage) -> bytes:
        async with self._client() as client:
            resp = await self._call(client.post("/depth", json={"image": _b64(image.data)}, timeout=300))
            return base64.b64decode(resp.json()["depth"])

    # ------------------------------------------------------------ helpers

    async def _call(self, coro):
        try:
            resp = await coro
        except httpx.ConnectError as exc:
            raise GenerationError(
                f"No se puede conectar con el worker de IA local en {self.url}. ¿Está iniciado? (gpu_worker/README.md)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError("El worker de IA local no respondió a tiempo") from exc
        if not resp.is_success:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise GenerationError(f"Worker local ({resp.status_code}): {str(detail)[:300]}")
        return resp

    async def _images(self, client: httpx.AsyncClient, req: GenerationRequest, count: int) -> list[bytes]:
        body = {"prompt": build_image_prompt(req), "count": count, "seed": req.seed % 2**32, "negative": NEGATIVE_PROMPT}
        resp = await self._call(client.post("/images", json=body, timeout=600))
        return [base64.b64decode(i) for i in resp.json()["images"]]

    async def _run_task(self, client: httpx.AsyncClient, payload: dict, on_progress: ProgressCallback,
                        span: tuple[int, int]) -> bytes:
        task_id = (await self._call(client.post("/tasks", json=payload))).json()["id"]
        lo, hi = span
        started = time.monotonic()
        try:
            while True:
                if time.monotonic() - started > self.timeout_s:
                    raise GenerationError("La GPU local tardó demasiado")
                task = (await self._call(client.get(f"/tasks/{task_id}"))).json()
                if task["status"] == "failed":
                    raise GenerationError(task.get("error") or "La GPU local falló")
                if task["status"] == "succeeded":
                    break
                await on_progress(int(lo + (hi - lo) * task["progress"] / 100), task["stage"])
                await asyncio.sleep(self.poll_interval)
            await on_progress(hi, "Descargando modelo")
            return (await self._call(client.get(f"/tasks/{task_id}/result"))).content
        finally:
            try:
                await client.delete(f"/tasks/{task_id}")
            except httpx.HTTPError:
                pass
