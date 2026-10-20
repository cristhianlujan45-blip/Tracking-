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


# generated views by role and the azimuth each role stands for (see projection.AZIMUTHS)
VIEW_AZIMUTHS = {"left": 90, "back": 180, "right": 270}
# the multi-view shape model expects square side/back views: farther off ones only paint the model
SHAPE_MAX_OFF_AXIS = 15


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _off_axis(role: str, poses: dict[str, tuple[float, float]]) -> float:
    azimuth = poses.get(role, (VIEW_AZIMUTHS[role], 0))[0]
    return abs((azimuth - VIEW_AZIMUTHS[role] + 180) % 360 - 180)


class LocalProvider:
    name = "local"
    # Hunyuan3D-2mini without its paint model (8 GB GPUs) returns bare shapes: colour them from the inputs
    paints_from_views = True

    def __init__(
        self,
        url: str = "http://127.0.0.1:8001",
        token: str = "",
        timeout_s: float = 1800,
        poll_interval: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
        launcher=None,
    ):
        """`launcher`: the app's LocalAI, which installs and starts the worker by itself when needed."""
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout_s = timeout_s
        self.poll_interval = poll_interval
        self.transport = transport  # tests inject the worker app in-process
        self.launcher = launcher
        self.capabilities = DEFAULT_CAPABILITIES
        self.info: dict = {}
        self.engines = [{"id": "hunyuan2", "label": "Hunyuan3D-2 (tu GPU)",
                         "description": "Gratis y privado, en tu tarjeta gráfica.", "native_text": False,
                         "max_views": 4, "tags": ["local"]}]

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
                if any("mini" in m for m in self.info.get("models", [])[:1]):
                    self.engines = [{**self.engines[0], "id": "hunyuan2mini", "label": "Hunyuan3D-2mini (tu GPU)"}]
                log.info("Worker GPU: %s · %s", self.info.get("gpu"), ", ".join(sorted(self.capabilities)))
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("No se pudo contactar con el worker en %s (%s); se reintentará al generar", self.url, exc)

    async def _ready(self) -> None:
        """With the built-in launcher, start the worker if it is not running yet (first use or after a crash)."""
        if self.launcher is None:
            return
        from ..local_ai import LocalAIError

        try:
            self.url = (await self.launcher.ensure_running()).rstrip("/")
        except LocalAIError as exc:
            raise GenerationError(str(exc)) from exc
        if not self.info:
            await self.start()

    # ------------------------------------------------------------ provider API

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        await self._ready()
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
            views, poses = {}, {}
            if len(images) == 1 and {"views", "multi_image"} <= self.capabilities:
                await on_progress(span[0], "Generando vistas laterales y de espalda")
                views, poses = await self._views(client, images[0].data, req.quality)
                # only views seen square from their side drive the shape; the rest just paint it
                square = [r for r in VIEW_AZIMUTHS if r in views and _off_axis(r, poses) <= SHAPE_MAX_OFF_AXIS]
                if square:
                    payload["images"] += [_b64(views[r]) for r in square]
                    payload["roles"] = ["front", *square]
                span = (span[0] + 10, span[1])
            glb = await self._run_task(client, payload, on_progress, span)
        return GenerationResult(model_glb=glb, reference_image=reference, views=views, view_poses=poses,
                                extra_files={f"view_{role}.png": data for role, data in views.items()})

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        await self._ready()
        async with self._client() as client:
            return await self._images(client, req, count)

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        await self._ready()
        async with self._client() as client:
            await on_progress(3, "Generando imagen de la textura")
            image = (await self._images(client, req, 1))[0]
            payload = {"kind": "texture", "images": [_b64(image)], "mesh_glb": _b64(glb), "seed": req.seed % 2**32}
            return await self._run_task(client, payload, on_progress, (15, 97))

    async def estimate_depth(self, image: InputImage) -> bytes:
        await self._ready()
        async with self._client() as client:
            resp = await self._call(client.post("/depth", json={"image": _b64(image.data)}, timeout=300))
            return base64.b64decode(resp.json()["depth"])

    async def upscale_image(self, png: bytes, scale: int) -> bytes:
        await self._ready()
        async with self._client() as client:
            resp = await self._call(client.post("/upscale", json={"image": _b64(png), "scale": scale}, timeout=900))
            return base64.b64decode(resp.json()["image"])

    # ------------------------------------------------------------ helpers

    async def _call(self, coro):
        try:
            resp = await coro
        except httpx.ConnectError as exc:
            raise GenerationError(
                f"No se puede conectar con la IA local en {self.url}: ábrela desde Ajustes → «Mi GPU»"
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
        body = {"prompt": build_image_prompt(req), "count": count, "seed": req.seed % 2**32, "negative": NEGATIVE_PROMPT,
                "subject": (req.prompt or "").strip()}
        resp = await self._call(client.post("/images", json=body, timeout=600))
        return [base64.b64decode(i) for i in resp.json()["images"]]

    async def _views(self, client: httpx.AsyncClient, image: bytes,
                     quality: str) -> tuple[dict[str, bytes], dict[str, tuple[float, float]]]:
        """AI side and back views of a single picture. Optional: on failure the picture is used alone."""
        try:
            resp = await self._call(client.post("/views", json={"image": _b64(image), "quality": quality}, timeout=900))
        except GenerationError as exc:
            log.warning("No se pudieron generar las vistas laterales: %s", exc)
            return {}, {}
        body = resp.json()
        views = {role: base64.b64decode(data) for role, data in body.get("views", {}).items() if role in VIEW_AZIMUTHS}
        poses = {role: (float(p[0]), float(p[1])) for role, p in body.get("poses", {}).items() if role in views}
        return views, poses

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
