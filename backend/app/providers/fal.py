"""Real AI generation through fal.ai (https://fal.ai/models).

* image -> 3D:       TRELLIS (``fal-ai/trellis``) or Hunyuan3D-2 (``fal-ai/hunyuan3d/v2``)
* multi-image -> 3D: TRELLIS multi-view (``fal-ai/trellis/multi``)
* text -> 3D:        text -> image with FLUX, then image -> 3D (the two-step pipeline most
  commercial tools use). The concept images endpoint exposes the first step on its own so the
  user can pick a design before spending on the 3D step.
"""

from __future__ import annotations

import asyncio
import base64
import time

import httpx

from ..styles import build_image_prompt
from .base import GenerationError, GenerationRequest, GenerationResult, InputImage, ProgressCallback

QUEUE_URL = "https://queue.fal.run"
SYNC_URL = "https://fal.run"

# quality -> model specific parameters
TRELLIS_QUALITY = {
    "draft": {"texture_size": 512, "mesh_simplify": 0.97, "ss_sampling_steps": 8, "slat_sampling_steps": 8},
    "standard": {"texture_size": 1024, "mesh_simplify": 0.95, "ss_sampling_steps": 12, "slat_sampling_steps": 12},
    "high": {"texture_size": 2048, "mesh_simplify": 0.9, "ss_sampling_steps": 20, "slat_sampling_steps": 20},
}
HUNYUAN_QUALITY = {
    "draft": {"num_inference_steps": 20, "octree_resolution": 192, "textured_mesh": False},
    "standard": {"num_inference_steps": 40, "octree_resolution": 256, "textured_mesh": True},
    "high": {"num_inference_steps": 50, "octree_resolution": 384, "textured_mesh": True},
}


def _data_uri(img: InputImage) -> str:
    return f"data:{img.mime};base64,{base64.b64encode(img.data).decode()}"


def _image_ext(data: bytes) -> str:
    return "jpg" if data.startswith(b"\xff\xd8") else "png"


class FalProvider:
    name = "fal"
    capabilities = frozenset({"text", "image", "multi_image", "concepts"})

    def __init__(
        self,
        api_key: str,
        image_to_3d_model: str,
        text_to_image_model: str,
        multi_image_model: str = "fal-ai/trellis/multi",
        timeout_s: float = 600,
    ):
        if not api_key:
            raise GenerationError("Falta FAL_KEY: configura tu clave de fal.ai o usa GEN_PROVIDER=mock")
        self.headers = {"Authorization": f"Key {api_key}"}
        self.image_to_3d_model = image_to_3d_model
        self.text_to_image_model = text_to_image_model
        self.multi_image_model = multi_image_model
        self.timeout_s = timeout_s

    # ------------------------------------------------------------ public API

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        async with httpx.AsyncClient(headers=self.headers, timeout=60) as client:
            reference_image: bytes | None = None
            if req.mode == "text":
                await on_progress(5, "Generando imagen de referencia")
                result = await self._run(client, self.text_to_image_model, self._image_payload(req, 1), on_progress, (5, 30))
                reference_image = await self._download(client, result["images"][0]["url"])
                images = [InputImage(reference_image, f"image/{'jpeg' if _image_ext(reference_image) == 'jpg' else 'png'}")]
            else:
                images = req.images
            if not images:
                raise GenerationError("No hay imagen de entrada")

            await on_progress(35, "Reconstruyendo modelo 3D")
            if req.mode == "multi_image" and len(images) > 1:
                model = self.multi_image_model
                payload = {
                    "image_urls": [_data_uri(i) for i in images],
                    "multiimage_algo": "stochastic",
                    "seed": req.seed,
                    **TRELLIS_QUALITY.get(req.quality, TRELLIS_QUALITY["standard"]),
                }
            else:
                model = self.image_to_3d_model
                payload = self._mesh_payload(_data_uri(images[0]), req)
            result = await self._run(client, model, payload, on_progress, (35, 90))

            mesh_url = (result.get("model_mesh") or result.get("model_glb") or {}).get("url")
            if not mesh_url:
                raise GenerationError("El modelo no devolvió ninguna malla")
            await on_progress(92, "Descargando modelo")
            glb = await self._download(client, mesh_url)
            return GenerationResult(
                model_glb=glb,
                reference_image=reference_image,
                reference_image_ext=_image_ext(reference_image or b""),
            )

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        async with httpx.AsyncClient(headers=self.headers, timeout=120) as client:
            resp = await client.post(f"{SYNC_URL}/{self.text_to_image_model}", json=self._image_payload(req, count))
            self._raise_for(resp, self.text_to_image_model)
            urls = [img["url"] for img in resp.json().get("images", [])]
            if not urls:
                raise GenerationError("El generador de imágenes no devolvió resultados")
            return list(await asyncio.gather(*(self._download(client, u) for u in urls)))

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        raise GenerationError("El retexturizado no está disponible con fal.ai; usa otro proveedor")

    # ------------------------------------------------------------ helpers

    def _image_payload(self, req: GenerationRequest, count: int) -> dict:
        return {"prompt": build_image_prompt(req), "image_size": "square_hd", "num_images": count, "seed": req.seed}

    def _mesh_payload(self, image_url: str, req: GenerationRequest) -> dict:
        if "hunyuan" in self.image_to_3d_model:
            params = HUNYUAN_QUALITY.get(req.quality, HUNYUAN_QUALITY["standard"])
            return {"input_image_url": image_url, "seed": req.seed, **params}
        return {"image_url": image_url, "seed": req.seed, **TRELLIS_QUALITY.get(req.quality, TRELLIS_QUALITY["standard"])}

    async def _run(
        self, client: httpx.AsyncClient, model: str, payload: dict, on_progress: ProgressCallback, span: tuple[int, int]
    ) -> dict:
        """Submit to the fal queue, poll until done, return the JSON result."""
        resp = await client.post(f"{QUEUE_URL}/{model}", json=payload)
        self._raise_for(resp, model)
        submitted = resp.json()
        status_url, response_url = submitted["status_url"], submitted["response_url"]

        lo, hi = span
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed > self.timeout_s:
                raise GenerationError(f"Tiempo de espera agotado en {model}")
            status = await client.get(status_url)
            self._raise_for(status, model)
            state = status.json().get("status")
            if state == "COMPLETED":
                break
            # fal does not report a percentage; approach `hi` asymptotically so the bar keeps moving
            fraction = 1 - 1 / (1 + elapsed / 30)
            label = "En cola" if state == "IN_QUEUE" else "Procesando"
            await on_progress(int(lo + (hi - lo) * fraction), f"{label} ({model})")
            await asyncio.sleep(2)

        result = await client.get(response_url)
        self._raise_for(result, model)
        return result.json()

    async def _download(self, client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _raise_for(resp: httpx.Response, model: str) -> None:
        if resp.is_success:
            return
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise GenerationError(f"fal.ai ({model}) respondió {resp.status_code}: {str(detail)[:300]}")
