"""Cloud AI generation through fal.ai with today's leading 3D engines.

Engines (selectable per job, see ENGINES):

* Hunyuan3D 3      — Tencent, native text-to-3D and image-to-3D, PBR materials, up to 1.5 M faces,
                     up to 4 reference views.
* TRELLIS 2        — Microsoft, image-to-3D with textures up to 4096 px.
* Tripo H3.1       — Tripo, PBR, "detailed" geometry and texture quality.
* Meshy 6          — Meshy, PBR, A/T-pose for characters, symmetry control.
* TRELLIS (v1)     — the previous generation, cheapest.

Retexturing an existing model uses Meshy 7 Retexture (2K / 4K / 8K, PBR, keeps the UVs).
Text-to-3D for engines without a native text mode goes text → image (FLUX) → 3D, the same
two-step pipeline the commercial tools use; the concept images endpoint exposes the first step.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from ..styles import STYLES, build_image_prompt
from .base import GenerationError, GenerationRequest, GenerationResult, InputImage, ProgressCallback

QUEUE_URL = "https://queue.fal.run"
SYNC_URL = "https://fal.run"
RETEXTURE_MODEL = "meshy/v7/retexture"
RETEXTURE_RESOLUTION = {"draft": "2k", "standard": "4k", "high": "8k"}


def _data_uri(img: InputImage) -> str:
    return f"data:{img.mime};base64,{base64.b64encode(img.data).decode()}"


def _image_ext(data: bytes) -> str:
    return "jpg" if data.startswith(b"\xff\xd8") else "png"


# ---------------------------------------------------------------- engine payloads
# Each builder gets the request and the image data URIs (first = main view) and returns the payload.

Payload = Callable[[GenerationRequest, list[str]], dict]


def _hunyuan3d_v3(req: GenerationRequest, images: list[str]) -> dict:
    faces = {"draft": 100_000, "standard": 500_000, "high": 1_000_000}[req.quality]
    payload = {
        "input_image_url": images[0],
        "enable_pbr": req.quality != "draft",
        "face_count": faces,
        "generate_type": "LowPoly" if req.style == "lowpoly" else "Normal",
    }
    # each picture in its own slot (front + back must not land on the side)
    for role, url in zip(req.roles or ("front", "left", "back", "right"), images):
        if role != "front":
            payload[f"{role}_image_url"] = url
    return payload


def _hunyuan3d_v3_text(req: GenerationRequest, images: list[str]) -> dict:
    faces = {"draft": 100_000, "standard": 500_000, "high": 1_000_000}[req.quality]
    style = STYLES.get(req.style, STYLES["auto"])[1]
    prompt = ", ".join(p for p in ((req.prompt or "").strip(), style) if p)[:1024]
    return {"prompt": prompt, "enable_pbr": req.quality != "draft", "face_count": faces,
            "generate_type": "LowPoly" if req.style == "lowpoly" else "Normal"}


def _trellis2(req: GenerationRequest, images: list[str]) -> dict:
    q = {"draft": (512, 1024, 200_000), "standard": (1024, 2048, 500_000), "high": (1536, 4096, 1_000_000)}
    resolution, texture, faces = q[req.quality]
    return {"image_url": images[0], "seed": req.seed, "resolution": resolution, "texture_size": texture,
            "decimation_target": faces}


def _tripo_h31(req: GenerationRequest, images: list[str]) -> dict:
    detailed = "detailed" if req.quality == "high" else "standard"
    return {"image_url": images[0], "pbr": req.quality != "draft", "texture": True,
            "texture_quality": detailed, "geometry_quality": detailed, "model_seed": req.seed,
            "texture_seed": req.seed, "orientation": "align_image"}


def _meshy6(req: GenerationRequest, images: list[str]) -> dict:
    poly = {"draft": 30_000, "standard": 100_000, "high": 300_000}[req.quality]
    payload = {"image_url": images[0], "enable_pbr": req.quality != "draft", "target_polycount": poly,
               "symmetry_mode": "on" if req.symmetry else "auto", "should_texture": True,
               "model_type": "lowpoly" if req.style == "lowpoly" else "standard"}
    if req.pose in ("a-pose", "t-pose"):
        payload["pose_mode"] = req.pose
    return payload


def _trellis(req: GenerationRequest, images: list[str]) -> dict:
    q = {"draft": (512, 0.97, 8), "standard": (1024, 0.95, 12), "high": (2048, 0.9, 20)}
    texture, simplify, steps = q[req.quality]
    if len(images) > 1:
        return {"image_urls": images, "multiimage_algo": "stochastic", "seed": req.seed, "texture_size": texture,
                "mesh_simplify": simplify, "ss_sampling_steps": steps, "slat_sampling_steps": steps}
    return {"image_url": images[0], "seed": req.seed, "texture_size": texture, "mesh_simplify": simplify,
            "ss_sampling_steps": steps, "slat_sampling_steps": steps}


@dataclass
class Engine:
    id: str
    label: str
    description: str
    image_model: str
    image_payload: Payload
    text_model: str | None = None  # native text-to-3D (skips the FLUX reference image)
    text_payload: Payload | None = None
    multi_model: str | None = None  # model used when several views are given
    max_views: int = 1
    tags: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "native_text": self.text_model is not None, "max_views": self.max_views, "tags": self.tags}


ENGINES: dict[str, Engine] = {e.id: e for e in [
    Engine("hunyuan3d_v3", "Hunyuan3D 3 (Tencent)", "Texto o imagen → 3D nativo, PBR, hasta 1 M de caras, hasta 4 vistas.",
           "fal-ai/hunyuan3d-v3/image-to-3d", _hunyuan3d_v3, "fal-ai/hunyuan3d-v3/text-to-3d", _hunyuan3d_v3_text,
           "fal-ai/hunyuan3d-v3/image-to-3d", 4, ["PBR", "texto nativo"]),
    Engine("trellis2", "TRELLIS 2 (Microsoft)", "Geometría muy fiel a la imagen y texturas de hasta 4K.",
           "fal-ai/trellis-2", _trellis2, tags=["4K"]),
    Engine("tripo_h31", "Tripo H3.1", "PBR y calidad «detailed» de geometría y textura.",
           "tripo3d/h3.1/image-to-3d", _tripo_h31, tags=["PBR", "detalle"]),
    Engine("meshy6", "Meshy 6", "PBR, simetría y pose A/T para personajes listos para rig.",
           "fal-ai/meshy/v6-preview/image-to-3d", _meshy6, tags=["PBR", "personajes"]),
    Engine("trellis", "TRELLIS 1 (económico)", "La generación anterior: rápida y barata.",
           "fal-ai/trellis", _trellis, multi_model="fal-ai/trellis/multi", max_views=4, tags=["barato"]),
]}
DEFAULT_ENGINE = "hunyuan3d_v3"


def _mesh_url(result: dict) -> str | None:
    for key in ("model_glb", "model_mesh"):
        if isinstance(result.get(key), dict) and result[key].get("url"):
            return result[key]["url"]
    glb = (result.get("model_urls") or {}).get("glb")
    return glb.get("url") if isinstance(glb, dict) else None


class FalProvider:
    name = "fal"
    capabilities = frozenset({"text", "image", "multi_image", "concepts", "depth", "retexture", "upscale"})

    def __init__(
        self,
        api_key: str,
        text_to_image_model: str = "fal-ai/flux/schnell",
        depth_model: str = "fal-ai/imageutils/depth",
        default_engine: str = DEFAULT_ENGINE,
        timeout_s: float = 900,
        transport: httpx.AsyncBaseTransport | None = None,
        poll_interval: float = 2.0,
    ):
        if not api_key:
            raise GenerationError("Falta FAL_KEY: configura tu clave de fal.ai o usa GEN_PROVIDER=mock")
        self.headers = {"Authorization": f"Key {api_key}"}
        self.text_to_image_model = text_to_image_model
        self.depth_model = depth_model
        self.default_engine = default_engine if default_engine in ENGINES else DEFAULT_ENGINE
        self.timeout_s = timeout_s
        self.transport = transport
        self.poll_interval = poll_interval
        self.engines = [e.public() for e in ENGINES.values()]

    def _client(self, timeout: float = 60) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self.headers, timeout=timeout, transport=self.transport)

    # ------------------------------------------------------------ public API

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        engine = ENGINES.get(req.engine or self.default_engine) or ENGINES[self.default_engine]
        async with self._client() as client:
            reference_image: bytes | None = None
            if req.mode == "text" and engine.text_model:
                await on_progress(5, f"Generando con {engine.label}")
                result = await self._run(client, engine.text_model, engine.text_payload(req, []), on_progress, (5, 90))
            else:
                if req.mode == "text":
                    await on_progress(5, "Generando imagen de referencia")
                    image = await self._run(client, self.text_to_image_model, self._image_payload(req, 1), on_progress,
                                            (5, 30))
                    reference_image = await self._download(client, image["images"][0]["url"])
                    mime = "image/jpeg" if _image_ext(reference_image) == "jpg" else "image/png"
                    images = [InputImage(reference_image, mime)]
                else:
                    images = req.images
                if not images:
                    raise GenerationError("No hay imagen de entrada")
                uris = [_data_uri(i) for i in images[: engine.max_views]]
                model = engine.multi_model if len(uris) > 1 and engine.multi_model else engine.image_model
                await on_progress(35, f"Reconstruyendo con {engine.label}")
                result = await self._run(client, model, engine.image_payload(req, uris), on_progress, (35, 90))

            url = _mesh_url(result)
            if not url:
                raise GenerationError(f"{engine.label} no devolvió ninguna malla")
            await on_progress(92, "Descargando modelo")
            glb = await self._download(client, url)
            return GenerationResult(model_glb=glb, reference_image=reference_image,
                                    reference_image_ext=_image_ext(reference_image or b""))

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        async with self._client(timeout=120) as client:
            resp = await client.post(f"{SYNC_URL}/{self.text_to_image_model}", json=self._image_payload(req, count))
            self._raise_for(resp, self.text_to_image_model)
            urls = [img["url"] for img in resp.json().get("images", [])]
            if not urls:
                raise GenerationError("El generador de imágenes no devolvió resultados")
            return list(await asyncio.gather(*(self._download(client, u) for u in urls)))

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        """Meshy 7 Retexture: new PBR textures (2K/4K/8K) on the existing geometry and UVs."""
        payload = {
            "model_url": f"data:model/gltf-binary;base64,{base64.b64encode(glb).decode()}",
            "texture_resolution": RETEXTURE_RESOLUTION.get(req.quality, "4k"),
            "enable_original_uv": True,
            "enable_pbr": True,
        }
        style = STYLES.get(req.style, STYLES["auto"])[1]
        if req.images:
            payload["image_style_url"] = _data_uri(req.images[0])
        payload["text_style_prompt"] = ", ".join(p for p in ((req.prompt or "").strip(), style) if p)[:600] or "detailed"
        async with self._client() as client:
            await on_progress(5, "Enviando a Meshy 7 Retexture")
            result = await self._run(client, RETEXTURE_MODEL, payload, on_progress, (5, 92))
            url = _mesh_url(result)
            if not url:
                raise GenerationError("El retexturizado no devolvió ningún modelo")
            await on_progress(94, "Descargando texturas")
            return await self._download(client, url)

    async def estimate_depth(self, image: InputImage) -> bytes:
        async with self._client(timeout=120) as client:
            resp = await client.post(f"{SYNC_URL}/{self.depth_model}", json={"image_url": _data_uri(image)})
            self._raise_for(resp, self.depth_model)
            url = (resp.json().get("image") or {}).get("url")
            if not url:
                raise GenerationError("El modelo de profundidad no devolvió ningún mapa")
            return await self._download(client, url)

    async def upscale_image(self, png: bytes, scale: int) -> bytes:
        payload = {"image_url": _data_uri(InputImage(png, "image/png")), "scale": scale,
                   "model": "RealESRGAN_x4plus", "tile": 512, "output_format": "png"}
        async with self._client(timeout=300) as client:
            resp = await client.post(f"{SYNC_URL}/fal-ai/esrgan", json=payload)
            self._raise_for(resp, "fal-ai/esrgan")
            url = (resp.json().get("image") or {}).get("url")
            if not url:
                raise GenerationError("El ampliador no devolvió ninguna imagen")
            return await self._download(client, url)

    # ------------------------------------------------------------ helpers

    def _image_payload(self, req: GenerationRequest, count: int) -> dict:
        return {"prompt": build_image_prompt(req), "image_size": "square_hd", "num_images": count, "seed": req.seed}

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
            await asyncio.sleep(self.poll_interval)

        result = await client.get(response_url)
        self._raise_for(result, model)
        return result.json()

    async def _download(self, client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url, follow_redirects=True, timeout=300)
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
        if resp.status_code == 403 and "balance" in str(detail).lower():
            raise GenerationError("Se acabó el saldo de fal.ai (nube). Usa la IA gratis de tu tarjeta gráfica: "
                                  "Ajustes → «Mi GPU · gratis»")
        raise GenerationError(f"fal.ai ({model}) respondió {resp.status_code}: {str(detail)[:300]}")
