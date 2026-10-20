"""Model backends for the local GPU worker.

* ``dummy``   – no GPU, no downloads: returns procedural meshes/images. Used to test the plumbing.
* ``hunyuan`` – Hunyuan3D-2mini (shape), optional Hunyuan3D-2mv (multi-view) and Hunyuan3D-Paint
                (texture), plus SD-Turbo / SDXL-Turbo for concept images. Tuned for 8 GB of VRAM:
                only one model lives on the GPU at a time, the rest wait in system RAM.

Heavy imports (torch, diffusers, hy3dgen) happen lazily inside the hunyuan backend so the worker
and its tests run on machines without them.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

# progress (0-100), stage label
Progress = Callable[[int, str], None]

VIEW_NAMES = ["front", "left", "back", "right"]


@dataclass
class MeshJob:
    images: list[Image.Image]
    seed: int = 0
    quality: str = "standard"
    # retexture: paint this GLB with the first image instead of generating a new shape
    mesh_glb: bytes | None = None
    extras: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def capabilities(self) -> set[str]: ...

    def info(self) -> dict: ...

    def generate_images(self, prompt: str, count: int, seed: int, negative: str = "") -> list[Image.Image]: ...

    def image_to_mesh(self, job: MeshJob, progress: Progress) -> bytes: ...

    def texture_mesh(self, job: MeshJob, progress: Progress) -> bytes: ...

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        """Grayscale depth map (mode L or I;16), brighter = closer."""
        ...

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        """Super-resolution ×2/×4 of a texture."""
        ...


def png_bytes(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ================================================================ dummy


class DummyBackend:
    """Fake GPU: deterministic shapes and images, with a short delay per stage."""

    name = "dummy"

    def __init__(self, delay: float = 0.2, texture: bool = True, multiview: bool = True):
        self.delay = delay
        self.texture = texture
        self.multiview = multiview

    def capabilities(self) -> set[str]:
        caps = {"text", "image", "concepts", "depth", "upscale"}
        if self.multiview:
            caps.add("multi_image")
        if self.texture:
            caps.add("retexture")
        return caps

    def info(self) -> dict:
        return {"gpu": "ninguna (dummy)", "vram_gb": 0, "models": ["dummy"]}

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        time.sleep(self.delay / 2)
        return image.convert("RGB").resize((image.width * scale, image.height * scale), Image.BICUBIC)

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        time.sleep(self.delay / 2)
        gray = np.asarray(image.convert("L"), dtype=float)
        rows = np.linspace(0, 255, gray.shape[0])[:, None]
        return Image.fromarray((0.6 * rows + 0.4 * gray).astype(np.uint8), "L")

    def generate_images(self, prompt: str, count: int, seed: int, negative: str = "") -> list[Image.Image]:
        images = []
        for i in range(count):
            rng = np.random.default_rng(seed + i)
            color = tuple(int(c) for c in rng.integers(40, 230, 3))
            img = Image.new("RGB", (512, 512), (245, 245, 247))
            ImageDraw.Draw(img).ellipse((120, 100, 392, 420), fill=color)
            images.append(img)
            time.sleep(self.delay / 2)
        return images

    def image_to_mesh(self, job: MeshJob, progress: Progress) -> bytes:
        import trimesh

        for pct, stage in [(10, "Quitando fondo"), (40, "Generando forma"), (80, "Limpiando malla")]:
            progress(pct, stage)
            time.sleep(self.delay)
        # color of the image centre, so different inputs give visibly different meshes
        r, g, b = job.images[0].convert("RGB").resize((1, 1)).getpixel((0, 0))
        mesh = trimesh.creation.icosphere(3, radius=0.5)
        mesh.apply_scale((1, 1 + (job.seed % 5) / 10, 1))
        mesh.visual.vertex_colors = np.tile([r, g, b, 255], (len(mesh.vertices), 1))
        return trimesh.Scene(mesh).export(file_type="glb", include_normals=True)

    def texture_mesh(self, job: MeshJob, progress: Progress) -> bytes:
        import trimesh

        progress(30, "Pintando textura")
        time.sleep(self.delay)
        scene = trimesh.load(io.BytesIO(job.mesh_glb), file_type="glb", force="scene")
        r, g, b = job.images[0].convert("RGB").resize((1, 1)).getpixel((0, 0))
        for geom in scene.geometry.values():
            geom.visual = trimesh.visual.ColorVisuals(geom, vertex_colors=np.tile([r, g, b, 255], (len(geom.vertices), 1)))
        return scene.export(file_type="glb", include_normals=True)


# ================================================================ hunyuan

# quality -> (diffusion steps, octree resolution, max faces after reduction)
HUNYUAN_QUALITY = {
    "draft": (15, 256, 20_000),
    "standard": (30, 320, 40_000),
    "high": (50, 380, 80_000),
}


class HunyuanBackend:
    """Hunyuan3D-2 family + SD-Turbo, scheduled so peak VRAM stays around 6–7 GB."""

    name = "hunyuan"

    def __init__(
        self,
        shape_model: str = "tencent/Hunyuan3D-2mini",
        shape_subfolder: str = "hunyuan3d-dit-v2-mini",
        image_model: str = "stabilityai/sd-turbo",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        texture: bool = False,
        multiview: bool = False,
        device: str = "cuda",
    ):
        import torch  # noqa: F401 - fail fast with a clear error if torch is missing

        self.shape_model, self.shape_subfolder = shape_model, shape_subfolder
        self.image_model = image_model
        self.depth_model = depth_model
        self._depth = None
        self._esrgan = None
        self.texture = texture
        self.multiview = multiview
        self.device = device
        self._shape = self._mv = self._paint = self._t2i = self._rembg = self._translator = None
        self._on_gpu: str | None = None

    # ------------------------------------------------------------ meta

    def capabilities(self) -> set[str]:
        caps = {"text", "image", "concepts", "depth", "upscale"}
        if self.multiview:
            caps.add("multi_image")
        if self.texture:
            caps.add("retexture")
        return caps

    def info(self) -> dict:
        import torch

        gpu, vram = "CPU", 0.0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu, vram = props.name, round(props.total_memory / 1024**3, 1)
        models = [f"{self.shape_model}/{self.shape_subfolder}", self.image_model, self.depth_model]
        if self.multiview:
            models.append("tencent/Hunyuan3D-2mv")
        if self.texture:
            models.append("tencent/Hunyuan3D-2 (paint)")
        return {"gpu": gpu, "vram_gb": vram, "models": models}

    # ------------------------------------------------------------ VRAM management

    def _activate(self, key: str) -> None:
        """Keep only one heavy pipeline on the GPU; park the previous one in system RAM."""
        import torch

        if self._on_gpu == key:
            return
        previous = getattr(self, f"_{self._on_gpu}", None) if self._on_gpu else None
        if previous is not None and hasattr(previous, "to"):
            previous.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        current = getattr(self, f"_{key}")
        if hasattr(current, "to"):
            current.to(self.device)
        self._on_gpu = key

    def _load_shape(self):
        if self._shape is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            log.info("Cargando %s …", self.shape_model)
            self._shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.shape_model, subfolder=self.shape_subfolder, use_safetensors=True, device="cpu"
            )
            _fast_decoding(self._shape)
        self._activate("shape")
        return self._shape

    def _load_mv(self):
        if self._mv is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            log.info("Cargando Hunyuan3D-2mv …")
            self._mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv", use_safetensors=True, device="cpu"
            )
            _fast_decoding(self._mv)
        self._activate("mv")
        return self._mv

    def _load_t2i(self):
        if self._t2i is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            log.info("Cargando %s …", self.image_model)
            self._t2i = AutoPipelineForText2Image.from_pretrained(
                self.image_model, torch_dtype=torch.float16, variant="fp16"
            )
        self._activate("t2i")
        return self._t2i

    def _load_paint(self):
        if self._paint is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            log.info("Cargando Hunyuan3D-Paint …")
            self._paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            # The paint pipeline needs ~16 GB if fully resident; offload streams it through 8 GB.
            if hasattr(self._paint, "enable_model_cpu_offload"):
                self._paint.enable_model_cpu_offload()
        # offloaded pipelines manage their own placement: just free whatever else is on the GPU
        self._activate("none")
        return self._paint

    _none = None  # sentinel for _activate("none")

    def to_english(self, text: str) -> str:
        """The image models only understand English: translate Spanish prompts on the CPU (Opus-MT, ~300 MB)."""
        if not looks_spanish(text):
            return text
        try:
            if self._translator is None:
                from transformers import pipeline

                self._translator = pipeline("translation", model=TRANSLATOR, device=-1)
            out = self._translator(text, max_length=256)[0]["translation_text"]
            log.info("Prompt traducido: %r → %r", text, out)
            return out
        except Exception:  # noqa: BLE001 - an untranslated prompt still works, just worse
            log.exception("No se pudo traducir el prompt")
            return text

    def _remove_background(self, img: Image.Image) -> Image.Image:
        if self._rembg is None:
            from hy3dgen.rembg import BackgroundRemover

            self._rembg = BackgroundRemover()
        img = img.convert("RGBA")
        alpha = np.asarray(img)[:, :, 3]
        return img if alpha.min() < 250 else self._rembg(img.convert("RGB"))  # keep existing cut-outs

    # ------------------------------------------------------------ tasks

    def generate_images(self, prompt: str, count: int, seed: int, negative: str = "") -> list[Image.Image]:
        import torch

        pipe = self._load_t2i()
        generator = torch.Generator(device="cpu").manual_seed(seed)  # CPU generator: reproducible across GPUs
        result = pipe(
            prompt=[prompt] * count, num_inference_steps=2, guidance_scale=0.0,
            width=512, height=512, generator=generator,
        )
        return list(result.images)

    def image_to_mesh(self, job: MeshJob, progress: Progress) -> bytes:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        steps, octree, max_faces = HUNYUAN_QUALITY.get(job.quality, HUNYUAN_QUALITY["standard"])
        progress(5, "Quitando fondo")
        images = [self._remove_background(img) for img in job.images]

        progress(25, "Generando forma 3D")
        try:
            mesh = self._shape_pass(images, job.seed, steps, octree, progress)
        except Exception as exc:  # noqa: BLE001 - only out-of-memory is retried
            if not _is_oom(exc):
                raise
            # plan B for small GPUs: free everything and retry lighter (smaller model, coarser grid)
            log.warning("Sin memoria de GPU; reintentando con un modelo más ligero")
            progress(30, "Poca memoria de GPU: reintentando en modo ligero")
            self._use_light_model()
            mesh = self._shape_pass(images[:1], job.seed, steps, min(octree, 320), progress, chunks=4000)

        progress(80, "Limpiando malla")
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=max_faces)

        if self.texture and job.quality != "draft":
            progress(85, "Texturizando (puede tardar)")
            mesh = self._load_paint()(mesh, image=images[0])
        return mesh.export(file_type="glb")

    def _shape_pass(self, images, seed: int, steps: int, octree: int, progress: Progress, chunks: int = 8000):
        import torch

        generator = torch.Generator(device="cpu").manual_seed(seed)
        if len(images) > 1 and self.multiview:
            progress(15, "Cargando modelo multi-vista")
            pipe = self._load_mv()
            image_arg = dict(zip(VIEW_NAMES, images))
        else:
            progress(15, "Cargando modelo de forma")
            pipe = self._load_shape()
            image_arg = images[0]
        return pipe(
            image=image_arg, num_inference_steps=steps, octree_resolution=octree,
            num_chunks=chunks, generator=generator, output_type="trimesh",
        )[0]

    def _use_light_model(self) -> None:
        """Switch to Hunyuan3D-2mini for the rest of the session and remember it for the next start."""
        import torch

        for key in ("shape", "mv", "t2i", "paint"):
            setattr(self, f"_{key}", None)
        self._on_gpu = None
        gc.collect()
        torch.cuda.empty_cache()
        if self.shape_model != LIGHT_SHAPE[0]:
            self.shape_model, self.shape_subfolder = LIGHT_SHAPE
            self.multiview = False  # the multi-view model is as big as the full one
            try:
                with open(LIGHT_MARKER, "w", encoding="utf-8") as fh:
                    fh.write("Hunyuan3D-2mini: la GPU se quedó sin memoria con el modelo completo\n")
            except OSError:
                pass

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        """Depth Anything V2. Small (~25 M params) stays on the GPU next to whatever else is loaded."""
        import torch

        if self._depth is None:
            from transformers import pipeline

            log.info("Cargando %s …", self.depth_model)
            self._depth = pipeline("depth-estimation", model=self.depth_model, device=0 if torch.cuda.is_available() else -1)
        with torch.inference_mode():
            result = self._depth(image.convert("RGB"))
        depth = result["predicted_depth"].squeeze().float().cpu().numpy()  # relative inverse depth: higher = closer
        depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
        img = Image.fromarray((depth * 65535).astype(np.uint16))  # 16-bit keeps smooth gradients for reliefs
        return img.resize(image.size, Image.BILINEAR) if img.size != image.size else img

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        """Real-ESRGAN ×4 (spandrel) in overlapping 256 px tiles so 8 GB of VRAM is plenty."""
        import torch
        from spandrel import ModelLoader

        if self._esrgan is None:
            path = esrgan_weights()
            self._esrgan = ModelLoader().load_from_file(path).model.eval()
        if self.device != "cpu":
            self._activate("none")  # free the big pipelines; the upscaler itself is ~70 MB
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        model = self._esrgan.to(self.device, dtype)
        src = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        _, _, h, w = src.shape
        tile, pad, s = 256, 16, 4
        out = torch.zeros((1, 3, h * s, w * s))
        with torch.inference_mode():
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    y0, x0 = max(y - pad, 0), max(x - pad, 0)
                    y1, x1 = min(y + tile + pad, h), min(x + tile + pad, w)
                    piece = model(src[:, :, y0:y1, x0:x1].to(self.device, dtype)).float().cpu()
                    th, tw = min(tile, h - y), min(tile, w - x)
                    out[:, :, y * s:(y + th) * s, x * s:(x + tw) * s] = piece[
                        :, :, (y - y0) * s:(y - y0 + th) * s, (x - x0) * s:(x - x0 + tw) * s]
        result = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        if scale != 4:
            result = result.resize((w * scale, h * scale), Image.LANCZOS)
        return result

    def texture_mesh(self, job: MeshJob, progress: Progress) -> bytes:
        import trimesh

        if not self.texture:
            raise RuntimeError("El texturizado está desactivado en el worker (inícialo con --texture)")
        progress(10, "Leyendo malla")
        mesh = trimesh.load(io.BytesIO(job.mesh_glb), file_type="glb", force="mesh")
        image = self._remove_background(job.images[0])
        progress(30, "Pintando textura (puede tardar)")
        return self._load_paint()(mesh, image=image).export(file_type="glb")


LIGHT_SHAPE = ("tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini")
LIGHT_MARKER = "modo_ligero.txt"  # written in the worker's folder; the app then starts it with the light model


def _is_oom(exc: Exception) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _fast_decoding(pipe) -> None:
    """FlashVDM (newer Hunyuan3D-2 code): the mesh decoding step goes from minutes to seconds."""
    if hasattr(pipe, "enable_flashvdm"):
        try:
            pipe.enable_flashvdm()
            log.info("FlashVDM activado")
        except Exception:  # noqa: BLE001 - the regular decoder still works
            log.exception("FlashVDM no disponible")


TRANSLATOR = "Helsinki-NLP/opus-mt-es-en"
SPANISH_WORDS = {"un", "una", "unos", "de", "del", "con", "el", "la", "los", "las", "y", "para", "que", "en", "sin",
                 "muy", "rojo", "roja", "azul", "verde", "negro", "negra", "blanco", "blanca", "pequeño", "grande"}


def looks_spanish(text: str) -> bool:
    words = [w.strip(".,;:!?¡¿()\"'").lower() for w in text.split()]
    return any(c in text for c in "áéíóúñ¿¡") or sum(w in SPANISH_WORDS for w in words) >= 1 and len(words) > 1


ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def esrgan_weights() -> str:
    import torch

    path = os.path.join(torch.hub.get_dir(), "forja3d", "RealESRGAN_x4plus.pth")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log.info("Descargando Real-ESRGAN …")
        torch.hub.download_url_to_file(ESRGAN_URL, path, progress=False)
    return path


def prefetch(shape_model: str, shape_subfolder: str, image_model: str, depth_model: str, multiview: bool,
             texture: bool, say: Callable[[int, str], None]) -> None:
    """Download every model the worker will use, once, so the first generation does not wait for it."""
    from huggingface_hub import snapshot_download

    # FlashVDM swaps in the turbo VAE of the same family
    vae_repo, vae = ((shape_model, "hunyuan3d-vae-v2-mini-turbo") if shape_model.endswith("mini")
                     else ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"))
    jobs = [(shape_model.split("/")[-1], lambda: snapshot_download(
        shape_model, allow_patterns=[f"{shape_subfolder}/*"])),
            ("FlashVDM", lambda: snapshot_download(vae_repo, allow_patterns=[f"{vae}/*"]))]
    if shape_model != LIGHT_SHAPE[0]:  # plan B when the GPU runs out of memory with the full model
        jobs.append((LIGHT_SHAPE[0].split("/")[-1] + " (plan B)", lambda: snapshot_download(
            LIGHT_SHAPE[0], allow_patterns=[f"{LIGHT_SHAPE[1]}/*", "hunyuan3d-vae-v2-mini-turbo/*"])))
    if multiview:
        jobs.append(("Hunyuan3D-2mv", lambda: snapshot_download(
            "tencent/Hunyuan3D-2mv", allow_patterns=["hunyuan3d-dit-v2-mv/*"])))
    # diffusers layout in fp16 only (the repos also hold full-precision single-file checkpoints)
    jobs.append((image_model.split("/")[-1], lambda: snapshot_download(
        image_model, allow_patterns=["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"])))
    jobs.append((depth_model.split("/")[-1], lambda: snapshot_download(depth_model)))
    if texture:
        jobs.append(("Hunyuan3D-Paint", lambda: snapshot_download(
            "tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-paint-v2-0/*", "hunyuan3d-delight-v2-0/*"])))

    def rembg():
        from hy3dgen.rembg import BackgroundRemover

        BackgroundRemover()

    jobs += [("U²-Net", rembg), ("Real-ESRGAN", esrgan_weights),
             ("Opus-MT es→en", lambda: snapshot_download(
                 TRANSLATOR, allow_patterns=["*.json", "*.spm", "*.txt", "*.bin", "*.safetensors"]))]
    for i, (label, fn) in enumerate(jobs):
        say(int(100 * i / len(jobs)), f"Descargando: {label}")
        fn()
    say(100, "Modelos listos")


def build_backend(name: str, **kwargs) -> Backend:
    if name == "dummy":
        return DummyBackend(**{k: v for k, v in kwargs.items() if k in ("delay", "texture", "multiview")})
    if name == "hunyuan":
        return HunyuanBackend(**{k: v for k, v in kwargs.items() if k != "delay"})
    raise ValueError(f"Backend desconocido: {name}")
