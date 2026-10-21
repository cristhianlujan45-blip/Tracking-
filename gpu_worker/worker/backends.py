"""Model backends for the local GPU worker.

* ``dummy``   – no GPU, no downloads: returns procedural meshes/images. Used to test the plumbing.
* ``hunyuan`` – Hunyuan3D-2mini (shape), optional Hunyuan3D-2mv (multi-view) and Hunyuan3D-Paint
                (texture), plus SD-Turbo / SDXL-Turbo for concept images and optional Zero123++ for
                the side and back views of a single picture. Tuned for 8 GB of VRAM:
                only one model lives on the GPU at a time, the rest wait in system RAM.

Heavy imports (torch, diffusers, hy3dgen) happen lazily inside the hunyuan backend so the worker
and its tests run on machines without them.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import sys
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
    # view of each image for multi-view shapes (default: VIEW_NAMES order)
    roles: list[str] | None = None


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

    def novel_views(self, image: Image.Image, quality: str = "standard", seed: int = 0) -> dict[str, Image.Image]:
        """AI-generated "left" / "back" / "right" views of a single front picture (transparent
        background); `view_poses` says where each one was really seen from."""
        ...


# Zero123++ v1.2 renders a 3×2 grid of 320 px views at azimuths 30, 90, 150, 210, 270, 330 from the
# input (reading order) and elevations alternating 20° / -10°. Checked on a one-eyed test character:
# its azimuth turns the camera towards the model's left like ours (cell 1 shows the face looking
# image-left, as the "left" view does), and no view needs mirroring. The grid has no straight back
# view, so "back" is the 30°-off one seen from lower down. role -> grid cell; poses in the app's
# convention: (azimuth from the front towards the model's left, elevation)
NOVEL_VIEW_MODEL = "sudo-ai/zero123plus-v1.2"
NOVEL_VIEW_PIPELINE = "sudo-ai/zero123plus-pipeline"  # its custom diffusers pipeline code
NOVEL_VIEW_CELLS = {"left": 1, "back": 3, "right": 4}
NOVEL_VIEW_POSES = {"left": (90.0, -10.0), "back": (210.0, -10.0), "right": (270.0, 20.0)}
NOVEL_VIEW_STEPS = {"draft": 28, "standard": 50, "high": 75}
NOVEL_VIEW_SIZE = 640


def png_bytes(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ================================================================ dummy


class DummyBackend:
    """Fake GPU: deterministic shapes and images, with a short delay per stage."""

    name = "dummy"
    view_poses = NOVEL_VIEW_POSES

    def __init__(self, delay: float = 0.2, texture: bool = True, multiview: bool = True, novel_views: bool = True,
                 chat: bool = False):
        self.delay = delay
        self.texture = texture
        self.multiview = multiview
        self.views = novel_views
        self.chat_enabled = chat
        self.last_job: MeshJob | None = None  # lets tests check what the app sent

    def capabilities(self) -> set[str]:
        caps = {"text", "image", "concepts", "depth", "upscale"}
        if self.multiview:
            caps.add("multi_image")
        if self.texture:
            caps.add("retexture")
        if self.views:
            caps.add("views")
        if self.chat_enabled:
            caps.add("chat")
        caps.add("cutout")
        return caps

    def novel_views(self, image: Image.Image, quality: str = "standard", seed: int = 0) -> dict[str, Image.Image]:
        """Mirrored, tinted copies of the picture: shapes and colours that tests can tell apart."""
        time.sleep(self.delay / 2)
        base = image.convert("RGBA").resize((512, 512))
        alpha = base.getchannel("A")
        tinted = Image.blend(base.convert("RGB"), Image.new("RGB", base.size, (40, 90, 220)), 0.5)
        tinted.putalpha(alpha)
        back = base.transpose(Image.FLIP_LEFT_RIGHT)
        return {"left": tinted, "back": back, "right": tinted.transpose(Image.FLIP_LEFT_RIGHT)}

    def info(self) -> dict:
        return {"gpu": "ninguna (dummy)", "vram_gb": 0, "models": ["dummy"]}

    def cutout(self, image: Image.Image) -> Image.Image:
        """Pixels close to the corner colour become transparent (enough for tests)."""
        rgba = np.array(image.convert("RGBA"))
        corner = rgba[0, 0, :3].astype(int)
        background = np.abs(rgba[:, :, :3].astype(int) - corner).sum(2) < 30
        if background.all():  # a flat picture: nothing to cut
            return image.convert("RGBA")
        rgba[background, 3] = 0
        return Image.fromarray(rgba, "RGBA")

    def chat(self, messages: list[dict], max_tokens: int = 400) -> str:
        """Echoes the request as a 'create' action, in the JSON shape the app's assistant expects."""
        import json

        text = messages[-1]["content"]
        return json.dumps({"action": "create", "prompt": text, "style": "auto", "reply": f"Creando {text}"})

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

        self.last_job = job
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
    "standard": (30, 384, 50_000),
    # FlashVDM decodes a 512 grid within 8 GB; out-of-memory falls back to the light model
    "high": (50, 512, 100_000),
}


class HunyuanBackend:
    """Hunyuan3D-2 family + SD-Turbo, scheduled so peak VRAM stays around 6–7 GB."""

    name = "hunyuan"
    view_poses = NOVEL_VIEW_POSES

    def __init__(
        self,
        shape_model: str = "tencent/Hunyuan3D-2mini",
        shape_subfolder: str = "hunyuan3d-dit-v2-mini",
        image_model: str = "stabilityai/sd-turbo",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        texture: bool = False,
        multiview: bool = False,
        novel_views: bool = False,
        device: str = "cuda",
        chat_model: str = "",
    ):
        import torch  # noqa: F401 - fail fast with a clear error if torch is missing

        self.shape_model, self.shape_subfolder = shape_model, shape_subfolder
        self.image_model = image_model
        self.depth_model = depth_model
        self._depth = None
        self._esrgan = None
        self.texture = texture
        self.multiview = multiview
        self.views = novel_views
        self.device = device
        self._shape = self._mv = self._paint = self._t2i = self._rembg = self._translator = self._mv_gen = None
        self.chat_model = chat_model
        self._llm = self._llm_tokenizer = None
        self._on_gpu: str | None = None
        # with 16 GB of RAM the parked models would not fit: drop them and reload from disk instead
        self.low_ram = 0 < total_ram_gb() < 24
        self.vram_gb = _vram_gb() if device != "cpu" else 0.0
        # 8 GB cards: big image models run with only the active part on the GPU (CPU offload)
        self.small_gpu = 0 < self.vram_gb < 10

    # ------------------------------------------------------------ meta

    def capabilities(self) -> set[str]:
        caps = {"text", "image", "concepts", "depth", "upscale"}
        if self.multiview:
            caps.add("multi_image")
        if self.texture:
            caps.add("retexture")
        if self.views:
            caps.add("views")
        if self.chat_model:
            caps.add("chat")
        caps.add("cutout")
        return caps

    def cutout(self, image: Image.Image) -> Image.Image:
        return self._remove_background(image)

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
        if self.views:
            models.append(NOVEL_VIEW_MODEL)
        if self.chat_model:
            models.append(self.chat_model)
        return {"gpu": gpu, "vram_gb": vram, "models": models, "low_ram": self.low_ram}

    # ------------------------------------------------------------ VRAM management

    def _activate(self, key: str) -> None:
        """Keep only one heavy pipeline on the GPU; park the previous one in system RAM."""
        import torch

        if self._on_gpu == key:
            return
        previous = getattr(self, f"_{self._on_gpu}", None) if self._on_gpu else None
        if previous is not None and hasattr(previous, "to"):
            # parked in RAM when there is room for it (no reload from disk next time); with little
            # free RAM it is dropped and reloaded from the disk cache when needed again
            if self.low_ram and ram_free_gb() - _size_gb(previous) < RAM_MARGIN_GB:
                setattr(self, f"_{self._on_gpu}", None)
                del previous
            else:
                previous.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        current = getattr(self, f"_{key}")
        if hasattr(current, "to"):
            current.to(self.device)
        self._on_gpu = key

    def _load_shape(self, progress: Progress | None = None):
        if self._shape is None:
            if self.low_ram:  # make room in RAM first: building it can take a few GB
                self._make_room(6)
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            log.info("Cargando %s …", self.shape_model)
            self._shape = self._load_hunyuan(Hunyuan3DDiTFlowMatchingPipeline, self.shape_model, self.shape_subfolder,
                                             progress, "Cargando modelo de forma")
            _fast_decoding(self._shape)
        self._activate("shape")
        return self._shape

    def _load_hunyuan(self, cls, repo: str, subfolder: str, progress: Progress | None = None, label: str = ""):
        """Built straight on the GPU a tensor at a time (little system RAM, see lowmem.py); Hunyuan's own
        loader (everything in RAM first) when that is not possible."""
        from .lowmem import load_pipeline

        if self.device != "cpu":
            self._activate("none")  # the GPU holds only this model while it is built
            try:
                folder = os.path.join(hy3d_models_dir(), repo, subfolder)
                if not os.path.isfile(os.path.join(folder, "model.fp16.safetensors")):
                    folder = hy3d_local(repo, subfolder)
                report = (lambda fraction, text: progress(15 + int(9 * fraction), f"{label}: {text}")) if progress else None
                pipe = load_pipeline(cls, folder, repo, subfolder, device=self.device, progress=report)
                self._on_gpu = None  # it is already on the GPU: _activate only records it
                return pipe
            except Exception as exc:  # noqa: BLE001 - any problem: the regular loader below
                log.warning("Carga con poca memoria no disponible (%s); se usa la de Hunyuan", exc)
                gc.collect()
                import torch

                torch.cuda.empty_cache()
        return cls.from_pretrained(repo, subfolder=subfolder, use_safetensors=True, device="cpu")

    def _load_mv(self, progress: Progress | None = None):
        if self._mv is None:
            if self.low_ram:  # make room in RAM first: building it can take a few GB
                self._make_room(6)
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            log.info("Cargando Hunyuan3D-2mv …")
            self._mv = self._load_hunyuan(Hunyuan3DDiTFlowMatchingPipeline, "tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv",
                                          progress, "Cargando modelo multi-vista")
            _fast_decoding(self._mv)
        self._activate("mv")
        return self._mv

    def _load_t2i(self):
        if self._t2i is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            log.info("Cargando %s …", self.image_model)
            if self.low_ram:
                self._make_room(8)
            self._t2i = AutoPipelineForText2Image.from_pretrained(
                self.image_model, torch_dtype=torch.float16, variant="fp16"
            )
            if self.small_gpu and hasattr(self._t2i, "enable_model_cpu_offload"):
                # SDXL-Turbo is ~7 GB in fp16: on an 8 GB card only the part working is on the GPU
                self._activate("none")
                self._t2i.enable_model_cpu_offload()
                self._t2i_offloaded = True
        if getattr(self, "_t2i_offloaded", False):
            self._activate("none")  # offloaded pipelines place themselves: just make room
        else:
            self._activate("t2i")
        return self._t2i

    def _load_mv_gen(self):
        """Zero123++ (~3 GB in fp16, ~5 GB of VRAM while running), from local Hub snapshots so it
        also loads offline: its pipeline code lives in a separate repo."""
        if self._mv_gen is None:
            import torch
            from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

            if self.low_ram:
                self._make_room(4, keep="rembg")
            log.info("Cargando %s …", NOVEL_VIEW_MODEL)
            pipe = DiffusionPipeline.from_pretrained(
                hub_dir(NOVEL_VIEW_MODEL), custom_pipeline=hub_dir(NOVEL_VIEW_PIPELINE),
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            # sampler recommended for v1.2
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            self._mv_gen = pipe
        self._activate("mv_gen")
        return self._mv_gen

    def _load_paint(self):
        if self._paint is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            log.info("Cargando Hunyuan3D-Paint …")
            self._activate("none")  # its models go straight to the GPU while loading: make room first
            self._paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder=PAINT_SUBFOLDER)
            # The paint pipeline needs ~16 GB if fully resident; offload streams it through 8 GB.
            if hasattr(self._paint, "enable_model_cpu_offload"):
                self._paint.enable_model_cpu_offload()
        # offloaded pipelines manage their own placement: just free whatever else is on the GPU
        self._activate("none")
        return self._paint

    _none = None  # sentinel for _activate("none")

    def _load_llm(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Cargando %s …", self.chat_model)
            self._llm_tokenizer = AutoTokenizer.from_pretrained(self.chat_model)
            if self.device != "cpu":
                self._activate("none")  # straight to the GPU: no copy of the weights in system RAM
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.chat_model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                **({"device_map": {"": self.device}} if self.device != "cpu" else {}))
        self._activate("llm")
        return self._llm, self._llm_tokenizer

    def chat(self, messages: list[dict], max_tokens: int = 400) -> str:
        """Small local text model for the app's assistant (understands requests, no key needed)."""
        import torch

        model, tok = self._load_llm()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
        img = img.convert("RGBA")
        if np.asarray(img)[:, :, 3].min() < 250:  # already a cut-out: keep it
            return clean_cutout(img)
        from rembg import remove

        if self._rembg is None:
            self._rembg = background_session()
        try:
            cut = remove(img.convert("RGB"), session=self._rembg, post_process_mask=True)
        except Exception as exc:  # noqa: BLE001 - only memory problems have a plan B
            if not _is_oom(exc):
                raise
            log.warning("BiRefNet sin memoria (%s); se usa U²-Net", exc)
            from rembg import new_session

            self._rembg = None
            gc.collect()
            self._rembg = new_session("u2net")  # ~170 MB instead of ~1 GB
            cut = remove(img.convert("RGB"), session=self._rembg, post_process_mask=True)
        return clean_cutout(cut.convert("RGBA"))

    # ------------------------------------------------------------ tasks

    def generate_images(self, prompt: str, count: int, seed: int, negative: str = "") -> list[Image.Image]:
        import torch

        pipe = self._load_t2i()
        images = []
        for i in range(count):  # one at a time: SDXL-Turbo already fills most of 8 GB of VRAM
            generator = torch.Generator(device="cpu").manual_seed(seed + i)  # CPU: reproducible across GPUs
            images += pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
                           width=512, height=512, generator=generator).images
        return images

    def novel_views(self, image: Image.Image, quality: str = "standard", seed: int = 0) -> dict[str, Image.Image]:
        import torch

        cond = _frame(self._remove_background(image), 320)
        pipe = self._load_mv_gen()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        grid = pipe(cond, num_inference_steps=NOVEL_VIEW_STEPS.get(quality, 50), generator=generator).images[0]
        tile = grid.width // 2
        views = {}
        for role, cell in NOVEL_VIEW_CELLS.items():
            row, col = divmod(cell, 2)
            view = self._remove_background(grid.crop((col * tile, row * tile, (col + 1) * tile, (row + 1) * tile)))
            try:  # Real-ESRGAN: 320 px views are too soft to paint a model with
                rgb = self.upscale(view.convert("RGB"), 4).resize((NOVEL_VIEW_SIZE,) * 2, Image.LANCZOS)
            except Exception:  # noqa: BLE001 - a plain resize still works
                log.exception("No se pudo ampliar la vista %s", role)
                rgb = view.convert("RGB").resize((NOVEL_VIEW_SIZE,) * 2, Image.LANCZOS)
            rgb.putalpha(view.getchannel("A").resize(rgb.size, Image.LANCZOS))
            views[role] = rgb
        return views

    def image_to_mesh(self, job: MeshJob, progress: Progress) -> bytes:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        steps, octree, max_faces = HUNYUAN_QUALITY.get(job.quality, HUNYUAN_QUALITY["standard"])
        progress(5, "Quitando fondo")
        images = [self._remove_background(img) for img in job.images]

        progress(25, "Generando forma 3D")
        mesh = None
        for attempt in ("full", "freed", "light"):
            out_of_memory = False
            try:
                if attempt == "light":
                    mesh = self._shape_pass(images[:1], None, job.seed, steps, min(octree, 320), progress,
                                            chunks=4000)
                else:
                    mesh = self._shape_pass(images, job.roles, job.seed, steps, octree, progress)
                break
            except Exception as exc:  # noqa: BLE001 - only out-of-memory is retried
                if not _is_oom(exc) or attempt == "light":
                    raise
                log.warning("Sin memoria (%s) en el intento %s", exc, attempt)
                out_of_memory = True  # handled below: inside `except` the failed frame still holds memory
            if out_of_memory and attempt == "full":
                # plan A': everything else out of RAM and VRAM, same model again
                progress(28, "Poca memoria: liberando memoria y reintentando")
                self._free_memory()
            elif out_of_memory:
                # plan B for small PCs: the light model (smaller, coarser grid), from now on
                progress(30, "Poca memoria: reintentando en modo ligero")
                self._use_light_model()

        progress(80, "Limpiando malla")
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=max_faces)
        mesh = tidy_mesh(mesh)

        if self.texture and job.quality != "draft":
            progress(85, "Texturizando (puede tardar)")
            mesh = self._load_paint()(mesh, image=images[0])
        return mesh.export(file_type="glb")

    def _shape_pass(self, images, roles, seed: int, steps: int, octree: int, progress: Progress,
                    chunks: int = 8000):
        import torch

        generator = torch.Generator(device="cpu").manual_seed(seed)
        if len(images) > 1 and self.multiview:
            progress(15, "Cargando modelo multi-vista")
            pipe = self._load_mv(progress)
            image_arg = dict(zip(roles or VIEW_NAMES, images))
        else:
            progress(15, "Cargando modelo de forma")
            pipe = self._load_shape(progress)
            image_arg = images[0]
        return pipe(
            image=image_arg, num_inference_steps=steps, octree_resolution=octree,
            num_chunks=chunks, generator=generator, output_type="trimesh",
        )[0]

    def _make_room(self, need_gb: float, keep: str | None = None) -> None:
        """Drop parked models (biggest first) until `need_gb` of RAM is free, keeping the rest loaded."""
        parked = [k for k in MODEL_KEYS if k not in (keep, self._on_gpu) and getattr(self, f"_{k}", None) is not None]
        parked.sort(key=lambda k: -_size_gb(getattr(self, f"_{k}")))
        for key in parked:
            if ram_free_gb() >= need_gb + RAM_MARGIN_GB:
                break
            log.info("Liberando %s de la RAM", key)
            setattr(self, f"_{key}", None)
            gc.collect()

    def warm_up(self) -> None:
        """Read the model files once in the background so the first loads come from Windows' file cache
        (memory the system takes back whenever it needs it), not from the disk."""
        roots = [os.path.join(hy3d_models_dir(), self.shape_model, self.shape_subfolder)]
        if self.multiview:
            roots.append(os.path.join(hy3d_models_dir(), "tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv"))
        roots.append(os.path.join(os.environ.get("U2NET_HOME", os.path.expanduser("~/.u2net"))))
        files = [os.path.join(r, n) for r in roots if os.path.isdir(r) for n in os.listdir(r)
                 if n.endswith((".safetensors", ".onnx"))]
        budget = (ram_free_gb() - 4) * 1024**3
        started = time.time()
        for path in files:
            try:
                size = os.path.getsize(path)
                if size > budget:
                    break
                budget -= size
                with open(path, "rb") as fh:
                    while fh.read(16 * 1024 * 1024):
                        pass
            except OSError:
                continue
        log.info("Archivos de modelos precargados en %.0f s", time.time() - started)

    def _free_memory(self, keep: str | None = None) -> None:
        """Drop every loaded model except `keep` (they are reloaded from disk when needed): before a big
        load on a PC with little RAM, and after running out of memory."""
        import torch

        for key in MODEL_KEYS:
            if key == keep or getattr(self, f"_{key}", None) is None:
                continue
            pipe = getattr(self, f"_{key}")
            if key == self._on_gpu and hasattr(pipe, "to"):
                try:
                    pipe.to("cpu")
                except Exception:  # noqa: BLE001 - dropping the reference is what matters
                    pass
            setattr(self, f"_{key}", None)
            if key == self._on_gpu:
                self._on_gpu = None
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _use_light_model(self) -> None:
        """Switch to Hunyuan3D-2mini for the rest of the session and remember it for the next start."""
        import torch

        for key in ("shape", "mv", "t2i", "paint", "mv_gen", "llm", "depth"):
            pipe = getattr(self, f"_{key}", None)
            if pipe is not None and hasattr(pipe, "to"):
                try:
                    pipe.to("cpu")
                except Exception:  # noqa: BLE001 - dropping the reference is what matters
                    pass
            setattr(self, f"_{key}", None)
        self._on_gpu = None
        gc.collect()
        torch.cuda.empty_cache()
        if self.shape_model != LIGHT_SHAPE[0]:
            self.shape_model, self.shape_subfolder = LIGHT_SHAPE
            hy3d_local(*LIGHT_SHAPE)  # ~4 GB, only the first time
            hy3d_local(LIGHT_SHAPE[0], "hunyuan3d-vae-v2-mini-turbo")
            self.multiview = False  # the multi-view model is as big as the full one
            try:
                with open(LIGHT_MARKER, "w", encoding="utf-8") as fh:
                    fh.write("Hunyuan3D-2mini: la GPU se quedó sin memoria con el modelo completo\n")
            except OSError:
                pass

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        """Depth Anything V2 on the CPU (a second or two): the GPU stays free for the big models."""
        import torch

        if self._depth is None:
            from transformers import pipeline

            log.info("Cargando %s …", self.depth_model)
            self._depth = pipeline("depth-estimation", model=self.depth_model, device=-1)
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


MODEL_KEYS = ("shape", "mv", "t2i", "paint", "mv_gen", "llm", "depth", "translator", "rembg", "esrgan")
RAM_MARGIN_GB = 3.0  # kept free for Windows, the app and the work itself


def _size_gb(obj) -> float:
    """Memory held by a model or pipeline (its torch modules); 0 for anything else."""
    try:
        import torch
    except ImportError:
        return 0.0
    if isinstance(obj, torch.nn.Module):
        modules = [obj]
    else:
        parts = getattr(obj, "components", None)
        values = parts.values() if isinstance(parts, dict) else vars(obj).values() if hasattr(obj, "__dict__") else []
        modules = [m for m in values if isinstance(m, torch.nn.Module)]
    seen, total = set(), 0
    for module in modules:
        for tensor in [*module.parameters(), *module.buffers()]:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
    return total / 1024**3


def ram_free_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "avail", "page_total", "page_avail",
                                                           "virtual_total", "virtual_avail", "extended")]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            # what can still be committed, not just idle RAM: os error 1455 is about the commit limit
            return min(status.avail, status.page_avail) / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0


def _vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:  # noqa: BLE001 - unknown means "no special handling"
        pass
    return 0.0


def total_ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "avail", "page_total", "page_avail",
                                                           "virtual_total", "virtual_avail", "extended")]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0


PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"  # the one Hunyuan3DPaintPipeline loads by default


def hy3d_models_dir() -> str:
    return os.path.expanduser(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen"))


def hy3d_local(repo: str, subfolder: str) -> str:
    """Download only what Hunyuan's loader reads (config + fp16 safetensors, ~5 GB instead of ~25 GB for
    the five copies in the folder) into the local folder it checks before going to the Hub."""
    from huggingface_hub import hf_hub_download

    target = os.path.join(hy3d_models_dir(), repo)
    for name in ("config.yaml", "model.fp16.safetensors"):
        hf_hub_download(repo, f"{subfolder}/{name}", local_dir=target)
    return os.path.join(target, subfolder)


BACKGROUND_MODEL = "birefnet-general"  # sharper edges than U²-Net (hair, fingers, holes); runs on the CPU


def background_session():
    """BiRefNet through rembg, falling back to U²-Net when it is not available."""
    from rembg import new_session

    try:
        return new_session(BACKGROUND_MODEL)
    except Exception:  # noqa: BLE001 - older rembg or download problem
        log.exception("BiRefNet no disponible; se usa U²-Net")
        return new_session("u2net")  # explicit: newer rembg defaults to a non-commercial model


def clean_cutout(img: Image.Image) -> Image.Image:
    """Keep the subject only: drop specks and shadows detached from it and fill pinholes in the mask,
    so the 3D model gets no floating bits or holes from a noisy background removal."""
    try:
        import cv2
    except ImportError:
        return img
    rgba = np.array(img.convert("RGBA"))
    mask = (rgba[:, :, 3] > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return img
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, bool)
    keep[1:] = areas >= 0.03 * areas.max()  # the subject and big parts of it (e.g. a separate weapon)
    subject = keep[labels]
    # holes completely enclosed by the subject smaller than 0.2 % of the picture are mask errors
    holes = (~subject).astype(np.uint8)
    hcount, hlabels, hstats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    border = set(np.unique(np.concatenate([hlabels[0], hlabels[-1], hlabels[:, 0], hlabels[:, -1]])))
    small = [i for i in range(1, hcount) if i not in border and hstats[i, cv2.CC_STAT_AREA] < 0.002 * mask.size]
    fill = np.isin(hlabels, small)
    alpha = rgba[:, :, 3].astype(np.float32)
    alpha[~(subject | fill)] = 0
    alpha[fill] = 255
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def tidy_mesh(mesh):
    """Consistent outward normals and small holes closed (cleaner shading, prints and rigging)."""
    import trimesh

    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception:  # noqa: BLE001 - cosmetic: the mesh is usable as it is
        log.exception("No se pudo ordenar la malla")
    return mesh


def _is_oom(exc: Exception) -> bool:
    """Out of GPU memory, or of system memory: Windows reports a full commit limit (RAM + page file)
    as os error 1455 "the paging file is too small" when a model is loaded."""
    text = str(exc).lower()
    return (type(exc).__name__ in ("OutOfMemoryError", "MemoryError") or getattr(exc, "winerror", None) == 1455
            or any(k in text for k in ("out of memory", "paging file", "archivo de paginación", "1455",
                                       "not enough memory", "can't allocate", "unable to allocate")))


def _frame(img: Image.Image, size: int, fill: float = 0.85) -> Image.Image:
    """Centre a cut-out on a square transparent canvas, the object filling `fill` of it."""
    img = img.convert("RGBA")
    box = img.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox() or (0, 0, *img.size)
    obj = img.crop(box)
    side = int(max(obj.size) / fill)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def hub_dir(repo: str) -> str:
    """Local snapshot of a Hub repo: the cached one when present (no network needed), else download it."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(repo)


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
             texture: bool, say: Callable[[int, str], None], novel_views: bool = False,
             chat_model: str = "") -> None:
    """Download every model the worker will use, once, so the first generation does not wait for it."""
    from huggingface_hub import snapshot_download

    # FlashVDM swaps in the turbo VAE of the same family
    vae_repo, vae = ((shape_model, "hunyuan3d-vae-v2-mini-turbo") if shape_model.endswith("mini")
                     else ("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo"))
    jobs = [(shape_model.split("/")[-1], lambda: hy3d_local(shape_model, shape_subfolder)),
            ("FlashVDM", lambda: hy3d_local(vae_repo, vae))]
    if multiview:
        jobs.append(("Hunyuan3D-2mv", lambda: hy3d_local("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv")))
    # diffusers layout in fp16 only (the repos also hold full-precision single-file checkpoints)
    jobs.append((image_model.split("/")[-1], lambda: snapshot_download(
        image_model, allow_patterns=["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"])))
    jobs.append((depth_model.split("/")[-1], lambda: snapshot_download(depth_model)))
    if texture:  # the folders Hunyuan3DPaintPipeline loads, where it looks first
        jobs.append(("Hunyuan3D-Paint", lambda: snapshot_download(
            "tencent/Hunyuan3D-2", local_dir=os.path.join(hy3d_models_dir(), "tencent/Hunyuan3D-2"),
            allow_patterns=[f"{PAINT_SUBFOLDER}/*", "hunyuan3d-delight-v2-0/*"])))
    if novel_views:  # weights plus the custom pipeline code, so loading later needs no network
        jobs.append(("Zero123++", lambda: [snapshot_download(r) for r in (NOVEL_VIEW_PIPELINE, NOVEL_VIEW_MODEL)]))

    def rembg():
        background_session()

    jobs += [("BiRefNet", rembg), ("Real-ESRGAN", esrgan_weights),
             ("Opus-MT es-en", lambda: snapshot_download(
                 TRANSLATOR, allow_patterns=["*.json", "*.spm", "*.txt", "*.bin", "*.safetensors"]))]
    if chat_model:
        jobs.append((chat_model.split("/")[-1], lambda: snapshot_download(
            chat_model, allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*"])))
    for i, (label, fn) in enumerate(jobs):
        say(int(100 * i / len(jobs)), f"Descargando: {label}")
        fn()
    say(100, "Modelos listos")


def build_backend(name: str, **kwargs) -> Backend:
    if name == "dummy":
        return DummyBackend(**{k: v for k, v in kwargs.items()
                               if k in ("delay", "texture", "multiview", "novel_views", "chat")})
    if name == "hunyuan":
        return HunyuanBackend(**{k: v for k, v in kwargs.items() if k != "delay"})
    raise ValueError(f"Backend desconocido: {name}")
