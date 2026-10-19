"""Offline provider: builds procedural, vertex-colored meshes so the whole app works without a GPU or API key.

The shape is picked from keywords in the prompt (or from a hash of the input) and the seed varies
the details, so the same input + seed always gives the same model.
"""

from __future__ import annotations

import asyncio
import hashlib
import io

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from .. import geometry
from ..creations import pseudo_depth
from .base import GenerationRequest, GenerationResult, InputImage, ProgressCallback

KEYWORDS = {
    "quadruped": ("perro", "dog", "gato", "cat", "caballo", "horse", "lobo", "wolf", "zorro", "fox", "vaca", "cow",
                  "tigre", "tiger", "león", "leon", "lion", "cachorro", "puppy", "oveja", "sheep", "ciervo", "deer",
                  "kitsune", "shiba"),
    "humanoid": ("personaje", "character", "humano", "human", "persona", "robot", "guerrero", "warrior", "soldado",
                 "chica", "chico", "hombre", "mujer", "caballero", "knight", "mago", "wizard", "astronauta",
                 "astronaut", "héroe", "heroe", "hero", "ninja", "zombi", "zombie", "elfo", "elf",
                 "guerrera", "samurái", "samurai", "heroína", "heroina", "maga", "princesa", "princess"),
    "tree": ("tree", "árbol", "arbol", "pino", "pine"),
    "mushroom": ("mushroom", "hongo", "seta", "champiñón", "champinon"),
    "snowman": ("snowman", "muñeco de nieve", "muneco de nieve"),
    "rock": ("rock", "roca", "piedra", "stone", "asteroid", "asteroide"),
    "donut": ("donut", "dona", "rosquilla", "ring", "anillo"),
    "cube": ("cube", "cubo", "box", "caja"),
}
COLORS = {
    (0.85, 0.15, 0.12): ("rojo", "roja", "red"),
    (0.15, 0.35, 0.9): ("azul", "blue"),
    (0.2, 0.7, 0.25): ("verde", "green"),
    (0.95, 0.8, 0.15): ("amarillo", "amarilla", "yellow"),
    (0.95, 0.5, 0.1): ("naranja", "orange"),
    (0.55, 0.25, 0.8): ("morado", "morada", "violeta", "purple"),
    (0.95, 0.5, 0.7): ("rosa", "rosado", "rosada", "pink"),
    (0.9, 0.72, 0.3): ("dorado", "dorada", "oro", "gold"),
    (0.92, 0.92, 0.92): ("blanco", "blanca", "white"),
    (0.12, 0.12, 0.14): ("negro", "negra", "black"),
}
FALLBACKS = ["rock", "donut", "mushroom", "tree", "snowman", "cube"]
STAGES = ["Analizando entrada", "Generando geometría", "Refinando malla", "Aplicando texturas", "Exportando GLB"]


def _colorize(mesh: trimesh.Trimesh, base: np.ndarray, rng: np.random.Generator) -> trimesh.Trimesh:
    """Vertical gradient + a bit of noise so the model looks shaded even under flat lighting."""
    z = mesh.vertices[:, 1]
    t = (z - z.min()) / max(np.ptp(z), 1e-6)
    shade = 0.65 + 0.35 * t[:, None] + rng.normal(0, 0.03, (len(z), 1))
    rgb = np.clip(base[None, :] * shade, 0, 1)
    rgba = np.hstack([rgb, np.ones((len(z), 1))]) * 255
    mesh.visual.vertex_colors = rgba.astype(np.uint8)
    return mesh


def _part(mesh: trimesh.Trimesh, color, rng, translate=(0, 0, 0)) -> trimesh.Trimesh:
    mesh.apply_translation(translate)
    return _colorize(mesh, np.array(color, dtype=float), rng)


def _y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """trimesh primitives are Z-aligned; glTF is Y-up."""
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return mesh


def _displace(mesh: trimesh.Trimesh, rng: np.random.Generator, amount: float) -> trimesh.Trimesh:
    # sum of random low-frequency waves: smooth lumps instead of spikes
    noise = np.zeros(len(mesh.vertices))
    for _ in range(6):
        direction = rng.normal(0, 1, 3)
        noise += np.sin(mesh.vertices @ direction * rng.uniform(2, 5) + rng.uniform(0, np.pi * 2))
    mesh.vertices = mesh.vertices + mesh.vertex_normals * (noise / 6)[:, None] * amount * 4
    return mesh


def _limb(p0, p1, radius: float) -> trimesh.Trimesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    capsule = trimesh.creation.capsule(height=float(np.linalg.norm(axis)), radius=radius, count=[16, 16])
    capsule.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis / np.linalg.norm(axis)))
    capsule.apply_translation((p0 + p1) / 2)
    return capsule


def build_humanoid(rng: np.random.Generator, hue: np.ndarray, arm_angle: float = 45.0) -> trimesh.Trimesh:
    """A-pose character facing +Z, ~1.8 units tall, fused into one surface like AI-generated meshes."""
    skin, pants, shoes = (0.86, 0.68, 0.55), (0.22, 0.24, 0.32), (0.15, 0.12, 0.1)
    a = np.radians(arm_angle)
    parts = []
    for side in (-1, 1):
        hip, knee, ankle = (0.1 * side, 0.92, 0), (0.11 * side, 0.5, 0.01), (0.11 * side, 0.1, 0)
        parts += [_part(_limb(hip, knee, 0.075), pants, rng), _part(_limb(knee, ankle, 0.058), pants, rng)]
        foot = trimesh.creation.box((0.1, 0.07, 0.24))
        parts.append(_part(foot, shoes, rng, (0.11 * side, 0.045, 0.06)))
        shoulder = np.array([0.19 * side, 1.38, 0])
        direction = np.array([np.sin(a) * side, -np.cos(a), 0])
        elbow = shoulder + 0.29 * direction
        wrist = elbow + 0.26 * direction
        parts += [_part(_limb(shoulder, elbow, 0.052), hue, rng), _part(_limb(elbow, wrist, 0.043), hue, rng),
                  _part(_limb(wrist, wrist + 0.1 * direction, 0.04), skin, rng)]
    torso = _limb((0, 0.95, 0), (0, 1.33, 0), 0.16)
    torso.apply_scale((1.12, 1, 0.72))
    torso.apply_translation((0, 0.05 * 0, 0))
    parts += [_part(torso, hue, rng), _part(_limb((0, 1.38, 0), (0, 1.52, 0), 0.055), skin, rng),
              _part(trimesh.creation.icosphere(3, radius=0.115), skin, rng, (0, 1.63, 0.01))]
    raw = trimesh.util.concatenate(parts)
    # fuse the primitives into a single closed surface, like a real generated character
    from .. import geometry

    fused = geometry.voxelize(raw, 80)
    fused.merge_vertices()
    trimesh.smoothing.filter_taubin(fused, iterations=12)
    fused.visual = trimesh.visual.ColorVisuals(fused, vertex_colors=geometry.transfer_colors(raw, fused))
    return fused


def build_quadruped(rng: np.random.Generator, hue: np.ndarray) -> trimesh.Trimesh:
    """Dog-like animal facing +Z (~1.1 long, ~0.75 tall), fused into one surface."""
    belly = tuple(np.clip(hue * 1.25, 0, 1))
    parts = []
    body = _limb((0, 0.5, -0.32), (0, 0.52, 0.3), 0.15)
    body.apply_scale((0.9, 1.0, 1.0))
    parts.append(_part(body, hue, rng))
    for side in (-1, 1):
        for z, shoulder_y in ((0.27, 0.48), (-0.3, 0.48)):
            knee = (0.1 * side, 0.27, z - 0.02)
            parts += [_part(_limb((0.1 * side, shoulder_y, z), knee, 0.055), hue, rng),
                      _part(_limb(knee, (0.1 * side, 0.05, z + 0.02), 0.042), belly, rng),
                      _part(_limb((0.1 * side, 0.035, z + 0.02), (0.1 * side, 0.035, z + 0.09), 0.04), belly, rng)]
    parts += [_part(_limb((0, 0.56, 0.36), (0, 0.72, 0.46), 0.075), hue, rng),
              _part(trimesh.creation.icosphere(3, radius=0.11), hue, rng, (0, 0.76, 0.5)),
              _part(_limb((0, 0.73, 0.55), (0, 0.7, 0.66), 0.05), belly, rng),
              _part(_limb((0, 0.56, -0.44), (0, 0.72, -0.62), 0.035), hue, rng)]
    for side in (-1, 1):  # ears
        parts.append(_part(trimesh.creation.cone(0.04, 0.1, sections=12), hue * 0.7, rng, (0.06 * side, 0.83, 0.49)))
    raw = trimesh.util.concatenate(parts)
    from .. import geometry

    fused = geometry.voxelize(raw, 80)
    fused.merge_vertices()
    trimesh.smoothing.filter_taubin(fused, iterations=12)
    fused.visual = trimesh.visual.ColorVisuals(fused, vertex_colors=geometry.transfer_colors(raw, fused))
    return fused


def build_shape(kind: str, rng: np.random.Generator, hue: np.ndarray) -> trimesh.Trimesh:
    if kind == "humanoid":
        return build_humanoid(rng, hue)
    if kind == "quadruped":
        return build_quadruped(rng, hue)
    if kind == "tree":
        parts = [_part(_y_up(trimesh.creation.cylinder(0.12, 0.6, sections=24)), (0.45, 0.28, 0.15), rng, (0, 0.3, 0))]
        for i, (r, h, y) in enumerate([(0.55, 0.7, 0.85), (0.42, 0.6, 1.2), (0.28, 0.5, 1.5)]):
            cone = _y_up(trimesh.creation.cone(r, h, sections=32))
            parts.append(_part(cone, (0.15, 0.55 + 0.08 * i, 0.22), rng, (0, y - h / 2, 0)))
    elif kind == "mushroom":
        stem = _part(_y_up(trimesh.creation.cylinder(0.14, 0.7, sections=32)), (0.93, 0.89, 0.8), rng, (0, 0.35, 0))
        cap = trimesh.creation.icosphere(4, radius=0.5)
        cap.vertices[:, 1] = np.maximum(cap.vertices[:, 1], -0.05) * 0.8
        parts = [stem, _part(cap, hue, rng, (0, 0.72, 0))]
    elif kind == "snowman":
        parts = [
            _part(trimesh.creation.icosphere(3, radius=r), (0.95, 0.96, 1.0), rng, (0, y, 0))
            for r, y in [(0.5, 0.5), (0.36, 1.25), (0.25, 1.78)]
        ]
        nose = trimesh.creation.cone(0.05, 0.25, sections=16)
        parts.append(_part(nose, (1.0, 0.5, 0.1), rng, (0, 1.78, 0.22)))
    elif kind == "donut":
        torus = trimesh.creation.torus(0.55, 0.25, major_sections=64, minor_sections=32)
        parts = [_part(torus, hue, rng, (0, 0.25, 0))]
    elif kind == "cube":
        box = trimesh.creation.box((0.9, 0.9, 0.9)).subdivide().subdivide()
        parts = [_part(box, hue, rng, (0, 0.45, 0))]
    else:  # rock
        rock = _displace(trimesh.creation.icosphere(4, radius=0.6), rng, 0.06)
        rock.apply_scale((1.0, 0.75, 0.9))
        parts = [_part(rock, hue * 0.6 + 0.2, rng, (0, 0.45, 0))]
    return trimesh.util.concatenate(parts)


def pick_color(prompt: str | None) -> np.ndarray | None:
    words = set((prompt or "").lower().replace(",", " ").split())
    for rgb, names in COLORS.items():
        if words & set(names):
            return np.array(rgb)
    return None


def pick_kind(prompt: str | None, seed: int) -> str:
    text = (prompt or "").lower()
    for kind, words in KEYWORDS.items():
        if any(w in text for w in words):
            return kind
    return FALLBACKS[seed % len(FALLBACKS)]


def _seed_hue(req: GenerationRequest, salt: bytes = b"") -> tuple[int, np.ndarray]:
    images = b"".join(img.data for img in req.images)
    digest = hashlib.sha256((req.prompt or "").encode() + images + req.seed.to_bytes(8, "little") + salt).digest()
    seed = int.from_bytes(digest[:8], "little")
    hue = pick_color(req.prompt)
    if hue is None:
        hue = np.array([digest[8], digest[9], digest[10]], dtype=float) / 255 * 0.7 + 0.25
    return seed, hue


def draw_concept(kind: str, hue: np.ndarray, rng: np.random.Generator, size: int = 512) -> bytes:
    """A flat illustration of the shape, standing in for a text -> image model."""
    img = Image.new("RGB", (size, size), (246, 246, 248))
    d = ImageDraw.Draw(img)
    c = tuple(int(x * 255) for x in hue)
    dark = tuple(int(x * 255 * 0.6) for x in hue)
    s, j = size, lambda: int(rng.integers(-18, 18))
    d.ellipse((s * 0.2, s * 0.84, s * 0.8, s * 0.92), fill=(225, 225, 230))  # shadow
    if kind == "humanoid":
        skin = (219, 173, 140)
        d.ellipse((s * 0.44, s * 0.08, s * 0.56, s * 0.22), fill=skin)
        d.rounded_rectangle((s * 0.4, s * 0.24, s * 0.6, s * 0.5), radius=18, fill=c)
        for side in (-1, 1):
            d.line([(s * (0.5 + 0.09 * side), s * 0.27), (s * (0.5 + 0.22 * side), s * 0.48)], fill=c, width=22)
            d.line([(s * (0.5 + 0.05 * side), s * 0.5), (s * (0.5 + 0.06 * side), s * 0.84)], fill=(56, 60, 80), width=28)
    elif kind == "tree":
        d.rectangle((s * 0.45, s * 0.65, s * 0.55, s * 0.87), fill=(115, 72, 40))
        for k, y in enumerate((0.62, 0.47, 0.32)):
            w = 0.3 - k * 0.06
            d.polygon([(s * (0.5 - w), s * y + 60), (s * (0.5 + w), s * y + 60), (s * 0.5 + j(), s * (y - 0.2))],
                      fill=(40, 140 + 20 * k, 60))
    elif kind == "mushroom":
        d.rectangle((s * 0.42, s * 0.5, s * 0.58, s * 0.87), fill=(238, 228, 205))
        d.pieslice((s * 0.15 + j(), s * 0.18, s * 0.85, s * 0.78), 180, 360, fill=c)
        for _ in range(5):
            x, y = rng.uniform(0.28, 0.7) * s, rng.uniform(0.28, 0.44) * s
            d.ellipse((x, y, x + 30, y + 24), fill=(250, 250, 250))
    elif kind == "snowman":
        for r, y in ((0.2, 0.68), (0.14, 0.42), (0.1, 0.22)):
            d.ellipse((s * (0.5 - r), s * (y - r), s * (0.5 + r), s * (y + r)), fill=(255, 255, 255), outline=(200, 205, 215), width=3)
        d.polygon([(s * 0.5, s * 0.22), (s * 0.5, s * 0.25), (s * 0.62, s * 0.235)], fill=(255, 130, 30))
        d.rounded_rectangle((s * 0.36, s * 0.3, s * 0.64, s * 0.34), radius=8, fill=c)  # scarf
        d.rectangle((s * 0.56 + j() / 2, s * 0.32, s * 0.61, s * 0.44), fill=c)
    elif kind == "donut":
        d.ellipse((s * 0.2, s * 0.3, s * 0.8, s * 0.8), fill=c)
        d.ellipse((s * 0.42, s * 0.47, s * 0.58, s * 0.63), fill=(246, 246, 248))
    elif kind == "cube":
        a, b, h = s * 0.5, s * 0.3, s * 0.25
        d.polygon([(a, b), (a + h, b + h * 0.5), (a, b + h), (a - h, b + h * 0.5)], fill=c)
        d.polygon([(a - h, b + h * 0.5), (a, b + h), (a, b + h * 2.1), (a - h, b + h * 1.6)], fill=dark)
        d.polygon([(a + h, b + h * 0.5), (a, b + h), (a, b + h * 2.1), (a + h, b + h * 1.6)],
                  fill=tuple(int(v * 0.8) for v in c))
    else:  # rock
        angles = np.sort(rng.uniform(0, 2 * np.pi, 11))
        radius = rng.uniform(0.22, 0.32, 11) * s
        d.polygon([(s * 0.5 + r * np.cos(t), s * 0.58 + r * 0.7 * np.sin(t)) for t, r in zip(angles, radius)], fill=c)
    img = img.filter(ImageFilter.SMOOTH)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class MockProvider:
    name = "mock"
    engines = [{"id": "demo", "label": "Demo (sin IA)", "description": "Formas procedurales para probar la app.",
                "native_text": True, "max_views": 4, "tags": []}]
    capabilities = frozenset({"text", "image", "multi_image", "concepts", "retexture", "depth"})

    def __init__(self, step_delay: float = 0.6):
        self.step_delay = step_delay

    async def _stages(self, on_progress: ProgressCallback, stages: list[str]) -> None:
        for i, stage in enumerate(stages):
            await on_progress(int(i / len(stages) * 100), stage)
            await asyncio.sleep(self.step_delay)

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        seed, hue = _seed_hue(req)
        await self._stages(on_progress, STAGES)
        mesh = build_shape(pick_kind(req.prompt, seed), np.random.default_rng(seed), hue)
        glb = geometry.mesh_to_glb(mesh)
        return GenerationResult(model_glb=glb)

    async def generate_images(self, req: GenerationRequest, count: int) -> list[bytes]:
        images = []
        for i in range(count):
            seed, hue = _seed_hue(req, salt=bytes([i]))
            images.append(draw_concept(pick_kind(req.prompt, seed), hue, np.random.default_rng(seed)))
        await asyncio.sleep(self.step_delay)
        return images

    async def retexture(self, glb: bytes, req: GenerationRequest, on_progress: ProgressCallback) -> bytes:
        seed, hue = _seed_hue(req, salt=b"retexture")
        await self._stages(on_progress, ["Analizando UVs", "Pintando textura", "Horneando materiales"])
        mesh = geometry.to_color_mesh(geometry.load_scene(glb))
        rng = np.random.default_rng(seed)
        if req.style == "sculpture":
            return geometry.mesh_to_glb(geometry.recolor(mesh, geometry.CLAY, rng))
        out = geometry.recolor(mesh, hue, rng)
        text = (req.prompt or "").lower()
        colors = out.visual.vertex_colors.astype(float)
        if any(w in text for w in ("rayas", "rayado", "stripes", "striped")):
            band = (np.floor(out.vertices[:, 1] / (out.extents[1] / 10)) % 2 == 0)
            colors[band, :3] = colors[band, :3] * 0.35 + 255 * 0.65
        elif any(w in text for w in ("lunares", "puntos", "manchas", "spots", "dots")):
            centers = out.vertices[rng.choice(len(out.vertices), size=min(25, len(out.vertices)), replace=False)]
            radius = out.extents.max() * 0.06
            near = np.min(np.linalg.norm(out.vertices[:, None] - centers[None], axis=2), axis=1) < radius
            colors[near, :3] = 250
        out.visual.vertex_colors = colors.astype(np.uint8)
        return geometry.mesh_to_glb(out)

    async def estimate_depth(self, image: InputImage) -> bytes:
        await asyncio.sleep(self.step_delay)
        return await asyncio.to_thread(pseudo_depth, image.data)
