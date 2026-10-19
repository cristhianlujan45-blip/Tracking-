"""Offline provider: builds a procedural, vertex-colored mesh so the whole app works without a GPU or API key.

The shape is picked from keywords in the prompt (or from a hash of the input), so the same input
always gives the same model.
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import trimesh

from .base import GenerationRequest, GenerationResult, ProgressCallback

KEYWORDS = {
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


def build_shape(kind: str, rng: np.random.Generator, hue: np.ndarray) -> trimesh.Trimesh:
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


class MockProvider:
    name = "mock"

    def __init__(self, step_delay: float = 0.6):
        self.step_delay = step_delay

    async def generate(self, req: GenerationRequest, on_progress: ProgressCallback) -> GenerationResult:
        digest = hashlib.sha256((req.prompt or "").encode() + (req.image_bytes or b"")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        hue = pick_color(req.prompt)
        if hue is None:
            hue = np.array([digest[8], digest[9], digest[10]], dtype=float) / 255 * 0.7 + 0.25

        for i, stage in enumerate(STAGES):
            await on_progress(int(i / len(STAGES) * 100), stage)
            await asyncio.sleep(self.step_delay)

        mesh = build_shape(pick_kind(req.prompt, seed), rng, hue)
        glb = trimesh.Scene(mesh).export(file_type="glb")
        return GenerationResult(model_glb=glb)
