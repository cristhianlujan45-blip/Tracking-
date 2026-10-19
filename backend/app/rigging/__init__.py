"""Auto-rigging and animation for humanoid and quadruped characters.

Pipeline: normalise the model (facing +Z, feet on the floor) → markers (automatic or the user's)
→ skeleton → heat-diffusion skin weights → animation clips (library, mocap, BVH) → skinned GLB.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import trimesh

from ..geometry import GeometryError, load_scene
from . import mocap, quadruped
from .animations import DEFAULT_PRESETS, PRESETS, Clip, preset_clip, retarget_bvh
from .gltf_writer import Primitive, write_glb
from .skeleton import (HUMANOID, MARKER_LABELS, MARKERS, RigError, Template, build_skeleton, estimate_markers,
                       guess_yaw, mirror_markers, yaw_matrix)
from .skinning import compute_skin, limit_influences

__all__ = ["mocap", "quadruped", "PRESETS", "DEFAULT_PRESETS", "MARKERS", "MARKER_LABELS", "TEMPLATES", "RigError",
           "RigResult", "prepare", "auto_markers", "rig", "mirror_markers", "presets_for", "default_presets"]

TEMPLATES: dict[str, Template] = {"humanoid": HUMANOID, "quadruped": quadruped.QUADRUPED}


def presets_for(template: str) -> dict:
    return quadruped.QUAD_PRESETS if template == "quadruped" else PRESETS


def default_presets(template: str) -> list[str]:
    return quadruped.DEFAULT_QUAD_PRESETS if template == "quadruped" else DEFAULT_PRESETS


@dataclass
class Prepared:
    meshes: list[trimesh.Trimesh]  # normalised: facing +Z, centred on x/z, feet at y=0
    combined: trimesh.Trimesh
    yaw: int
    template: str


@dataclass
class RigResult:
    glb: bytes
    markers: dict[str, list[float]]
    yaw: int
    skeleton: dict[str, list[float]]
    clips: list[str]
    template: str


def _detect_template(vertices: np.ndarray) -> str:
    ext = np.ptp(vertices, axis=0)
    return "quadruped" if quadruped.looks_quadruped(vertices) and max(ext[0], ext[2]) > 1.15 * ext[1] else "humanoid"


def prepare(glb: bytes, yaw: int | None = None, template: str | None = None) -> Prepared:
    try:
        scene = load_scene(glb)
    except GeometryError as exc:
        raise RigError(str(exc)) from exc
    meshes = [m.copy() for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise RigError("El modelo no contiene mallas")
    combined = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes])
    if template in (None, "auto"):
        template = _detect_template(combined.vertices)
    if template not in TEMPLATES:
        raise RigError(f"Tipo de esqueleto desconocido: {template}")
    if yaw is None:
        yaw = quadruped.guess_yaw(combined.vertices) if template == "quadruped" else guess_yaw(combined.vertices)
    transform = yaw_matrix(yaw)
    combined.apply_transform(transform)
    lo, hi = combined.bounds
    shift = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    combined.apply_translation(shift)
    for m in meshes:
        m.apply_transform(transform)
        m.apply_translation(shift)
    return Prepared(meshes, combined, int(yaw) % 360, template)


def _estimate(prep: Prepared) -> dict:
    return quadruped.estimate_markers(prep.combined) if prep.template == "quadruped" else estimate_markers(prep.combined)


def auto_markers(glb: bytes, yaw: int | None = None, template: str | None = None) -> dict:
    prep = prepare(glb, yaw, template)
    t = TEMPLATES[prep.template]
    return {"yaw": prep.yaw, "template": prep.template, "view": t.view, "markers": _estimate(prep),
            "labels": t.marker_labels, "bounds": [[float(x) for x in b] for b in prep.combined.bounds]}


def _png(image) -> bytes:
    out = io.BytesIO()
    image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB").save(out, format="PNG")
    return out.getvalue()


def _primitive(mesh: trimesh.Trimesh, joints: np.ndarray, weights: np.ndarray, index: int) -> Primitive:
    prim = Primitive(positions=np.asarray(mesh.vertices), normals=np.asarray(mesh.vertex_normals),
                     faces=np.asarray(mesh.faces), joints=joints, weights=weights, name=f"Body{index}")
    visual = mesh.visual
    if visual.kind == "texture":
        material = getattr(visual, "material", None)
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            prim.base_color = tuple(float(c) / (255.0 if np.max(factor) > 1 else 1.0) for c in factor)
        if image is not None and visual.uv is not None and len(visual.uv) == len(mesh.vertices):
            uv = np.asarray(visual.uv, dtype=float).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # trimesh flips V on import
            prim.uv, prim.texture_png = uv, _png(image)
        elif factor is None:
            try:
                prim.colors = visual.to_color().vertex_colors
            except Exception:  # noqa: BLE001
                pass
    elif visual.kind in ("vertex", "face"):
        prim.colors = np.asarray(visual.vertex_colors, dtype=np.uint8)
    return prim


def rig(glb: bytes, markers: dict | None = None, yaw: int | None = None, presets: list[str] | None = None,
        bvh: list[tuple[str, str]] | None = None, in_place: bool = True, template: str | None = None) -> RigResult:
    """Rig a character and bake the requested clips. `bvh` is a list of (name, file text)."""
    prep = prepare(glb, yaw, template)
    used_markers = markers or _estimate(prep)
    if prep.template == "quadruped":
        if "_leg_x" not in used_markers:  # the editor only moves the visible handles
            used_markers = {**used_markers, "_leg_x": quadruped.estimate_markers(prep.combined)["_leg_x"]}
        skeleton = quadruped.build_skeleton(prep.combined.vertices, used_markers)
    else:
        skeleton = build_skeleton(prep.combined.vertices, used_markers)
    tpl = skeleton.template

    weights = compute_skin(prep.combined, skeleton)
    deform_index = np.array([tpl.index[n] for n in tpl.deform], dtype=np.uint16)
    primitives, start = [], 0
    for i, mesh in enumerate(prep.meshes):
        n = len(mesh.vertices)
        idx, w = limit_influences(weights[start:start + n])
        primitives.append(_primitive(mesh, deform_index[idx], w, i))
        start += n

    library = presets_for(prep.template)
    names = presets if presets is not None else default_presets(prep.template)
    if template in (None, "auto"):  # the caller did not know the body type: drop clips made for the other one
        names = [n for n in names if n in library or (prep.template == "humanoid" and n in mocap.catalogue())]
        if presets and not names and not bvh:
            names = default_presets(prep.template)
    clips: list[Clip] = []
    for name in names:
        if name in library:
            clips.append(preset_clip(skeleton, name))
        elif name in mocap.catalogue() and prep.template == "humanoid":
            clips.append(retarget_bvh(skeleton, mocap.clip_text(name), name=mocap.catalogue()[name]["label"],
                                      in_place=in_place))
        else:
            raise RigError(f"Animación no disponible para este tipo de esqueleto: {name}")
    for name, text in bvh or []:
        clips.append(retarget_bvh(skeleton, text, name=name, in_place=in_place))
    return RigResult(write_glb(primitives, skeleton, clips), used_markers, prep.yaw, skeleton.as_dict(),
                     [c.name for c in clips], prep.template)
