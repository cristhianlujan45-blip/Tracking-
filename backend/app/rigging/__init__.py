"""Auto-rigging and animation for humanoid characters.

Pipeline: normalise the model (facing +Z, feet on the floor) → markers (automatic or the user's)
→ skeleton → heat-diffusion skin weights → animation clips (library + BVH) → skinned GLB.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import trimesh

from ..geometry import GeometryError, load_scene
from .animations import DEFAULT_PRESETS, PRESETS, Clip, preset_clip, retarget_bvh
from .gltf_writer import Primitive, write_glb
from .skeleton import (DEFORM_JOINTS, JOINT_INDEX, MARKER_LABELS, MARKERS, RigError, Skeleton, build_skeleton,
                       estimate_markers, guess_yaw, mirror_markers, yaw_matrix)
from .skinning import compute_skin, limit_influences

__all__ = ["PRESETS", "DEFAULT_PRESETS", "MARKERS", "MARKER_LABELS", "RigError", "RigResult", "prepare", "auto_markers",
           "rig", "mirror_markers"]


@dataclass
class Prepared:
    meshes: list[trimesh.Trimesh]  # normalised: facing +Z, centred on x/z, feet at y=0
    combined: trimesh.Trimesh
    yaw: int


@dataclass
class RigResult:
    glb: bytes
    markers: dict[str, list[float]]
    yaw: int
    skeleton: dict[str, list[float]]
    clips: list[str]


def prepare(glb: bytes, yaw: int | None = None) -> Prepared:
    try:
        scene = load_scene(glb)
    except GeometryError as exc:
        raise RigError(str(exc)) from exc
    meshes = [m.copy() for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise RigError("El modelo no contiene mallas")
    combined = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes])
    if yaw is None:
        yaw = guess_yaw(combined.vertices)
    transform = yaw_matrix(yaw)
    combined.apply_transform(transform)
    lo, hi = combined.bounds
    shift = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    combined.apply_translation(shift)
    for m in meshes:
        m.apply_transform(transform)
        m.apply_translation(shift)
    return Prepared(meshes, combined, int(yaw) % 360)


def auto_markers(glb: bytes, yaw: int | None = None) -> dict:
    prep = prepare(glb, yaw)
    return {"yaw": prep.yaw, "markers": estimate_markers(prep.combined),
            "bounds": [[float(x) for x in b] for b in prep.combined.bounds]}


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
        bvh: list[tuple[str, str]] | None = None, in_place: bool = True) -> RigResult:
    """Rig a humanoid and bake the requested clips. `bvh` is a list of (name, file text)."""
    prep = prepare(glb, yaw)
    used_markers = markers or estimate_markers(prep.combined)
    skeleton = build_skeleton(prep.combined.vertices, used_markers)

    weights = compute_skin(prep.combined, skeleton)
    deform_index = np.array([JOINT_INDEX[n] for n in DEFORM_JOINTS], dtype=np.uint16)
    primitives, start = [], 0
    for i, mesh in enumerate(prep.meshes):
        n = len(mesh.vertices)
        idx, w = limit_influences(weights[start:start + n])
        primitives.append(_primitive(mesh, deform_index[idx], w, i))
        start += n

    clips: list[Clip] = [preset_clip(skeleton, name) for name in (presets if presets is not None else DEFAULT_PRESETS)]
    for name, text in bvh or []:
        clips.append(retarget_bvh(skeleton, text, name=name, in_place=in_place))
    return RigResult(write_glb(primitives, skeleton, clips), used_markers, prep.yaw, skeleton.as_dict(),
                     [c.name for c in clips])
