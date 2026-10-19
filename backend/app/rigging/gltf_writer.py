"""Minimal glTF 2.0 (GLB) writer for skinned, animated characters.

trimesh cannot write skins or animations, so this module serialises them directly. The output is
checked against the Khronos glTF validator in the tests and imports in Blender, three.js, Unity
(glTFast) and Unreal (Interchange).

Layout: an "Armature" root node holding the joint hierarchy (rest rotations are identity, so each
joint's translation is its offset from its parent) and a root "Character" node with the skinned
mesh. One animation per clip, rotations on every deforming joint plus the Hips translation.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass

import numpy as np

from ..geometry import srgb_to_linear
from .animations import Clip
from .skeleton import Skeleton

FLOAT, UINT, USHORT, UBYTE = 5126, 5125, 5123, 5121
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass
class Primitive:
    positions: np.ndarray  # (n, 3) float
    normals: np.ndarray  # (n, 3) float
    faces: np.ndarray  # (m, 3) int
    joints: np.ndarray  # (n, 4) uint16, indices into JOINT_NAMES
    weights: np.ndarray  # (n, 4) float, rows sum to 1
    uv: np.ndarray | None = None  # (n, 2), glTF convention (v down)
    colors: np.ndarray | None = None  # (n, 4) uint8
    texture_png: bytes | None = None
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    name: str = "Body"
    # extra PBR maps (PNG) sharing TEXCOORD_0: "normal", "metallicRoughness", "occlusion", "emissive"
    maps: dict[str, bytes] | None = None
    metallic: float = 0.0
    roughness: float = 0.8
    emissive: tuple[float, float, float] | None = None


class _Buffer:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, raw: bytes, target: int | None = None) -> int:
        self.data += b"\0" * (-len(self.data) % 4)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            view["target"] = target
        self.data += raw
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, array: np.ndarray, component: int, kind: str, target: int | None = None,
                 normalized: bool = False, bounds: bool = False) -> int:
        dtype = {FLOAT: np.float32, UINT: np.uint32, USHORT: np.uint16, UBYTE: np.uint8}[component]
        arr = np.ascontiguousarray(array, dtype=dtype)
        acc = {
            "bufferView": self.view(arr.tobytes(), target),
            "componentType": component,
            "count": int(arr.shape[0]),
            "type": kind,
        }
        if normalized:
            acc["normalized"] = True
        if bounds:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _continuous(quats: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive keys stay in one hemisphere (no spins when interpolating)."""
    q = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def tangents(positions: np.ndarray, normals: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (xyz + handedness w) from UV derivatives (Lengyel), for normal mapping."""
    p0, p1, p2 = (positions[faces[:, i]] for i in range(3))
    t0, t1, t2 = (uv[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    tan, bit = np.zeros_like(positions, dtype=float), np.zeros_like(positions, dtype=float)
    for i in range(3):
        np.add.at(tan, faces[:, i], sdir)
        np.add.at(bit, faces[:, i], tdir)
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    t = tan - n * np.sum(n * tan, axis=1, keepdims=True)  # Gram-Schmidt
    length = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    t = np.where(length > 1e-12, t / np.maximum(length, 1e-12), fallback / np.linalg.norm(fallback, axis=1, keepdims=True))
    # glTF UVs have V pointing down, which flips the bitangent relative to the UV derivative
    w = np.where(np.sum(np.cross(n, t) * bit, axis=1) < 0, 1.0, -1.0)
    return np.hstack([t, w[:, None]])


def write_glb(primitives: list[Primitive], skeleton: Skeleton, clips: list[Clip],
              deform_joints: list[str] | None = None) -> bytes:
    buf = _Buffer()
    names, parents = skeleton.template.names, skeleton.template.parents
    n_joints = len(names)
    joint_node = lambda j: 2 + j  # noqa: E731 - node 0: Armature, node 1: Character

    # ---------------- nodes
    nodes: list[dict] = [{"name": "Armature", "children": [joint_node(0)]}, {"name": "Character", "mesh": 0, "skin": 0}]
    for j, name in enumerate(names):
        parent = parents[j]
        local = skeleton.positions[j] - (skeleton.positions[parent] if parent >= 0 else 0)
        node = {"name": name, "translation": [float(x) for x in local]}
        children = [joint_node(c) for c, p in enumerate(parents) if p == j]
        if children:
            node["children"] = children
        nodes.append(node)

    ibm = np.tile(np.eye(4, dtype=np.float32), (n_joints, 1, 1))
    ibm[:, 3, :3] = -skeleton.positions  # column-major translation of the inverse bind matrix
    skin = {"name": "Skeleton", "joints": [joint_node(j) for j in range(n_joints)], "skeleton": joint_node(0),
            "inverseBindMatrices": buf.accessor(ibm.reshape(n_joints, 16), FLOAT, "MAT4")}

    # ---------------- meshes & materials
    materials, textures, images = [], [], []
    image_index: dict[bytes, int] = {}  # identical maps (e.g. one ORM image used twice) are stored once
    gl_prims = []

    def texture(png: bytes) -> int:
        if png not in image_index:
            images.append({"bufferView": buf.view(png), "mimeType": "image/png"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            image_index[png] = len(textures) - 1
        return image_index[png]

    for prim in primitives:
        attrs = {
            "POSITION": buf.accessor(prim.positions, FLOAT, "VEC3", ARRAY_BUFFER, bounds=True),
            "NORMAL": buf.accessor(prim.normals / np.maximum(np.linalg.norm(prim.normals, axis=1, keepdims=True), 1e-12),
                                   FLOAT, "VEC3", ARRAY_BUFFER),
            "JOINTS_0": buf.accessor(prim.joints, USHORT, "VEC4", ARRAY_BUFFER),
            "WEIGHTS_0": buf.accessor(prim.weights, FLOAT, "VEC4", ARRAY_BUFFER),
        }
        pbr = {"baseColorFactor": [float(c) for c in prim.base_color],
               "metallicFactor": float(prim.metallic), "roughnessFactor": float(prim.roughness)}
        material = {"name": f"{prim.name}_material", "pbrMetallicRoughness": pbr}
        if prim.uv is not None and (prim.texture_png or prim.maps):
            attrs["TEXCOORD_0"] = buf.accessor(prim.uv, FLOAT, "VEC2", ARRAY_BUFFER)
            if prim.texture_png:
                pbr["baseColorTexture"] = {"index": texture(prim.texture_png)}
            maps = prim.maps or {}
            if "metallicRoughness" in maps:
                pbr["metallicRoughnessTexture"] = {"index": texture(maps["metallicRoughness"])}
            if "normal" in maps:
                material["normalTexture"] = {"index": texture(maps["normal"])}
                attrs["TANGENT"] = buf.accessor(tangents(prim.positions, prim.normals, prim.uv, prim.faces),
                                                FLOAT, "VEC4", ARRAY_BUFFER)
            if "occlusion" in maps:
                material["occlusionTexture"] = {"index": texture(maps["occlusion"])}
            if "emissive" in maps:
                material["emissiveTexture"] = {"index": texture(maps["emissive"])}
                material["emissiveFactor"] = [float(c) for c in (prim.emissive or (1.0, 1.0, 1.0))]
        if prim.emissive and "emissiveFactor" not in material and any(prim.emissive):
            material["emissiveFactor"] = [float(c) for c in prim.emissive]
        if prim.colors is not None:
            # glTF vertex colours are linear; the app keeps them in sRGB
            attrs["COLOR_0"] = buf.accessor(srgb_to_linear(prim.colors), UBYTE, "VEC4", ARRAY_BUFFER, normalized=True)
        materials.append(material)
        gl_prims.append({"attributes": attrs, "material": len(materials) - 1,
                         "indices": buf.accessor(prim.faces.reshape(-1), UINT, "SCALAR", ELEMENT_ARRAY_BUFFER)})

    # ---------------- animations
    animated = [names.index(n) for n in (deform_joints or skeleton.template.deform)]
    animations = []
    for clip in clips:
        times = buf.accessor(clip.times.astype(np.float32), FLOAT, "SCALAR", bounds=True)
        samplers, channels = [], []
        for j in animated:
            out = buf.accessor(_continuous(clip.rotations[:, j].copy()), FLOAT, "VEC4")
            samplers.append({"input": times, "output": out, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": joint_node(j), "path": "rotation"}})
        hips_t = skeleton.positions[0][None] + clip.hips_offset
        samplers.append({"input": times, "output": buf.accessor(hips_t, FLOAT, "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": joint_node(0), "path": "translation"}})
        animations.append({"name": clip.name, "samplers": samplers, "channels": channels})

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "Forja3D auto-rig"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0, 1]}],
        "nodes": nodes,
        "skins": [skin],
        "meshes": [{"name": "Character", "primitives": gl_prims}],
        "materials": materials,
        "buffers": [{"byteLength": 0}],
        "bufferViews": buf.views,
        "accessors": buf.accessors,
    }
    if animations:
        gltf["animations"] = animations
    if images:
        gltf["images"], gltf["textures"] = images, textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]

    binary = bytes(buf.data) + b"\0" * (-len(buf.data) % 4)
    gltf["buffers"][0]["byteLength"] = len(binary)
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    out = io.BytesIO()
    out.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(binary)))
    out.write(struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk)
    out.write(struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    return out.getvalue()
