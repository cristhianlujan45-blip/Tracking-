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

from .animations import Clip
from .skeleton import JOINT_NAMES, PARENTS, Skeleton

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


def write_glb(primitives: list[Primitive], skeleton: Skeleton, clips: list[Clip],
              deform_joints: list[str] | None = None) -> bytes:
    buf = _Buffer()
    n_joints = len(JOINT_NAMES)
    joint_node = lambda j: 2 + j  # noqa: E731 - node 0: Armature, node 1: Character

    # ---------------- nodes
    nodes: list[dict] = [{"name": "Armature", "children": [joint_node(0)]}, {"name": "Character", "mesh": 0, "skin": 0}]
    for j, name in enumerate(JOINT_NAMES):
        parent = PARENTS[j]
        local = skeleton.positions[j] - (skeleton.positions[parent] if parent >= 0 else 0)
        node = {"name": name, "translation": [float(x) for x in local]}
        children = [joint_node(c) for c, p in enumerate(PARENTS) if p == j]
        if children:
            node["children"] = children
        nodes.append(node)

    ibm = np.tile(np.eye(4, dtype=np.float32), (n_joints, 1, 1))
    ibm[:, 3, :3] = -skeleton.positions  # column-major translation of the inverse bind matrix
    skin = {"name": "Skeleton", "joints": [joint_node(j) for j in range(n_joints)], "skeleton": joint_node(0),
            "inverseBindMatrices": buf.accessor(ibm.reshape(n_joints, 16), FLOAT, "MAT4")}

    # ---------------- meshes & materials
    materials, textures, images = [], [], []
    gl_prims = []
    for prim in primitives:
        attrs = {
            "POSITION": buf.accessor(prim.positions, FLOAT, "VEC3", ARRAY_BUFFER, bounds=True),
            "NORMAL": buf.accessor(prim.normals / np.maximum(np.linalg.norm(prim.normals, axis=1, keepdims=True), 1e-12),
                                   FLOAT, "VEC3", ARRAY_BUFFER),
            "JOINTS_0": buf.accessor(prim.joints, USHORT, "VEC4", ARRAY_BUFFER),
            "WEIGHTS_0": buf.accessor(prim.weights, FLOAT, "VEC4", ARRAY_BUFFER),
        }
        material = {"name": f"{prim.name}_material",
                    "pbrMetallicRoughness": {"baseColorFactor": [float(c) for c in prim.base_color],
                                             "metallicFactor": 0.0, "roughnessFactor": 0.8}}
        if prim.uv is not None and prim.texture_png:
            attrs["TEXCOORD_0"] = buf.accessor(prim.uv, FLOAT, "VEC2", ARRAY_BUFFER)
            images.append({"bufferView": buf.view(prim.texture_png), "mimeType": "image/png"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
        if prim.colors is not None:
            attrs["COLOR_0"] = buf.accessor(prim.colors, UBYTE, "VEC4", ARRAY_BUFFER, normalized=True)
        materials.append(material)
        gl_prims.append({"attributes": attrs, "material": len(materials) - 1,
                         "indices": buf.accessor(prim.faces.reshape(-1), UINT, "SCALAR", ELEMENT_ARRAY_BUFFER)})

    # ---------------- animations
    animated = [JOINT_NAMES.index(n) for n in (deform_joints or [n for n in JOINT_NAMES if not n.endswith("_End")])]
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
