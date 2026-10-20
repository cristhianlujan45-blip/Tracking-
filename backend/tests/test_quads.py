"""Quad remesh for animation and editing: all quads, a UV unwrap that never cuts a quad, the texture kept."""

import io
import zipfile

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from app import geometry, quads

pytest.importorskip("pynanoinstantmeshes")


def two_tone_capsule() -> bytes:
    """Red top half, blue bottom half (vertex colours, like a painted AI model)."""
    mesh = trimesh.creation.capsule(height=1.2, radius=0.4, count=[48, 48])
    mesh = mesh.subdivide()
    colours = np.where(mesh.vertices[:, 2:3] > 0, [[220, 40, 40, 255]], [[40, 60, 220, 255]]).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colours)
    return geometry.mesh_to_glb(mesh)


@pytest.fixture(scope="module")
def remeshed():
    glb = two_tone_capsule()
    return glb, *quads.quad_remesh(glb, 1500, texture=512)


def test_all_quads_closed_and_near_the_target(remeshed):
    _, _, files, qm = remeshed
    assert qm.quads == len(qm.faces) and 1000 < qm.quads < 2200
    edges = {}
    for f in qm.faces:
        for a, b in zip(f, np.roll(f, -1)):
            edges.setdefault((min(a, b), max(a, b)), []).append((a, b))
    assert all(len(e) == 2 and e[0] != e[1] for e in edges.values())  # closed, manifold, one winding
    obj = files["model.obj"].decode().splitlines()
    faces = [line.split()[1:] for line in obj if line.startswith("f ")]
    assert len(faces) == qm.quads and all(len(f) == 4 and all("/" in c for c in f) for f in faces)
    assert "mtllib model.mtl" in obj and b"map_Kd model.png" in files["model.mtl"]
    assert Image.open(io.BytesIO(files["model.png"])).size == (512, 512)


def test_uv_charts_are_whole_quads_unflipped_and_not_overlapping(remeshed):
    _, _, _, qm = remeshed
    assert all(len(fu) == len(f) for f, fu in zip(qm.faces, qm.face_uv))  # one UV per corner: no quad cut
    assert qm.uv.min() >= 0 and qm.uv.max() <= 1
    tris = np.array([[fu[0], fu[i], fu[i + 1]] for fu in qm.face_uv for i in range(1, len(fu) - 1)])
    assert (quads._signed_areas(qm.uv, tris) > 0).all()  # no chart mirrored or folded
    # charts (faces linked through shared UVs) must not cover each other's texels
    parent = list(range(len(qm.uv)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for fu in qm.face_uv:
        for j in fu[1:]:
            parent[root(j)] = root(fu[0])
    chart_of = np.array([root(fu[0]) for fu in qm.face_uv])
    assert len(set(chart_of)) > 1
    size = 1024
    seen = np.zeros((size, size), np.int32)
    for chart in set(chart_of):
        img = Image.new("1", (size, size), 0)
        draw = ImageDraw.Draw(img)
        for fu in (fu for fu, c in zip(qm.face_uv, chart_of) if c == chart):
            draw.polygon([tuple(p) for p in qm.uv[fu] * size], fill=1)
        seen += np.asarray(img, dtype=np.int32)
    assert (seen > 1).sum() == 0 and (seen > 0).mean() > 0.3  # no overlap, and a well-filled texture


def test_texture_keeps_the_colours(remeshed):
    _, glb, _, qm = remeshed
    back = geometry.to_color_mesh(geometry.load_scene(glb))
    top = back.visual.vertex_colors[back.vertices[:, 2] > 0.25, :3].mean(0)
    bottom = back.visual.vertex_colors[back.vertices[:, 2] < -0.25, :3].mean(0)
    assert top[0] > 180 and top[2] < 80 and bottom[2] > 180 and bottom[0] < 80
    # a model that already has a texture is baked from it too
    again, _, qm2 = quads.quad_remesh(glb, 800, texture=256)
    mesh = geometry.to_color_mesh(geometry.load_scene(again))
    assert mesh.visual.vertex_colors[mesh.vertices[:, 2] > 0.25, 0].mean() > 170 and qm2.quads == len(qm2.faces)
