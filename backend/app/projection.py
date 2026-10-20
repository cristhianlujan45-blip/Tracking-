"""Colour a mesh from its reference views (front / left / back / right), like a 3D artist projecting
a model sheet. Used when the shape engine returns an untextured mesh (Hunyuan3D-2mini on an 8 GB GPU
has no room for its paint model) and for character sheets.

Each view is an orthographic picture of the model (transparent background). The model is assumed
to face +Z with +Y up, as in glTF. For every vertex the views that see it (depth-tested) are blended
by how squarely they face the surface; vertices no view sees (top of the head, armpits…) get the
colour of their neighbours by diffusion over the mesh.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage, sparse

from .geometry import load_scene, mesh_to_glb

# camera direction (from the model towards the camera) and the image axes in model space
VIEWS = {
    "front": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0])),
    "back": (np.array([0, 0, -1.0]), np.array([-1.0, 0, 0])),
    "left": (np.array([1.0, 0, 0]), np.array([0, 0, -1.0])),   # the model's left side: its face looks image-left
    "right": (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
}
UP = np.array([0, 1.0, 0])


class ProjectionError(Exception):
    pass


def _alpha_box(img: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(img[..., 3] > 128)
    if not len(xs):
        raise ProjectionError("La vista no tiene silueta (¿fondo sin quitar?)")
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear RGB and alpha at pixel coordinates (u right, v down)."""
    h, w = img.shape[:2]
    u, v = np.clip(u, 0, w - 1.001), np.clip(v, 0, h - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return c[:, :3], c[:, 3]


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _align(a: np.ndarray, b: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int],
           grid: int = 192) -> tuple[float, float, float, float]:
    """Fit the mesh's silhouette onto the picture's: returns (sx, sy, cx, cy) so that pixel
    u = cx + (a - mid_a) * sx, v = cy - (b - mid_b) * sy. Starts from the bounding boxes and refines
    scale and offset by maximising the overlap (IoU) of the two silhouettes."""
    x0, y0, x1, y1 = box
    mid_a, mid_b = (a.max() + a.min()) / 2, (b.max() + b.min()) / 2
    s0 = (y1 - y0) / max(np.ptp(b), 1e-9)
    base = (s0, s0, (x0 + x1) / 2, (y0 + y1) / 2)
    h, w = mask.shape
    k = grid / max(h, w)  # work on a small copy of the picture
    small = ndimage.zoom(mask.astype(float), k, order=1) > 0.5
    gh, gw = small.shape
    area = small.sum()
    sub = np.random.default_rng(0).choice(len(a), min(len(a), 40000), replace=False)
    da, db = a[sub] - mid_a, b[sub] - mid_b

    def iou(sx, sy, cx, cy):
        u = np.round((cx + da * sx) * k).astype(int)
        v = np.round((cy - db * sy) * k).astype(int)
        ok = (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
        sil = np.zeros_like(small)
        sil[v[ok], u[ok]] = True
        sil = ndimage.binary_closing(sil, iterations=2)
        inter = (sil & small).sum()
        return inter / max(sil.sum() + area - inter, 1)

    best, best_score = base, iou(*base)
    span = (y1 - y0)
    for step in (0.04, 0.015):  # coarse then fine
        improved = True
        while improved:
            improved = False
            sx, sy, cx, cy = best
            for cand in ((sx * (1 + step), sy, cx, cy), (sx * (1 - step), sy, cx, cy),
                         (sx, sy * (1 + step), cx, cy), (sx, sy * (1 - step), cx, cy),
                         (sx, sy, cx + step * span, cy), (sx, sy, cx - step * span, cy),
                         (sx, sy, cx, cy + step * span), (sx, sy, cx, cy - step * span)):
                if not 0.85 < cand[0] / s0 < 1.15 or not 0.9 < cand[1] / s0 < 1.1:
                    continue
                score = iou(*cand)
                if score > best_score + 1e-4:
                    best, best_score, improved = cand, score, True
    return best


def _visible(pts2d: np.ndarray, depth: np.ndarray, res: int, tol: float) -> np.ndarray:
    """Depth test by splatting vertices into a z-buffer (dense meshes cover every pixel)."""
    lo, hi = pts2d.min(0), pts2d.max(0)
    scale = (res - 1) / max(hi - lo)
    px = np.round((pts2d - lo) * scale).astype(int)
    zbuf = np.full((res, res), -np.inf)
    np.maximum.at(zbuf, (px[:, 1], px[:, 0]), depth)
    # close pinholes: a pixel takes the nearest surface among its neighbours
    padded = np.pad(zbuf, 1, constant_values=-np.inf)
    near = np.max([padded[1 + dy:res + 1 + dy, 1 + dx:res + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return depth >= near[px[:, 1], px[:, 0]] - tol


def project_views(mesh: trimesh.Trimesh, views: dict[str, Image.Image], mirror_side: bool = True,
                  depth_res: int = 384) -> trimesh.Trimesh:
    views = {k: v for k, v in views.items() if k in VIEWS and v is not None}
    if not views:
        raise ProjectionError("Hacen falta vistas (frente, lateral o espalda) para texturizar")
    if mirror_side:  # symmetric characters: the missing side is the mirror of the other one
        for a, b in (("left", "right"), ("right", "left")):
            if a in views and b not in views:
                views[b] = views[a].transpose(Image.FLIP_LEFT_RIGHT)

    mesh = mesh.copy()
    verts = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    size = float(np.ptp(verts, axis=0).max())
    color_sum = np.zeros((len(verts), 3))
    weight_sum = np.zeros(len(verts))

    for name, image in views.items():
        cam, right = VIEWS[name]
        img = np.asarray(image.convert("RGBA"), dtype=float)
        box = _alpha_box(img)
        mask = img[..., 3] > 128
        a, b = verts @ right, verts @ UP  # image axes in model units
        sx, sy, cx, cy = _align(a, b, mask, box)
        u = cx + (a - (a.max() + a.min()) / 2) * sx
        v = cy - (b - (b.max() + b.min()) / 2) * sy
        rgb, alpha = _sample(img, u, v)
        # outlines and anti-aliasing live at the edge of the drawing: trust pixels well inside it
        inside = ndimage.distance_transform_edt(mask)
        edge = _smoothstep(_sample(np.dstack([inside] * 4), u, v)[1], 1.0, 0.012 * (box[3] - box[1]) + 2)
        facing = normals @ cam
        seen = _visible(np.column_stack([a, b]), verts @ cam, depth_res, 0.012 * size)
        # a picture only describes surfaces it sees squarely; grazing ones would smear its pixels
        w = _smoothstep(facing, 0.3, 0.8) * facing ** 2 * (alpha / 255.0) * edge * seen
        if name in ("left", "right"):
            w *= 0.8  # side pictures are usually narrower: trust them a little less
        color_sum += rgb * w[:, None]
        weight_sum += w

    known = weight_sum > 0.02
    colors = np.zeros((len(verts), 3))
    colors[known] = color_sum[known] / weight_sum[known, None]
    colors = _diffuse(mesh, colors, known)
    rgba = np.column_stack([np.clip(colors, 0, 255), np.full(len(verts), 255)]).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    return mesh


ROLES = ["front", "left", "back", "right"]  # order of the app's multi-view slots


def paint_glb(glb: bytes, images: list[bytes], roles: list[str] | None = None) -> bytes:
    """Vertex-colour an untextured model from its input pictures; textured models are returned as is."""
    from .sheet import remove_background

    scene = load_scene(glb)
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes or any(getattr(m.visual, "kind", None) == "texture" for m in meshes):
        return glb
    views = {}
    for role, data in zip(roles or ROLES, images):
        views[role] = remove_background(Image.open(io.BytesIO(data)))
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    return mesh_to_glb(project_views(mesh, views))


def _diffuse(mesh: trimesh.Trimesh, colors: np.ndarray, known: np.ndarray, iterations: int = 400) -> np.ndarray:
    """Fill unseen vertices with the average of their neighbours until it settles."""
    if known.all():
        return colors
    if not known.any():
        return np.full_like(colors, 180.0)
    edges = mesh.edges_unique
    n = len(mesh.vertices)
    adj = sparse.coo_matrix((np.ones(len(edges) * 2), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                            shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(1)).ravel()
    degree[degree == 0] = 1
    out = colors.copy()
    out[~known] = colors[known].mean(0)
    for _ in range(iterations):
        out[~known] = (adj @ out)[~known] / degree[~known, None]
    return out
