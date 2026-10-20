"""Colour a mesh from its reference views (front / left / back / right), like a 3D artist projecting
a model sheet. Used when the shape engine returns an untextured mesh (Hunyuan3D-2mini on an 8 GB GPU
has no room for its paint model) and for character sheets.

Each view is an orthographic picture of the model (transparent background). The model is assumed
to face +Z with +Y up, as in glTF. For every point the views that see it (depth-tested) are blended
by how squarely they face the surface; points no view sees (top of the head, armpits…) get the
colour of their neighbours by diffusion over the mesh. With a single picture the back is painted
from the front's colour regions (mirrored, without its details: no second face on the back), unless
the local AI generated side/back views for it: those come with the pose (azimuth, elevation) they
were really seen from and a lower trust than real pictures.

`paint_glb` bakes the result into a real UV texture (xatlas unwrap, 2048 px), so the detail of the
pictures survives instead of being limited by the vertex count; without xatlas it falls back to
vertex colours.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage, sparse

from .geometry import load_scene, mesh_to_glb


def view_axes(azimuth: float, elevation: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera direction (from the model towards the camera), image right and image up in model space
    for a camera orbiting the model: azimuth 0 = front, 90 = the model's left side (+X), 180 = back;
    elevation > 0 looks down from above."""
    az, el = np.radians(azimuth), np.radians(elevation)
    cam = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.array([np.cos(az), 0.0, -np.sin(az)])
    return cam, right, np.cross(cam, right)


# the model's left side: camera at +X, its face looks image-left
AZIMUTHS = {"front": 0, "left": 90, "back": 180, "right": 270}
VIEWS = {name: view_axes(az) for name, az in AZIMUTHS.items()}


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


class _View:
    """One picture placed on the model: alignment, depth buffer and per-point colour + trust."""

    def __init__(self, name: str, image: Image.Image, verts: np.ndarray, size: float, depth_res: int,
                 trust: float = 1.0, pose: tuple[float, float] | None = None):
        # pose (azimuth, elevation) places pictures taken off the four axes, e.g. AI-generated views
        self.cam, self.right, self.up = view_axes(*pose) if pose else VIEWS[name]
        self.trust = trust * (0.8 if name in ("left", "right") else 1.0)  # side pictures are usually narrower
        self.img = np.asarray(image.convert("RGBA"), dtype=float)
        box = _alpha_box(self.img)
        mask = self.img[..., 3] > 128
        a, b = verts @ self.right, verts @ self.up  # image axes in model units
        self.mid = ((a.max() + a.min()) / 2, (b.max() + b.min()) / 2)
        self.sx, self.sy, self.cx, self.cy = _align(a, b, mask, box)
        # outlines and anti-aliasing live at the edge of the drawing: trust pixels well inside it
        self.inside = np.dstack([ndimage.distance_transform_edt(mask)] * 4)
        self.edge_px = 0.012 * (box[3] - box[1]) + 2
        # depth buffer from the (dense) vertices; a pixel takes the nearest surface among its neighbours
        pts = np.column_stack([a, b])
        self.lo = pts.min(0)
        self.res = depth_res
        self.zscale = (depth_res - 1) / max(np.ptp(pts, axis=0).max(), 1e-9)
        px = self._px(pts)
        zbuf = np.full((depth_res, depth_res), -np.inf)
        np.maximum.at(zbuf, (px[:, 1], px[:, 0]), verts @ self.cam)
        padded = np.pad(zbuf, 1, constant_values=-np.inf)
        self.near = np.max([padded[1 + dy:depth_res + 1 + dy, 1 + dx:depth_res + 1 + dx]
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        self.tol = 0.012 * size

    def _px(self, pts2d: np.ndarray) -> np.ndarray:
        return np.clip(np.round((pts2d - self.lo) * self.zscale).astype(int), 0, self.res - 1)

    def colour(self, points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = points @ self.right, points @ self.up
        u = self.cx + (a - self.mid[0]) * self.sx
        v = self.cy - (b - self.mid[1]) * self.sy
        rgb, alpha = _sample(self.img, u, v)
        edge = _smoothstep(_sample(self.inside, u, v)[1], 1.0, self.edge_px)
        px = self._px(np.column_stack([a, b]))
        seen = points @ self.cam >= self.near[px[:, 1], px[:, 0]] - self.tol
        facing = normals @ self.cam
        # a picture only describes surfaces it sees squarely; grazing ones would smear its pixels
        w = _smoothstep(facing, 0.3, 0.8) * facing ** 2 * (alpha / 255.0) * edge * seen * self.trust
        return rgb, w


def _back_from_front(front: Image.Image) -> Image.Image:
    """A stand-in back view for single pictures. The back usually matches the figure's sides more
    than its front (hair around the head, the shirt's colour, trousers…), so each row takes the
    colour found along the silhouette's edges at that height; details are never copied."""
    img = front.convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
    small = img.copy()
    small.thumbnail((256, 256))
    arr = np.asarray(small, dtype=np.uint8)
    mask = arr[..., 3] >= 128
    h, w = mask.shape
    depth = ndimage.distance_transform_edt(mask)
    ys = np.nonzero(mask.any(1))[0]
    if not len(ys):
        return img
    band = max(2.0, 0.06 * (np.ptp(np.nonzero(mask.any(0))[0]) + 1))
    rows = np.full((h, 3), np.nan)
    for y in ys:
        edge = mask[y] & (depth[y] <= band)
        rows[y] = np.median(arr[y, edge if edge.any() else mask[y], :3], axis=0)
    rows[ys] = ndimage.gaussian_filter1d(rows[ys], sigma=max(1.0, 0.015 * len(ys)), axis=0, mode="nearest")
    rgb = np.where(mask[..., None], np.nan_to_num(rows)[:, None, :], 0).astype(np.uint8)
    out = Image.fromarray(np.dstack([rgb, arr[..., 3]]), "RGBA").resize(img.size, Image.BILINEAR)
    out.putalpha(img.getchannel("A"))
    return out


def _prepare(mesh: trimesh.Trimesh, views: dict[str, Image.Image], mirror_side: bool, depth_res: int,
             trust: dict[str, float] | None = None, poses: dict[str, tuple[float, float]] | None = None) -> list[_View]:
    views = {k: v for k, v in views.items() if k in VIEWS and v is not None and _has_silhouette(v)}
    if not views:
        raise ProjectionError("Hacen falta vistas (frente, lateral o espalda) para texturizar")
    trust, poses = dict(trust or {}), dict(poses or {})
    if mirror_side:  # symmetric characters: the missing side is the mirror of the other one
        for a, b in (("left", "right"), ("right", "left")):
            if a in views and b not in views:
                views[b] = views[a].transpose(Image.FLIP_LEFT_RIGHT)
                if a in trust:
                    trust[b] = trust[a]
                if a in poses:  # mirrored camera: azimuth -az
                    poses[b] = ((360 - poses[a][0]) % 360, poses[a][1])
    if set(views) == {"front"}:
        views["back"], trust["back"] = _back_from_front(views["front"]), 0.5
    verts = np.asarray(mesh.vertices, dtype=float)
    size = float(np.ptp(verts, axis=0).max())
    return [_View(name, image, verts, size, depth_res, trust.get(name, 1.0), poses.get(name))
            for name, image in views.items()]


def _has_silhouette(image: Image.Image) -> bool:
    """A picture whose background removal left nothing cannot paint anything."""
    return (np.asarray(image.convert("RGBA"))[..., 3] > 128).mean() > 0.002


def _blend(prepared: list[_View], points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    color_sum = np.zeros((len(points), 3))
    weight_sum = np.zeros(len(points))
    for view in prepared:
        rgb, w = view.colour(points, normals)
        color_sum += rgb * w[:, None]
        weight_sum += w
    return color_sum, weight_sum


def _vertex_colours(mesh: trimesh.Trimesh, prepared: list[_View]) -> np.ndarray:
    color_sum, weight_sum = _blend(prepared, np.asarray(mesh.vertices, float), np.asarray(mesh.vertex_normals, float))
    known = weight_sum > 0.02
    colors = np.zeros((len(mesh.vertices), 3))
    colors[known] = color_sum[known] / weight_sum[known, None]
    return np.clip(_diffuse(mesh, colors, known), 0, 255)


def project_views(mesh: trimesh.Trimesh, views: dict[str, Image.Image], mirror_side: bool = True,
                  depth_res: int = 384, trust: dict[str, float] | None = None,
                  poses: dict[str, tuple[float, float]] | None = None) -> trimesh.Trimesh:
    """Vertex-coloured copy of `mesh` painted from its views."""
    prepared = _prepare(mesh, views, mirror_side, depth_res, trust, poses)
    mesh = mesh.copy()
    colors = _vertex_colours(mesh, prepared)
    rgba = np.column_stack([colors, np.full(len(colors), 255)]).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    return mesh


def bake_views(mesh: trimesh.Trimesh, views: dict[str, Image.Image], size: int = 2048, mirror_side: bool = True,
               depth_res: int = 512, trust: dict[str, float] | None = None,
               poses: dict[str, tuple[float, float]] | None = None) -> trimesh.Trimesh:
    """Textured copy of `mesh`: UV-unwrapped with xatlas and every texel painted from the views.
    `trust` weighs views by role (e.g. AI-generated ones below real pictures), `poses` gives
    (azimuth, elevation) for views not taken square on the four axes."""
    import xatlas

    prepared = _prepare(mesh, views, mirror_side, depth_res, trust, poses)
    verts = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    vcol = _vertex_colours(mesh, prepared)  # smooth fallback where no picture is trustworthy

    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, mesh.faces)
    pack = xatlas.PackOptions()
    pack.resolution, pack.padding, pack.bilinear = size, 4, True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmap, faces, uv = atlas[0]

    # which face covers each texel: draw the UV triangles with their index as the "colour"
    corners = np.column_stack([uv[:, 0] * size, (1 - uv[:, 1]) * size])  # image rows go down
    ids = Image.new("I", (size, size), 0)
    draw = ImageDraw.Draw(ids)
    for i, tri in enumerate(corners[faces]):
        draw.polygon([tuple(p) for p in tri], fill=i + 1)
    face_of = np.asarray(ids, dtype=np.int64) - 1
    ty, tx = np.nonzero(face_of >= 0)
    f = face_of[ty, tx]
    # barycentric coordinates of the texel centres
    p = np.column_stack([tx + 0.5, ty + 0.5])
    a, b, c = corners[faces[f, 0]], corners[faces[f, 1]], corners[faces[f, 2]]
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den[np.abs(den) < 1e-12] = 1e-12
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bary = np.clip(np.column_stack([1 - l1 - l2, l1, l2]), 0, 1)
    bary /= bary.sum(1, keepdims=True)
    src = vmap[faces[f]]  # original vertex of each corner
    points = np.einsum("nk,nkj->nj", bary, verts[src])
    nrm = np.einsum("nk,nkj->nj", bary, normals[src])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    fallback = np.einsum("nk,nkj->nj", bary, vcol[src])

    texels = np.zeros((size, size, 3))
    for chunk in np.array_split(np.arange(len(points)), max(1, len(points) // 500_000)):
        color_sum, weight_sum = _blend(prepared, points[chunk], nrm[chunk])
        prior = 0.05  # low-trust texels lean on the smooth vertex colours instead of noise
        texels[ty[chunk], tx[chunk]] = (color_sum + fallback[chunk] * prior) / (weight_sum + prior)[:, None]
    # fill the gutters between charts with the nearest painted texel (no dark seams when filtered)
    empty = face_of < 0
    if empty.any():
        _, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
        texels = texels[iy, ix]
    image = Image.fromarray(np.clip(texels, 0, 255).astype(np.uint8), "RGB")

    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.8)
    out = trimesh.Trimesh(verts[vmap], faces, vertex_normals=normals[vmap], process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    return out


ROLES = ["front", "left", "back", "right"]  # order of the app's multi-view slots


def paint_glb(glb: bytes, images: list[bytes], roles: list[str] | None = None,
              trust: dict[str, float] | None = None, poses: dict[str, tuple[float, float]] | None = None) -> bytes:
    """Paint an untextured model from its input pictures; textured models are returned as is.
    `trust` and `poses` are per role, see `bake_views`."""
    from .sheet import remove_background

    scene = load_scene(glb)
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes or any(getattr(m.visual, "kind", None) == "texture" for m in meshes):
        return glb
    views = {}
    for role, data in zip(roles or ROLES, images):
        views[role] = remove_background(Image.open(io.BytesIO(data)))
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    try:
        return mesh_to_glb(bake_views(mesh, views, trust=trust, poses=poses))
    except ImportError:  # xatlas missing: vertex colours still look right, just less detailed
        return mesh_to_glb(project_views(mesh, views, trust=trust, poses=poses))


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
