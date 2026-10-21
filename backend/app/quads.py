"""Quad remeshing for animation and editing (Blender, Maya, ZBrush): clean all-quad topology that
keeps the model's texture.

1. Instant Meshes (field-aligned) turns the AI's dense triangle soup into evenly spaced quads that
   follow the shape, with few poles: it bends well when rigged and subdivides cleanly.
2. UV unwrap made of whole quads: charts grow over the quad mesh while their faces look roughly the
   same way, each one is flattened with LSCM (angle-preserving) and packed by its real outline.
   A UV seam never cuts a quad in two, so the OBJ / FBX stays 100 % quads.
3. The original model's colours (texture or vertex colours) are baked into the new UV texture from
   its surface, so the remeshed model looks the same with a fraction of the faces.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .geometry import GeometryError, _quiet_stdout, load_scene, mesh_to_glb, to_color_mesh

CHART_COS = 0.5  # faces of one UV chart stay within ~60° of its mean direction
MIN_CHART = 4  # smaller charts join a neighbour (fewer seams)
SAMPLES = 1_500_000  # surface samples of the original model used to bake the texture


@dataclass
class QuadMesh:
    vertices: np.ndarray  # (n, 3)
    faces: list[np.ndarray]  # quads (4 indices), and triangles only where the remesher left a pole
    uv: np.ndarray | None = None  # (m, 2) UV coordinates
    face_uv: list[np.ndarray] | None = None  # UV index per face corner
    texture: Image.Image | None = None

    @property
    def quads(self) -> int:
        return sum(len(f) == 4 for f in self.faces)


# ---------------------------------------------------------------- remesh


def _clean_source(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:  # specks and floaters would become stray little quad blobs
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= areas.sum() * 0.002]
        mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return mesh


def instant_quads(mesh: trimesh.Trimesh, target_faces: int) -> tuple[np.ndarray, list[np.ndarray]]:
    import pynanoinstantmeshes

    target = int(np.clip(target_faces, 200, 400_000))
    with _quiet_stdout():
        verts, faces = pynanoinstantmeshes.remesh(
            np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32), -1,
            rosy=4, posy=4, face_count=max(target // 4, 50),  # pure-quad output splits each face in 4
            creaseAngle=-1.0, align_to_boundaries=not mesh.is_watertight, smooth_iter=2, deterministic=True)
    verts, faces = np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    out = []
    for f in faces:  # a corner repeated twice is a triangle; anything smaller is dropped
        f = f[np.concatenate([[True], f[1:] != f[:-1]])]
        if len(f) > 1 and f[0] == f[-1]:
            f = f[:-1]
        if len(f) >= 3:
            out.append(f)
    if not out:
        raise GeometryError("No se pudo generar la malla de quads (¿modelo demasiado pequeño o roto?)")
    used = np.unique(np.concatenate(out))
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    verts, out = verts[used], [remap[f] for f in out]
    # same outward winding as the original (glTF and Blender both expect it)
    tris = _triangulate(out)
    normals = trimesh.Trimesh(verts, tris, process=False).face_normals
    _, near = cKDTree(mesh.triangles_center).query(verts[tris].mean(1))
    if (np.einsum("ij,ij->i", normals, mesh.face_normals[near]) < 0).mean() > 0.5:
        out = [f[::-1] for f in out]
    return verts, out


def _triangulate(faces: list[np.ndarray]) -> np.ndarray:
    return np.array([[f[0], f[i], f[i + 1]] for f in faces for i in range(1, len(f) - 1)], dtype=np.int64)


def _tri_owner(faces: list[np.ndarray]) -> np.ndarray:
    return np.array([n for n, f in enumerate(faces) for _ in range(len(f) - 2)], dtype=np.int64)


# ---------------------------------------------------------------- UV charts of whole quads


def _face_graph(faces: list[np.ndarray]) -> list[list[int]]:
    edges: dict[tuple[int, int], list[int]] = {}
    for n, f in enumerate(faces):
        for a, b in zip(f, np.roll(f, -1)):
            edges.setdefault((min(a, b), max(a, b)), []).append(n)
    neighbours: list[list[int]] = [[] for _ in faces]
    for owners in edges.values():
        if len(owners) == 2:
            a, b = owners
            neighbours[a].append(b)
            neighbours[b].append(a)
    return neighbours


def charts(verts: np.ndarray, faces: list[np.ndarray]) -> np.ndarray:
    """Chart id per face: region growing while faces stay within CHART_COS of the seed face's direction."""
    tris = _triangulate(faces)
    tri_normals = trimesh.Trimesh(verts, tris, process=False).face_normals
    tri_area = trimesh.Trimesh(verts, tris, process=False).area_faces
    owner = _tri_owner(faces)
    normal = np.zeros((len(faces), 3))
    np.add.at(normal, owner, tri_normals * tri_area[:, None])
    area = np.bincount(owner, tri_area, minlength=len(faces))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    graph = _face_graph(faces)
    chart = np.full(len(faces), -1)
    count = 0
    for seed in np.argsort(-area):
        if chart[seed] >= 0:
            continue
        chart[seed] = count
        # measured against the seed's direction, not a running mean: a mean drifts and lets a chart
        # wrap around an arm or a leg, which then folds over itself when flattened
        direction = normal[seed]
        frontier = list(graph[seed])
        while frontier:
            n = frontier.pop()
            if chart[n] >= 0 or normal[n] @ direction < CHART_COS:
                continue
            chart[n] = count
            frontier.extend(m for m in graph[n] if chart[m] < 0)
        count += 1
    # tiny charts join the neighbour they share most edges with
    sizes = np.bincount(chart)
    for n in np.argsort(sizes[chart]):
        if sizes[chart[n]] >= MIN_CHART:
            continue
        others = [chart[m] for m in graph[n] if chart[m] != chart[n]]
        if others:
            best = max(set(others), key=lambda c: (others.count(c), sizes[c]))
            sizes[chart[n]] -= 1
            sizes[best] += 1
            chart[n] = best
    return np.unique(chart, return_inverse=True)[1]


def lscm(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Least squares conformal map of one chart (vertices indexed 0..n-1), scaled to its 3D size."""
    n = len(verts)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    e1 = p1 - p0
    x_axis = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    normal = np.cross(e1, p2 - p0)
    area2 = np.linalg.norm(normal, axis=1)
    ok = area2 > 1e-14
    y_axis = np.cross(normal / np.maximum(area2[:, None], 1e-14), x_axis)
    # triangle corners in its own plane: (0, 0), (x1, 0), (x2, y2)
    x1 = np.linalg.norm(e1, axis=1)
    x2, y2 = np.einsum("ij,ij->i", p2 - p0, x_axis), np.einsum("ij,ij->i", p2 - p0, y_axis)
    w = 1 / np.sqrt(np.maximum(area2, 1e-14))
    # complex coefficients W_j = (x_k - x_l) + i (y_k - y_l) for corners (j, k, l) in cyclic order
    xs = np.stack([np.zeros_like(x1), x1, x2], 1)
    ys = np.stack([np.zeros_like(x1), np.zeros_like(x1), y2], 1)
    rows, cols, wr, wi = [], [], [], []
    tri_index = np.arange(len(tris))[ok]
    for j in range(3):
        k, m = (j + 1) % 3, (j + 2) % 3
        rows.append(tri_index)
        cols.append(tris[ok, j])
        wr.append(((xs[:, m] - xs[:, k]) * w)[ok])
        wi.append(((ys[:, m] - ys[:, k]) * w)[ok])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    wr, wi = np.concatenate(wr), np.concatenate(wi)
    t = len(tris)
    mr = sparse.csr_matrix((wr, (rows, cols)), shape=(t, n))
    mi = sparse.csr_matrix((wi, (rows, cols)), shape=(t, n))
    a = sparse.bmat([[mr, -mi], [mi, mr]]).tocsc()
    # pin the two vertices farthest apart along the chart's main direction
    centred = verts - verts.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    proj = centred @ axis
    pa, pb = int(np.argmin(proj)), int(np.argmax(proj))
    if pa == pb:
        return np.zeros((n, 2))
    length = np.linalg.norm(verts[pb] - verts[pa])
    pinned = np.array([pa, pb, pa + n, pb + n])
    values = np.array([0.0, length, 0.0, 0.0])
    free = np.setdiff1d(np.arange(2 * n), pinned)
    rhs = -a[:, pinned] @ values
    sol = np.zeros(2 * n)
    sol[pinned] = values
    af = a[:, free]
    sol[free] = spsolve((af.T @ af).tocsc(), af.T @ rhs)  # normal equations: a direct sparse solve
    uv = np.column_stack([sol[:n], sol[n:]])
    if _signed_areas(uv, tris).sum() < 0:  # mirrored solution
        uv[:, 0] = -uv[:, 0]
    return uv


def _signed_areas(uv: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2


def _flatten(verts: np.ndarray, faces: list[np.ndarray], chart_faces: list[int]) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    """LSCM of one chart; a chart that folds over itself is split in two along its long axis."""
    sub = [faces[i] for i in chart_faces]
    used = np.unique(np.concatenate(sub))
    local = {v: i for i, v in enumerate(used)}
    loc_faces = [np.array([local[v] for v in f]) for f in sub]
    tris = _triangulate(loc_faces)
    uv = lscm(verts[used], tris)
    areas = _signed_areas(uv, tris)
    area3d = trimesh.Trimesh(verts[used], tris, process=False).area
    flat = np.abs(areas).sum()
    if len(chart_faces) > 1 and ((areas <= 0).any() or flat < 1e-12 or not np.isfinite(uv).all()
                                 or _overlaps(uv, tris, flat)):
        centres = np.array([verts[f].mean(0) for f in sub])
        axis = np.linalg.svd(centres - centres.mean(0), full_matrices=False)[2][0]
        side = (centres - centres.mean(0)) @ axis > 0
        if side.all() or not side.any():
            side = np.arange(len(sub)) < len(sub) // 2
        halves = [[c for c, s in zip(chart_faces, side) if s == want] for want in (True, False)]
        return [piece for half in halves for piece in _flatten(verts, faces, half)]
    if flat > 1e-12:
        uv *= np.sqrt(area3d / flat)  # same texel density on every chart
    return [(chart_faces, used, uv)]


def _overlaps(uv: np.ndarray, tris: np.ndarray, flat: float, res: int = 1024) -> bool:
    """Whether parts of a flattened chart land on top of each other: the area its triangles cover
    together falls short of the sum of their areas."""
    if len(tris) < 8:
        return False
    span = max(float(np.ptp(uv, 0).max()), 1e-12)
    px = (uv - uv.min(0)) / span * (res - 1)
    img = Image.new("1", (res, res), 0)
    draw = ImageDraw.Draw(img)
    for tri in px[tris]:
        draw.polygon([tuple(p) for p in tri], fill=1)
    covered = np.count_nonzero(np.asarray(img))
    expected = flat / span**2 * (res - 1) ** 2
    return covered < 0.97 * expected


def _chart_mask(uv: np.ndarray, tris: np.ndarray, scale: float, cell: float, grow: int) -> np.ndarray:
    """Cells of the packing grid a chart covers at `scale` (texels per unit), grown by `grow` cells."""
    px = uv * scale / cell + grow
    shape = (int(np.ceil(px[:, 1].max())) + grow + 1, int(np.ceil(px[:, 0].max())) + grow + 1)
    img = Image.new("1", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for tri in px[tris]:
        draw.polygon([tuple(p) for p in tri], fill=1)
    mask = np.asarray(img, dtype=bool)
    if grow:
        mask = ndimage.binary_dilation(mask, iterations=grow)
    return mask


def _pack(pieces: list[tuple[np.ndarray, np.ndarray]], scale: float, size: int, cell: float, grow: int
          ) -> list[tuple[np.ndarray, int, int, bool]] | None:
    """Place each chart (largest first) where its real outline fits, lowest row first, upright or
    turned 90°. Returns (mask, row, col, turned) per chart, or None when they do not all fit."""
    from scipy.signal import fftconvolve

    grid = int(size / cell)
    occupied = np.zeros((grid, grid), dtype=np.float32)
    order = sorted(range(len(pieces)), key=lambda i: -np.ptp(pieces[i][0], 0).prod())
    placed: list = [None] * len(pieces)
    for i in order:
        uv, tris = pieces[i]
        best = None
        for turned in (False, True):
            coords = uv[:, ::-1] * [1, -1] if turned else uv  # 90° turn (a rotation, never a mirror)
            coords = coords - coords.min(0)
            mask = _chart_mask(coords, tris, scale, cell, grow)
            mh, mw = mask.shape
            if mh > grid or mw > grid:
                continue
            hits = fftconvolve(occupied, mask[::-1, ::-1].astype(np.float32), mode="valid")
            free = np.argwhere(hits < 0.5)
            if not len(free):
                continue
            row, col = free[np.lexsort((free[:, 1], free[:, 0]))[0]]
            if best is None or (row + mh, col) < (best[1] + best[0].shape[0], best[2]):
                best = (mask, int(row), int(col), turned)
        if best is None:
            return None
        mask, row, col, turned = best
        occupied[row:row + mask.shape[0], col:col + mask.shape[1]] += mask
        placed[i] = best
    return placed


def _layout(pieces: list[tuple[np.ndarray, np.ndarray]], size: int, padding: int) -> list[np.ndarray]:
    """UV coordinates (0..1) of every chart, packed by outline at the largest density that fits."""
    cell = max(2.0, size / 512)
    grow = int(np.ceil(padding / cell))
    area = sum(np.abs(_signed_areas(uv, tris)).sum() for uv, tris in pieces)
    scale = float(np.sqrt(0.72 * size * size / max(area, 1e-12)))
    placed, best = _pack(pieces, scale, size, cell, grow), None
    for _ in range(30):  # shrink until everything fits, then grow back while it still does
        if placed is not None:
            break
        scale *= 0.92
        placed = _pack(pieces, scale, size, cell, grow)
    if placed is None:
        raise GeometryError("No se pudo desplegar el modelo en un mapa UV")
    best = (scale, placed)
    for _ in range(4):
        bigger = _pack(pieces, scale * 1.04, size, cell, grow)
        if bigger is None:
            break
        scale *= 1.04
        best = (scale, bigger)
    scale, placed = best
    out = []
    for (uv, _), (_, row, col, turned) in zip(pieces, placed):
        coords = uv[:, ::-1] * [1, -1] if turned else uv
        coords = coords - coords.min(0)
        px = coords * scale + (np.array([col, row]) + grow) * cell
        out.append(px / size)
    return out


def _charts_flat(verts: np.ndarray, faces: list[np.ndarray]) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    chart = charts(verts, faces)
    pieces = []
    for c in range(chart.max() + 1):
        pieces += _flatten(verts, faces, list(np.nonzero(chart == c)[0]))
    return pieces


def unwrap(verts: np.ndarray, faces: list[np.ndarray], size: int = 2048, padding: int = 4
           ) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int]]:
    """UV coordinates (0..1), the UV index of every face corner, and the texture size to bake."""
    pieces = _charts_flat(verts, faces)
    local_tris = []
    for chart_faces, used, _ in pieces:
        local = {v: i for i, v in enumerate(used)}
        local_tris.append(_triangulate([np.array([local[v] for v in faces[f]]) for f in chart_faces]))
    layout = _layout([(uv, tris) for (_, _, uv), tris in zip(pieces, local_tris)], size, padding)
    uv_all, face_uv, offset = [], [None] * len(faces), 0
    for (chart_faces, used, _), uv in zip(pieces, layout):
        local = {v: i for i, v in enumerate(used)}
        for f in chart_faces:
            face_uv[f] = np.array([local[v] for v in faces[f]]) + offset
        uv_all.append(uv)
        offset += len(used)
    return np.vstack(uv_all), face_uv, (size, size)


def fast_unwrap(verts: np.ndarray, faces: np.ndarray, size: int = 2048, padding: int = 4
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV unwrap of a triangle mesh in xatlas's output format (vertex map, faces, uv), several times
    faster than xatlas on 100k-face models with a similar share of the texture used."""
    uv, face_uv, _ = unwrap(np.asarray(verts, dtype=float), list(np.asarray(faces)), size, padding)
    corner_faces = np.array(face_uv, dtype=np.int64)
    vmap = np.zeros(len(uv), dtype=np.int64)
    vmap[corner_faces.ravel()] = np.asarray(faces).ravel()
    return vmap, corner_faces, uv


# ---------------------------------------------------------------- texture bake


def _surface_samples(scene: trimesh.Scene, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, normals and sRGB colours densely covering the original model."""
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    total = sum(p.area for p in parts) or 1.0
    points, normals, colours = [], [], []
    for part in parts:
        n = max(1000, int(count * part.area / total))
        pts, face = trimesh.sample.sample_surface(part, n, seed=3)
        bary = trimesh.triangles.points_to_barycentric(part.triangles[face], pts)
        colours.append(_colour_at(part, face, bary))
        points.append(pts)
        normals.append(part.face_normals[face])
    return np.vstack(points), np.vstack(normals), np.vstack(colours)


def _colour_at(part: trimesh.Trimesh, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
    visual = part.visual
    try:
        if visual.kind == "texture" and visual.uv is not None:
            material = visual.material
            image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
            factor = np.asarray(getattr(material, "baseColorFactor", None) if getattr(
                material, "baseColorFactor", None) is not None else [255, 255, 255, 255], dtype=float)[:3]
            factor = factor / 255 if factor.max() > 1 else factor
            if image is not None:
                uv = np.einsum("nk,nkj->nj", bary, np.asarray(visual.uv)[part.faces[face]])
                img = np.asarray(image.convert("RGB"), dtype=float)
                h, w = img.shape[:2]
                u = np.mod(uv[:, 0], 1.0) * (w - 1)
                v = (1 - np.mod(uv[:, 1], 1.0)) * (h - 1)
                x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
                x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
                fx, fy = (u - x0)[:, None], (v - y0)[:, None]
                rgb = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
                       + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
                return rgb * factor
            return np.tile(factor * 255, (len(face), 1))
        colours = to_color_mesh(trimesh.Scene(part)).visual.vertex_colors[:, :3].astype(float)
        return np.einsum("nk,nkj->nj", bary, colours[part.faces[face]])
    except Exception:  # noqa: BLE001 - an unreadable material bakes grey, the shape is what matters
        return np.full((len(face), 3), 200.0)


def bake(qm: QuadMesh, scene: trimesh.Scene, size: tuple[int, int]) -> Image.Image:
    """Texture of the quad mesh painted from the original surface (nearest samples facing the same way)."""
    w, h = size
    tris = _triangulate(qm.faces)
    tri_uv = np.array([[fu[0], fu[i], fu[i + 1]] for fu in qm.face_uv for i in range(1, len(fu) - 1)])
    corners = np.column_stack([qm.uv[:, 0] * w, (1 - qm.uv[:, 1]) * h])
    ids = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(ids)
    for i, tri in enumerate(corners[tri_uv]):
        draw.polygon([tuple(p) for p in tri], fill=i + 1)
    face_of = np.asarray(ids, dtype=np.int64) - 1
    ty, tx = np.nonzero(face_of >= 0)
    f = face_of[ty, tx]
    a, b, c = corners[tri_uv[f, 0]], corners[tri_uv[f, 1]], corners[tri_uv[f, 2]]
    p = np.column_stack([tx + 0.5, ty + 0.5])
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den[np.abs(den) < 1e-12] = 1e-12
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bary = np.clip(np.column_stack([1 - l1 - l2, l1, l2]), 0, 1)
    bary /= bary.sum(1, keepdims=True)
    points = np.einsum("nk,nkj->nj", bary, qm.vertices[tris[f]])
    tri_normals = trimesh.Trimesh(qm.vertices, tris, process=False).face_normals
    normals = tri_normals[f]

    src_pts, src_nrm, src_col = _surface_samples(scene, SAMPLES)
    tree = cKDTree(src_pts)
    texels = np.zeros((h, w, 3))
    for chunk in np.array_split(np.arange(len(points)), max(1, len(points) // 400_000)):
        dist, near = tree.query(points[chunk], k=6, workers=-1)
        facing = np.einsum("nkj,nj->nk", src_nrm[near], normals[chunk])
        scale = np.maximum(dist[:, :1], 1e-9)
        weight = np.exp(-(dist / (scale * 2)) ** 2) * np.clip(facing, 0.05, 1) ** 2
        weight = np.where(facing > -0.2, weight, 1e-6)  # never the other side of a thin part
        colour = (src_col[near] * weight[..., None]).sum(1) / weight.sum(1, keepdims=True)
        texels[ty[chunk], tx[chunk]] = colour
    empty = face_of < 0
    if empty.any():  # gutters take the nearest painted texel: no dark seams once filtered
        _, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
        texels = texels[iy, ix]
    return Image.fromarray(np.clip(texels, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------- outputs


def to_trimesh(qm: QuadMesh) -> trimesh.Trimesh:
    """Triangulated copy for glTF / the viewer (each UV corner its own vertex)."""
    tri_uv = np.array([[fu[0], fu[i], fu[i + 1]] for fu in qm.face_uv for i in range(1, len(fu) - 1)])
    tris = _triangulate(qm.faces)
    corner_vertex = np.zeros(len(qm.uv), dtype=np.int64)
    corner_vertex[tri_uv.ravel()] = tris.ravel()
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=qm.texture, metallicFactor=0.0,
                                                   roughnessFactor=0.8)
    smooth = trimesh.Trimesh(qm.vertices, tris, process=False).vertex_normals
    return trimesh.Trimesh(qm.vertices[corner_vertex], tri_uv, vertex_normals=smooth[corner_vertex], process=False,
                           visual=trimesh.visual.TextureVisuals(uv=qm.uv, material=material))


def to_obj(qm: QuadMesh, name: str = "model") -> dict[str, bytes]:
    """OBJ that keeps the quads, with its material and texture: {"model.obj", "model.mtl", "model.png"}."""
    lines = ["# Forja3D quad remesh (Instant Meshes, UV charts of whole quads)", f"mtllib {name}.mtl",
             f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in qm.vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in qm.uv]
    lines += ["usemtl forja3d", "s 1"]
    lines += ["f " + " ".join(f"{a + 1}/{b + 1}" for a, b in zip(f, fu)) for f, fu in zip(qm.faces, qm.face_uv)]
    mtl = f"newmtl forja3d\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}.png\n"
    png = io.BytesIO()
    qm.texture.save(png, format="PNG", optimize=True)
    return {f"{name}.obj": ("\n".join(lines) + "\n").encode(), f"{name}.mtl": mtl.encode(), f"{name}.png": png.getvalue()}


def quad_remesh(glb: bytes, target_faces: int = 5000, texture: int = 2048,
                name: str = "model") -> tuple[bytes, dict[str, bytes], QuadMesh]:
    """(textured GLB for the viewer, OBJ + MTL + PNG with the real quads, the quad mesh)."""
    scene = load_scene(glb)
    source = _clean_source(to_color_mesh(scene))
    verts, faces = instant_quads(source, target_faces)
    qm = QuadMesh(verts, faces)
    qm.uv, qm.face_uv, size = unwrap(verts, faces, texture)
    qm.texture = bake(qm, scene, size)
    return mesh_to_glb(to_trimesh(qm)), to_obj(qm, name), qm
