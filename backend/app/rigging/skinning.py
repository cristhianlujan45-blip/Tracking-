"""Automatic skin weights by heat diffusion ("bone heat", Baran & Popović, Pinocchio, SIGGRAPH 2007).

Each bone acts as a heat source; heat spreads over the surface and the steady state gives smooth
weights that follow the shape of the body instead of raw distance:

    (L + M·H) w_b = M·H·p_b

L: cotangent Laplacian, M: lumped vertex areas, H: 1/d² to the nearest visible bone, p_b: 1 where
bone b is that bone. One sparse factorisation is shared by all bones. Large meshes are solved on a
decimated proxy and the weights carried back by nearest neighbour.

"Visible" means the straight path from the vertex to the bone stays inside the body (tested on a
filled voxel grid), so a big belly is not claimed by an arm hanging next to it, nor the arm by the
spine. Each bone also has an envelope: its reach is a few times the thickness of the flesh it owns,
so thin limbs do not claim a wide belly whose surface is as far from the spine as from them.
Fingers and the jaw only reach their own region (hand, lower face). A final smoothing pass evens
out the weights without adding influences beyond the 4 a vertex keeps.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .skeleton import RigError, Skeleton

MAX_PROXY_VERTICES = 15000
MAX_INFLUENCES = 4
VOXELS = 128  # visibility grid resolution along the longest side
CANDIDATES = 4  # nearest bones tested for visibility, per vertex
SAMPLES = 48  # points tested along each vertex → bone path
SMOOTH_ITERATIONS = 2
BEHIND = 1.0  # extra distance per unit a vertex lies behind a bone's root joint
ENVELOPE = (1.8, 3.0)  # a bone's reach fades between these multiples of its flesh thickness


def _point_segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                             behind: float = 0.0) -> np.ndarray:
    """(n, B) distances from every point to every bone segment. With `behind`, points beyond the
    bone's root joint (above the hip for a thigh, behind the shoulder for an arm) count as that
    much farther per unit: that flesh belongs to the parent bone."""
    seg = ends - starts  # (B, 3)
    seg_len2 = np.maximum((seg**2).sum(axis=1), 1e-12)
    rel = points[:, None, :] - starts[None, :, :]  # (n, B, 3)
    proj = (rel * seg[None]).sum(axis=2) / seg_len2[None]
    t = np.clip(proj, 0.0, 1.0)
    closest = starts[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    if behind:
        dist = dist + behind * np.maximum(-proj, 0.0) * np.sqrt(seg_len2)[None]
    return dist


def _cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Positive semi-definite cotangent Laplacian and lumped (1/3) vertex areas."""
    n = len(vertices)
    tri = vertices[faces]  # (f, 3, 3)
    weights_i, weights_j, weights_v = [], [], []
    for k in range(3):
        i, j, o = faces[:, (k + 1) % 3], faces[:, (k + 2) % 3], faces[:, k]
        a = vertices[i] - vertices[o]
        b = vertices[j] - vertices[o]
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        cot = (a * b).sum(axis=1) / np.maximum(cross, 1e-12)
        w = np.clip(0.5 * cot, 1e-8, 1e4)  # clamp obtuse (negative) cotangents for a stable M-matrix
        weights_i += [i, j]
        weights_j += [j, i]
        weights_v += [w, w]
    W = sp.coo_matrix((np.concatenate(weights_v), (np.concatenate(weights_i), np.concatenate(weights_j))),
                      shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
    face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.bincount(faces.ravel(), weights=np.repeat(face_area / 3.0, 3), minlength=n)
    return L.tocsr(), np.maximum(area, 1e-12)


def _closest_on_segments(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Closest point on segment i for point i (row-wise)."""
    seg = ends - starts
    t = np.clip(((points - starts) * seg).sum(axis=1) / np.maximum((seg**2).sum(axis=1), 1e-12), 0.0, 1.0)
    return starts + t[:, None] * seg


class _Interior:
    """Filled voxel grid of the body: is a point inside?"""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        self.pitch = float(mesh.extents.max()) / VOXELS
        grid = mesh.voxelized(self.pitch)
        try:
            grid = grid.fill()
        except Exception:  # noqa: BLE001 - open shells cannot be filled: only the surface counts
            pass
        self.matrix = np.asarray(grid.matrix, dtype=bool)
        transform = np.asarray(grid.transform, dtype=float)  # axis-aligned: scale + offset
        # one empty voxel of padding all round: points outside the grid land on it
        self.shape = np.array(self.matrix.shape) + 2
        self.flat = np.pad(self.matrix, 1).ravel()
        self.scale = (1.0 / np.diag(transform)[:3]).astype(np.float32)
        self.origin = (transform[:3, 3] - np.diag(transform)[:3]).astype(np.float32)

    def inside(self, points: np.ndarray) -> np.ndarray:
        q = np.rint((points - self.origin) * self.scale).astype(np.int32)
        np.clip(q, 0, (self.shape - 1).astype(np.int32), out=q)
        return self.flat[(q[..., 0] * self.shape[1] + q[..., 1]) * self.shape[2] + q[..., 2]]

    def visible(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """(n,) True where the straight path start → end stays inside (ignoring the voxels right at
        the start, on the surface, and at the end)."""
        start, seg = start.astype(np.float32), (end - start).astype(np.float32)
        length = np.linalg.norm(seg, axis=1)
        f = np.linspace(0.0, 1.0, SAMPLES, dtype=np.float32)[1:-1]
        pts = start[:, None, :] + f[None, :, None] * seg[:, None, :]
        along = f[None, :] * length[:, None]
        check = (along > 1.5 * self.pitch) & (length[:, None] - along > 0.5 * self.pitch)
        return ~np.any(check & ~self.inside(pts), axis=1)


def _interior(v: np.ndarray, faces: np.ndarray) -> _Interior | None:
    try:
        interior = _Interior(v, faces)
    except Exception:  # noqa: BLE001 - voxelisation failed: no visibility test
        return None
    return interior if interior.matrix.any() else None


def _sees(interior: _Interior, v: np.ndarray, s: np.ndarray, e: np.ndarray, rows: np.ndarray,
          bones: np.ndarray) -> np.ndarray:
    """Visibility of bone bones[i] from vertex rows[i], in chunks."""
    out = np.zeros(len(rows), dtype=bool)
    for chunk in np.array_split(np.arange(len(rows)), max(1, len(rows) // 4000)):
        if len(chunk):
            r, b = rows[chunk], bones[chunk]
            out[chunk] = interior.visible(v[r], _closest_on_segments(v[r], s[b], e[b]))
    return out


def _visible_nearest(interior: _Interior | None, v: np.ndarray, s: np.ndarray, e: np.ndarray,
                     dist: np.ndarray) -> np.ndarray:
    """Per vertex, the nearest bone it can see (falls back to the nearest one)."""
    order = np.argsort(dist, axis=1)[:, :min(CANDIDATES, dist.shape[1])]
    chosen = order[:, 0].copy()
    if interior is None:
        return chosen
    found = np.zeros(len(v), dtype=bool)
    for k in range(order.shape[1]):
        rows = np.flatnonzero(~found & np.isfinite(dist[np.arange(len(v)), order[:, k]]))
        vis = _sees(interior, v, s, e, rows, order[rows, k])
        chosen[rows[vis]] = order[rows[vis], k]
        found[rows[vis]] = True
    return chosen


def _drop_hidden(interior: _Interior, weights: np.ndarray, v: np.ndarray, s: np.ndarray, e: np.ndarray,
                 chosen: np.ndarray) -> np.ndarray:
    """No weight from a bone the vertex cannot see (heat that leaked around the body, e.g. from an
    arm into the belly next to it). The vertex's own bone always stays."""
    out = weights.copy()
    top = np.argsort(-weights, axis=1)[:, :2 * MAX_INFLUENCES]
    for k in range(top.shape[1]):
        bones = top[:, k]
        rows = np.flatnonzero((weights[np.arange(len(v)), bones] > 1e-3) & (bones != chosen))
        hidden = ~_sees(interior, v, s, e, rows, bones[rows])
        out[rows[hidden], bones[rows[hidden]]] = 0.0
    return out


def _envelope(dist: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """(n, B) 0..1: fades a bone out beyond a few times the median distance of the vertices it owns."""
    out = np.ones_like(dist)
    rows = np.arange(len(dist))
    for b in range(dist.shape[1]):
        own = dist[rows[chosen == b], b]
        own = own[np.isfinite(own)]
        if len(own) < 10:
            continue
        r = float(np.median(own))
        t = np.clip((dist[:, b] - ENVELOPE[0] * r) / ((ENVELOPE[1] - ENVELOPE[0]) * r + 1e-12), 0.0, 1.0)
        out[:, b] = 1.0 - t * t * (3 - 2 * t)
    return out


def bone_regions(vertices: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """(n, B) 0..1: where each bone may take weight (fingers: their hand; jaw: the lower face)."""
    names = skeleton.template.deform
    allowed = np.ones((len(vertices), len(names)))
    index = {n: i for i, n in enumerate(names)}
    if "LeftHandIndex1" in index:
        from .hands import hand_vertices

        for side in ("Left", "Right"):
            hand = np.zeros(len(vertices))
            hand[hand_vertices(vertices, skeleton, side)] = 1.0
            for n, i in index.items():
                if n.startswith(f"{side}Hand") and n != f"{side}Hand":
                    allowed[:, i] = hand
    if "Jaw" in index:
        from .face import jaw_region

        allowed[:, index["Jaw"]] = jaw_region(vertices, skeleton)
    return allowed


def _smooth(weights: np.ndarray, faces: np.ndarray, iterations: int = SMOOTH_ITERATIONS) -> np.ndarray:
    """Keep each vertex's top bones, average with the neighbours a few times, renormalise."""
    n = len(weights)
    top = np.argsort(-weights, axis=1)[:, :MAX_INFLUENCES]
    support = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(support, top, True, axis=1)
    support &= weights > 0
    w = np.where(support, weights, 0.0)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    i, j = np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0  # duplicate edges count once
    degree = np.asarray(A.sum(axis=1)).ravel()
    A = sp.diags(1.0 / np.maximum(degree, 1)) @ A
    lonely = degree == 0
    for _ in range(iterations):
        w = np.where(support, 0.5 * w + 0.5 * (A @ w), 0.0)
        w[lonely] = weights[lonely] * support[lonely]
    total = w.sum(axis=1, keepdims=True)
    return np.where(total > 1e-12, w / np.maximum(total, 1e-12), w)


def heat_weights(vertices: np.ndarray, faces: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """(n, B) skin weights for `vertices`, columns in `skeleton.bone_segments()` order."""
    _, starts, ends = skeleton.bone_segments()
    scale = float(np.ptp(vertices[:, 1])) or 1.0  # solve at unit height: resolution/scale independent
    v, s, e = vertices / scale, starts / scale, ends / scale

    regions = bone_regions(vertices, skeleton)
    raw = _point_segment_distances(v, s, e, behind=BEHIND)  # (n, B)
    rows = np.arange(len(v))
    interior = _interior(v, faces)
    chosen = _visible_nearest(interior, v, s, e, np.where(regions > 0.3, raw, np.inf))
    regions = regions * _envelope(np.where(regions > 0.3, raw, np.inf), chosen)
    dist = np.where(regions > 0.3, raw, np.inf)
    lost = ~np.isfinite(dist).any(axis=1)  # outside every envelope: keep the first choice
    dist[lost, chosen[lost]] = raw[lost, chosen[lost]]
    regions[lost, chosen[lost]] = 1.0
    chosen = _visible_nearest(interior, v, s, e, dist)
    nearest = dist[rows, chosen][:, None]
    # bones within 1 % of the chosen one's distance share the vertex (ties at joints)
    p = (np.abs(dist - nearest) <= nearest * 0.01 + 1e-9).astype(float)
    p[rows, chosen] = 1.0
    p /= p.sum(axis=1, keepdims=True)
    d = np.linalg.norm(v - _closest_on_segments(v, s[chosen], e[chosen]), axis=1)  # true distance
    H = 1.0 / np.maximum(d, 1e-3) ** 2

    L, area = _cotangent_laplacian(v, faces)
    MH = area * H
    solver = splu((L + sp.diags(MH)).tocsc())
    weights = np.column_stack([solver.solve(MH * p[:, b]) for b in range(p.shape[1])])
    weights = np.clip(weights, 0.0, None) * regions
    if interior is not None:
        weights = _drop_hidden(interior, weights, v, s, e, chosen)
    return _smooth(weights, faces)


def limit_influences(weights: np.ndarray, k: int = MAX_INFLUENCES) -> tuple[np.ndarray, np.ndarray]:
    """Top-k bones per vertex, renormalised: (indices uint16 (n, k), weights float32 (n, k))."""
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1)
    total = w.sum(axis=1, keepdims=True)
    fallback = total[:, 0] <= 1e-9
    w = np.where(total > 1e-9, w / np.maximum(total, 1e-12), 0.0)
    if fallback.any():  # isolated vertices: bind rigidly to the strongest bone
        w[fallback] = [1.0] + [0.0] * (k - 1)
    idx = np.where(w > 0, idx, 0)  # unused slots point at joint 0 (glTF validators expect it)
    return idx.astype(np.uint16), w.astype(np.float32)


def compute_skin(mesh: trimesh.Trimesh, skeleton: Skeleton) -> np.ndarray:
    """(n_vertices, B) weights for `mesh` (any topology: merges, decimates and transfers as needed)."""
    work = mesh.copy()
    work.merge_vertices()
    work.update_faces(work.nondegenerate_faces())
    work.remove_unreferenced_vertices()
    if len(work.faces) == 0:
        raise RigError("La malla no tiene caras válidas")
    if len(work.vertices) > MAX_PROXY_VERTICES:
        work = work.simplify_quadric_decimation(face_count=MAX_PROXY_VERTICES * 2)
    weights = heat_weights(np.asarray(work.vertices), np.asarray(work.faces), skeleton)
    if len(work.vertices) == len(mesh.vertices) and np.allclose(work.vertices, mesh.vertices):
        return weights
    _, nearest = cKDTree(work.vertices).query(mesh.vertices)
    return weights[nearest]
