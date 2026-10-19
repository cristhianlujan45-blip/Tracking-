"""Automatic skin weights by heat diffusion ("bone heat", Baran & Popović, Pinocchio, SIGGRAPH 2007).

Each bone acts as a heat source; heat spreads over the surface and the steady state gives smooth
weights that follow the shape of the body instead of raw distance:

    (L + M·H) w_b = M·H·p_b

L: cotangent Laplacian, M: lumped vertex areas, H: 1/d² to the nearest bone, p_b: 1 where bone b
is the nearest. One sparse factorisation is shared by all bones. Large meshes are solved on a
decimated proxy and the weights carried back by nearest neighbour.
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


def _point_segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """(n, B) distances from every point to every bone segment."""
    seg = ends - starts  # (B, 3)
    seg_len2 = np.maximum((seg**2).sum(axis=1), 1e-12)
    rel = points[:, None, :] - starts[None, :, :]  # (n, B, 3)
    t = np.clip((rel * seg[None]).sum(axis=2) / seg_len2[None], 0.0, 1.0)
    closest = starts[None] + t[..., None] * seg[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


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


def heat_weights(vertices: np.ndarray, faces: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """(n, B) skin weights for `vertices`, columns in `skeleton.bone_segments()` order."""
    _, starts, ends = skeleton.bone_segments()
    scale = float(np.ptp(vertices[:, 1])) or 1.0  # solve at unit height: resolution/scale independent
    v, s, e = vertices / scale, starts / scale, ends / scale

    dist = _point_segment_distances(v, s, e)  # (n, B)
    nearest = dist.min(axis=1, keepdims=True)
    # bones within 1 % of the nearest distance share the vertex (ties at joints)
    p = (dist <= nearest * 1.01 + 1e-9).astype(float)
    p /= p.sum(axis=1, keepdims=True)
    H = 1.0 / np.maximum(nearest.ravel(), 1e-3) ** 2

    L, area = _cotangent_laplacian(v, faces)
    MH = area * H
    solver = splu((L + sp.diags(MH)).tocsc())
    weights = np.column_stack([solver.solve(MH * p[:, b]) for b in range(p.shape[1])])
    return np.clip(weights, 0.0, None)


def limit_influences(weights: np.ndarray, k: int = MAX_INFLUENCES) -> tuple[np.ndarray, np.ndarray]:
    """Top-k bones per vertex, renormalised: (indices uint16 (n, k), weights float32 (n, k))."""
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1)
    total = w.sum(axis=1, keepdims=True)
    fallback = total[:, 0] <= 1e-9
    w = np.where(total > 1e-9, w / np.maximum(total, 1e-12), 0.0)
    if fallback.any():  # isolated vertices: bind rigidly to the strongest bone
        w[fallback] = [1.0] + [0.0] * (k - 1)
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
