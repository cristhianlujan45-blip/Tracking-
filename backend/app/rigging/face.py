"""Face rig: a jaw bone placed from the head geometry and procedural blendshapes (morph targets).

Head frame: the Head joint sits at the chin marker (bottom of the head), HeadTop_End at the top;
heights are fractions of that span. Blendshapes are vertex offsets in the bind pose:

* jawOpen: the lower face turned down about the jaw pivot.
* smile: mouth corners pulled up and back.
* blink_L / blink_R: only when the eyes can be found as dark spots on the front of the upper head
  (vertex colours or texture); each eye's region is squashed onto its lower lid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .skinning import _point_segment_distances

if TYPE_CHECKING:
    from .skeleton import Skeleton

MORPHS = ("jawOpen", "blink_L", "blink_R", "smile")
MOUTH, PIVOT, CHIN = 0.3, 0.42, 0.08  # heights in the head frame
JAW_OPEN_DEGREES = 20.0


def _smooth(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def head_mask(v: np.ndarray, sk: Skeleton) -> np.ndarray:
    """Vertices of the head: above the neck and closer to the head than to any other bone."""
    head, top = sk["Head"], sk["HeadTop_End"]
    hh = float(top[1] - head[1])
    out = np.zeros(len(v), dtype=bool)
    if hh <= 0:
        return out
    near_axis = np.linalg.norm(v[:, [0, 2]] - head[[0, 2]], axis=1) < 1.5 * hh
    cand = np.flatnonzero((v[:, 1] > head[1] - 0.2 * hh) & near_axis)
    names, starts, ends = sk.bone_segments()
    own = [names.index(n) for n in ("Head", "Neck", "Jaw") if n in names]
    for chunk in np.array_split(cand, max(1, len(cand) // 20000)):  # bounded memory on big meshes
        if len(chunk):
            nearest = _point_segment_distances(v[chunk], starts, ends).argmin(axis=1)
            out[chunk[np.isin(nearest, own)]] = True
    return out


def place_jaw(v: np.ndarray, sk: Skeleton) -> tuple[np.ndarray, np.ndarray] | None:
    """(pivot, chin tip) of the jaw from the front-lower part of the head, or None."""
    head, top = sk["Head"], sk["HeadTop_End"]
    hh = float(top[1] - head[1])
    if hh <= 1e-6:
        return None
    pts = v[head_mask(v, sk)]
    rel = (pts[:, 1] - head[1]) / hh
    lower, cranium = pts[(rel > 0) & (rel < 0.45)], pts[(rel > 0) & (rel < 0.7)]
    if len(lower) < 12 or len(cranium) < 20:
        return None
    front, back = float(np.percentile(lower[:, 2], 97)), float(np.percentile(cranium[:, 2], 3))
    depth = front - back
    if depth < 0.15 * hh:
        return None
    centre = (front + back) / 2
    pivot = np.array([0.0, head[1] + PIVOT * hh, centre - 0.1 * depth])
    chin_band = lower[(lower[:, 1] - head[1]) / hh < 0.3]
    chin_front = float(np.percentile(chin_band[:, 2], 95)) if len(chin_band) >= 4 else front
    chin = np.array([0.0, head[1] + CHIN * hh, chin_front - 0.22 * depth])
    if chin[2] <= pivot[2] + 0.05 * depth:
        return None
    return pivot, chin


@dataclass
class Face:
    """Landmarks measured once on the whole character, applied to every primitive."""

    sk: Skeleton
    hh: float  # head height
    depth: float  # head depth (z)
    half_width: float  # head half width at the mouth
    front: float  # z of the face surface at the mouth
    pivot: np.ndarray
    eyes: list[tuple[np.ndarray, float, float]] = field(default_factory=list)  # (centre, rx, ry)

    @property
    def names(self) -> list[str]:
        return [n for n in MORPHS if not n.startswith("blink") or self.eyes]

    def jaw_weight(self, v: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
        """0..1: how much each vertex belongs to the lower jaw."""
        head = head_mask(v, self.sk) if head is None else head
        y0 = self.sk["Head"][1]
        mouth = y0 + MOUTH * self.hh
        w = 1.0 - _smooth(v[:, 1], mouth - 0.05 * self.hh, mouth + 0.05 * self.hh)
        w = w * _smooth(v[:, 1], y0 - 0.15 * self.hh, y0 + 0.02 * self.hh)  # fade into the neck
        w = w * _smooth(v[:, 2], self.pivot[2] - 0.05 * self.depth, self.pivot[2] + 0.25 * self.depth)
        return np.where(head, w, 0.0)

    def targets(self, v: np.ndarray, normals: np.ndarray) -> dict[str, np.ndarray]:
        """(n, 3) offsets per blendshape for these vertices (zeros where the face is not)."""
        head = head_mask(v, self.sk)
        out: dict[str, np.ndarray] = {}

        turn = Rotation.from_rotvec([np.radians(JAW_OPEN_DEGREES), 0.0, 0.0])  # chin down and back
        rel = v - self.pivot
        out["jawOpen"] = (turn.apply(rel) - rel) * self.jaw_weight(v, head)[:, None]

        smile = np.zeros_like(v)
        mouth_y = self.sk["Head"][1] + MOUTH * self.hh
        facing = head & (normals[:, 2] > -0.2) & (v[:, 2] > self.pivot[2])
        r = 0.11 * self.hh
        for side in (1, -1):
            corner = np.array([side * 0.45 * self.half_width, mouth_y, self.front])
            g = np.exp(-np.sum((v - corner) ** 2, axis=1) / (2 * r * r)) * facing
            smile += g[:, None] * np.array([side * 0.012, 0.04, -0.02]) * self.hh
        out["smile"] = smile

        if self.eyes:
            blink = np.zeros_like(v)
            for centre, rx, ry in self.eyes:
                e = np.sqrt(((v[:, 0] - centre[0]) / (1.5 * rx)) ** 2 + ((v[:, 1] - centre[1]) / (1.7 * ry)) ** 2)
                fall = np.clip((1 - e) / 0.45, 0.0, 1.0) * head * (v[:, 2] > centre[2] - 0.3 * self.depth)
                lid = centre[1] - 0.35 * ry
                dy = (lid - v[:, 1]) * np.where(v[:, 1] > lid, 1.0, 0.25)  # upper lid comes down, lower rises
                blink[:, 1] += dy * fall
                blink[:, 2] += 0.1 * ry * fall
            left = _smooth(v[:, 0], -0.02 * self.hh, 0.02 * self.hh)[:, None]  # character's left is +X
            out["blink_L"], out["blink_R"] = blink * left, blink * (1 - left)
        return {n: out[n] for n in self.names}


def _find_eyes(v: np.ndarray, normals: np.ndarray, colors: np.ndarray, head: np.ndarray, sk: Skeleton,
               hh: float, pivot_z: float, depth: float) -> list[tuple[np.ndarray, float, float]]:
    """Dark compact spots on the front of the upper head, paired left/right (or one central eye)."""
    y0 = sk["Head"][1]
    rel = (v[:, 1] - y0) / hh
    region = head & (rel > 0.33) & (rel < 0.85) & (v[:, 2] > pivot_z + 0.15 * depth) & (normals[:, 2] > 0.25)
    if region.sum() < 30:
        return []
    rgb = colors[region, :3].astype(float) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    ref = float(np.median(lum))
    if ref < 0.12:  # the whole face is dark: eyes cannot be told apart
        return []
    dark = lum < min(0.55 * ref, ref - 0.18)
    if dark.sum() < 4 or dark.mean() > 0.2:
        return []
    face_pts = v[region]
    pts = face_pts[dark]
    spacing = float(np.median(cKDTree(face_pts).query(face_pts, k=2)[0][:, 1]))
    tree = cKDTree(pts)
    reach = max(0.05 * hh, 1.6 * spacing)  # neighbouring dark vertices belong to one spot
    n_comp, labels = connected_components(tree.sparse_distance_matrix(tree, reach), directed=False)
    spots = []
    for c in range(n_comp):
        q = pts[labels == c]
        ext = np.ptp(q, axis=0)
        if len(q) >= 2 and ext[0] < 0.45 * hh and ext[1] < 0.35 * hh:
            spots.append((len(q), q.mean(axis=0), max(ext[0] / 2, 0.04 * hh), max(ext[1] / 2, 0.035 * hh)))
    if not spots:
        return []
    best, best_score = None, 0
    for i, a in enumerate(spots):  # a symmetric pair at the same height
        for b in spots[i + 1:]:
            if a[1][0] * b[1][0] < 0 and abs(a[1][1] - b[1][1]) < 0.12 * hh \
                    and abs(a[1][0] + b[1][0]) < 0.25 * hh and a[0] + b[0] > best_score:
                best, best_score = [a, b], a[0] + b[0]
    if best is None:
        central = [s for s in spots if abs(s[1][0]) < 0.12 * hh]  # one big eye in the middle
        if not central:
            return []
        best = [max(central, key=lambda s: s[0])]
    return [(s[1], s[2], s[3]) for s in best]


def _measure(v: np.ndarray, sk: Skeleton, head: np.ndarray) -> Face | None:
    if "Jaw" not in sk.template.index:
        return None
    hh = float(sk["HeadTop_End"][1] - sk["Head"][1])
    pts = v[head]
    if len(pts) < 20 or hh <= 0:
        return None
    rel = (pts[:, 1] - sk["Head"][1]) / hh
    band, cranium = pts[np.abs(rel - MOUTH) < 0.1], pts[(rel > 0) & (rel < 0.7)]
    if len(band) < 6 or len(cranium) < 6:
        return None
    return Face(sk, hh, float(np.ptp(cranium[:, 2])) or hh, float(np.percentile(np.abs(band[:, 0]), 95)),
                float(np.percentile(band[:, 2], 95)), sk["Jaw"])


def jaw_region(v: np.ndarray, sk: Skeleton) -> np.ndarray:
    """0..1 per vertex: where the jaw bone may take skin weight."""
    head = head_mask(v, sk)
    face = _measure(v, sk, head)
    return face.jaw_weight(v, head) if face else np.zeros(len(v))


def analyse_face(v: np.ndarray, normals: np.ndarray, colors: np.ndarray | None, sk: Skeleton) -> Face | None:
    """Face landmarks for a skeleton with a jaw; None when it has none."""
    if "Jaw" not in sk.template.index:
        return None
    head = head_mask(v, sk)
    face = _measure(v, sk, head)
    if face is not None and colors is not None and len(colors) == len(v):
        face.eyes = _find_eyes(v, normals, colors, head, sk, face.hh, face.pivot[2], face.depth)
    return face
