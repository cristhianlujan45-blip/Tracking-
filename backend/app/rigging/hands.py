"""Finger chains (thumb + four fingers, three bones each) placed from the hand geometry.

The hand is every vertex past the wrist that belongs to this forearm. Its axis is the forearm's,
turned towards the fingertips; across it, the flat direction of the palm (PCA). Slices across the
hand are split where the mesh has gaps: where 4 (or 5) strips cross the tip region the fingers are
separate and each strip is followed down to the knuckles; a small strip at one edge below the
knuckles is the thumb. Mitten-like hands with no separated fingers still get four evenly spaced
chains, and a thumb on the side facing forward / towards the body.

Hand frame: s along the fingers (e1), u across the palm towards the thumb (e2), w = e1 × e2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from .skinning import _point_segment_distances

if TYPE_CHECKING:
    from .skeleton import Skeleton

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
PHALANX = (0.0, 0.45, 0.75, 1.0)  # joints along a finger, as a fraction of its length (base → tip)
DETAIL = 12  # vertex spacings across the palm needed to look for gaps (coarser hands are mittens)
Y, Z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
Strip = tuple[float, float, float]  # (s, low u, high u) of one finger in one slice


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def hand_vertices(v: np.ndarray, sk: Skeleton, side: str) -> np.ndarray:
    """Indices of the vertices of one hand: past the wrist and closer to this forearm/hand (or its
    fingers) than to any other bone (a thigh next to a hanging hand is left out)."""
    elbow, wrist = sk[f"{side}ForeArm"], sk[f"{side}Hand"]
    axis = wrist - elbow
    fl = float(np.linalg.norm(axis))
    if fl < 1e-6:
        return np.zeros(0, dtype=int)
    axis = axis / fl
    rel = v - wrist
    t = rel @ axis
    radial = np.linalg.norm(rel - t[:, None] * axis, axis=1)
    cand = np.flatnonzero((t > -0.02 * fl) & (t < 1.1 * fl) & (radial < 0.6 * fl))
    if not len(cand):
        return cand
    names, starts, ends = sk.bone_segments()
    ends = ends.copy()
    hand = names.index(f"{side}Hand")
    ends[hand] = wrist + 1.1 * fl * axis  # the whole hand, whatever the template's hand length
    own = [i for i, n in enumerate(names) if n.startswith(f"{side}Hand") or n == f"{side}ForeArm"]
    nearest = _point_segment_distances(v[cand], starts, ends).argmin(axis=1)
    return cand[np.isin(nearest, own)]


def _runs(values: np.ndarray, spacing: np.ndarray, gap: float, min_count: int) -> list[tuple[float, float, int]]:
    """(low, high, count) of the groups of values separated by a gap wider than `gap` and than
    the local vertex spacing on both sides (so a coarse palm is not split into strips)."""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    x, local = values[order], spacing[order]
    jump = np.diff(x) > np.maximum(gap, 1.2 * np.minimum(local[:-1], local[1:]))
    groups = np.split(x, np.flatnonzero(jump) + 1)
    return [(float(g[0]), float(g[-1]), len(g)) for g in groups if len(g) >= min_count]


class _Hand:
    """Hand points in the hand frame, cut into slices across the fingers."""

    def __init__(self, p: np.ndarray, wrist: np.ndarray, e1: np.ndarray, e2: np.ndarray, length: float):
        self.p, self.wrist, self.e1, self.length = p, wrist, e1, length
        self.nn = cKDTree(p).query(p, k=2)[0][:, 1]  # local vertex spacing
        spacing = self.spacing = float(np.median(self.nn))
        self.gap = 0.01 * length
        self.slab = max(0.025 * length, spacing)
        self.set_across(e2)

    def set_across(self, e2: np.ndarray) -> None:
        self.e2, self.e3 = e2, np.cross(self.e1, e2)
        rel = self.p - self.wrist
        self.s, self.u, self.w = rel @ self.e1, rel @ e2, rel @ self.e3
        self.slices: list[tuple[float, list[tuple[float, float, int]]]] = []
        detailed = np.ptp(np.percentile(self.u, [5, 95])) >= DETAIL * self.spacing
        for at in np.linspace(0.1 * self.length, 0.95 * self.length, 18):
            if not detailed:
                self.slices.append((float(at), []))
                continue
            sel = np.abs(self.s - at) < self.slab
            n = int(sel.sum())
            runs = _runs(self.u[sel], self.nn[sel], self.gap, max(3, int(0.03 * n))) if n >= 6 else []
            narrow = max(2 * self.spacing, 0.03 * self.length)  # a flat wall seen edge-on, not a finger
            self.slices.append((float(at), [r for r in runs if r[1] - r[0] >= narrow]))

    def world(self, s: float, u: float, w: float) -> np.ndarray:
        return self.wrist + s * self.e1 + u * self.e2 + w * self.e3

    def mask(self, strips: list[Strip], margin: float = 0.0) -> np.ndarray:
        sel = np.zeros(len(self.s), dtype=bool)
        for at, lo, hi in strips:
            sel |= (np.abs(self.s - at) < self.slab) & (self.u >= lo - margin) & (self.u <= hi + margin)
        return sel

    def centre(self, sel: np.ndarray, at: float, u_default: float) -> tuple[float, float]:
        """(u, w) centre of the points `sel` around distance `at` along the hand."""
        near = sel & (np.abs(self.s - at) < 2 * self.slab)
        if near.sum() < 2:
            near = sel
        if not near.any():
            return u_default, 0.0
        return float(np.mean(self.u[near])), float(np.mean(self.w[near]))

    def fingers_slice(self) -> tuple[int, int] | None:
        """(slice, strips) where 4 (or else 5) separate strips cross the outer part of the hand."""
        best = [(len(r), k) for k, (at, r) in enumerate(self.slices) if at >= 0.4 * self.length and len(r) in (4, 5)]
        if not best:
            return None
        count = 4 if any(c == 4 for c, _ in best) else 5
        ks = [k for c, k in best if c == count]
        return ks[len(ks) // 2], count

    def track(self, k0: int, run: tuple[float, float, int], pitch: float) -> list[Strip]:
        """Follow one strip up and down the slices while it stays separate."""
        width = run[1] - run[0]
        out = [(self.slices[k0][0], run[0], run[1])]
        for step in (1, -1):
            c, k = (run[0] + run[1]) / 2, k0 + step
            while 0 <= k < len(self.slices):
                at, runs = self.slices[k]
                if not runs:
                    break
                lo, hi, _ = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - c))
                if abs((lo + hi) / 2 - c) > 0.6 * pitch or hi - lo > 1.8 * width + self.gap:
                    break  # merged with the palm or a neighbour
                out.append((at, lo, hi))
                c, k = (lo + hi) / 2, k + step
        return sorted(out)

    def thumb_strips(self, below: float) -> tuple[int, list[Strip]]:
        """Below the knuckles, a small separate strip at one edge is the thumb: (side +1 / -1 / 0,
        its strips)."""
        high, low = [], []
        for at, runs in self.slices:
            if at >= below or len(runs) < 2:
                continue
            span = runs[-1][1] - runs[0][0]
            first, last = runs[0][1] - runs[0][0], runs[-1][1] - runs[-1][0]
            if last <= 0.45 * span and last <= first:
                high.append((at, runs[-1][0], runs[-1][1]))
            elif first <= 0.45 * span:
                low.append((at, runs[0][0], runs[0][1]))
        if len(high) >= 3 and len(high) >= len(low):
            return 1, high
        return (-1, low) if len(low) >= 3 else (0, [])


def place_fingers(v: np.ndarray, sk: Skeleton, side: str) -> dict[str, np.ndarray] | None:
    """World positions of the finger joints of one hand, or None when the hand is too small or
    degenerate to place them."""
    idx = hand_vertices(v, sk, side)
    if len(idx) < 24:
        return None
    p = v[idx]
    sign = 1.0 if side == "Left" else -1.0
    wrist = sk[f"{side}Hand"]
    fore = _unit(wrist - sk[f"{side}ForeArm"])
    fl = float(np.linalg.norm(wrist - sk[f"{side}ForeArm"]))

    # main axis: the forearm, turned towards the fingertips (a thumb or wide palm would tilt a PCA)
    e1 = fore
    for _ in range(2):
        s = (p - wrist) @ e1
        far = p[s > 0.7 * np.percentile(s, 99)]
        d = _unit(far.mean(axis=0) - wrist) if len(far) >= 3 else e1
        if d @ fore < np.cos(np.radians(40)):
            break
        e1 = d
    q = p - p.mean(axis=0)
    flat = q - (q @ e1)[:, None] * e1
    evals, evecs = np.linalg.eigh(np.cov(flat.T))
    if evals[2] > 1.5 * evals[1]:  # a flat palm: its width is the across direction
        e2 = _unit(evecs[:, 2] - (evecs[:, 2] @ e1) * e1)
    else:  # mitten / round hand: the thumb goes forward
        e2 = _unit(Z - (Z @ e1) * e1)
        if not e2.any():
            e2 = _unit(np.cross(e1, Y))
    if not e2.any():
        return None
    if e2 @ Z + 0.3 * (e2 @ (-sign * np.array([1.0, 0, 0]))) < 0:  # thumb forward / towards the body
        e2 = -e2

    length = float(np.percentile((p - wrist) @ e1, 99))
    if length < 0.15 * fl:
        return None
    hand = _Hand(p, wrist, e1, e2, length)
    spacing = hand.spacing

    # which edge is the thumb: a fifth strip that ends early, or a separate strip below the knuckles
    found = hand.fingers_slice()
    if found and found[1] == 5:
        runs = hand.slices[found[0]][1]
        pitch = (runs[-1][1] - runs[0][0]) / 5
        first, last = (hand.track(found[0], r, pitch)[-1][0] for r in (runs[0], runs[-1]))
        flip = first < last
    else:
        flip = hand.thumb_strips(hand.slices[found[0]][0] if found else 0.65 * length)[0] < 0
    if flip:
        hand.set_across(-e2)
        found = hand.fingers_slice()

    out: dict[str, np.ndarray] = {}
    thumb: list[Strip] = []
    if found:  # separated fingers
        k0, count = found
        runs = hand.slices[k0][1]
        pitch = (runs[-1][1] - runs[0][0]) / count
        tracks = [hand.track(k0, r, pitch) for r in runs]
        if count == 5:
            thumb, tracks = tracks[-1], tracks[:-1]
        base = float(np.clip(np.median([t[0][0] for t in tracks]) - 0.06 * length, 0.3 * length, 0.8 * length))
        strips = [hand.mask(t, hand.gap) | ((hand.s > t[-1][0]) & (hand.u >= t[-1][1] - hand.gap)
                                            & (hand.u <= t[-1][2] + hand.gap)) for t in reversed(tracks)]
        # each finger's centre line across the palm, fitted through its strips (u = a·s + b)
        lines = [np.polyfit([x[0] for x in t], [(x[1] + x[2]) / 2 for x in t], 1 if len(t) > 1 else 0)
                 for t in reversed(tracks)]
    else:  # mitten: four even strips across the tip
        base = 0.52 * length
        band = (hand.s > 0.45 * length) & (hand.s < 0.85 * length)
        lo, hi = np.percentile(hand.u[band] if band.sum() >= 8 else hand.u, [4, 96])
        step = (hi - lo) / 4
        strips = [np.abs(hand.u - (hi - (k + 0.5) * step)) < max(step / 2, spacing) for k in range(4)]
        lines = [None] * 4
    for name, sel, line in zip(FINGERS[1:], strips, lines):  # index (thumb side) first
        sel = sel & (hand.s > base - hand.slab)
        if sel.sum() < 3:
            return None
        end = float(np.percentile(hand.s[sel], 98))
        if end - base < 0.12 * length:
            return None
        cu = float(np.median(hand.u[sel]))
        for i, f in enumerate(PHALANX):
            at = base + f * (end - base)
            cu_at, cw = hand.centre(sel, at, cu)
            if line is not None:  # the palm around the knuckles would pull a plain average
                cu_at = float(np.polyval(line, at))
            out[f"{side}Hand{name}{i + 1}" if i < 3 else f"{side}Hand{name}_End"] = hand.world(at, cu_at, cw)

    # thumb: from near the wrist out to its strip, or along the palm edge on the thumb side
    if not thumb:
        thumb = hand.thumb_strips(base)[1] if hand.thumb_strips(base)[0] > 0 else []
    palm = (hand.s > 0.1 * length) & (hand.s < 0.6 * length)
    edge = float(np.percentile(hand.u[palm] if palm.sum() >= 8 else hand.u, 96))
    if edge <= 0:
        return None
    if thumb:
        sel = hand.mask(thumb, hand.gap) | ((hand.s > thumb[-1][0]) & (hand.u >= thumb[-1][1] - hand.gap))
        t_end = float(np.percentile(hand.s[sel], 98))
        if t_end > 0.8 * length:  # reaches the fingertips: an edge of the fingers, not a thumb
            thumb = []
        else:
            u_end = float(np.mean(hand.u[sel & (hand.s > t_end - 2 * hand.slab)]))
            u_base = 0.5 * float(np.mean(hand.u[hand.mask(thumb[:1])]))
    if not thumb:
        sel = palm & (hand.u > 0.3 * edge)
        t_end, u_end, u_base = 0.55 * length, 0.8 * edge, 0.35 * edge
    t_base = 0.12 * length
    if t_end - t_base < 0.1 * length:
        return None
    radius = max(0.1 * length, 2 * spacing)
    for i, f in enumerate(PHALANX):
        at, uu = t_base + f * (t_end - t_base), u_base + f * (u_end - u_base)
        near = (np.abs(hand.s - at) < radius) & (np.abs(hand.u - uu) < radius)
        if i and (near & sel).sum() >= 3:  # the joints past the palm follow the thumb's own points
            near = near & sel
            uu = float(np.mean(hand.u[near]))
        cw = float(np.mean(hand.w[near])) if near.sum() >= 2 else 0.0
        out[f"{side}HandThumb{i + 1}" if i < 3 else f"{side}HandThumb_End"] = hand.world(at, uu, cw)

    if not all(np.isfinite(x).all() for x in out.values()):
        return None
    return out
