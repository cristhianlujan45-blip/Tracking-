"""Humanoid skeleton: template, automatic marker placement and joint layout.

Conventions (glTF): Y up, the character faces +Z, so its left side is +X. Joint names follow the
Mixamo convention, which Unity Humanoid, Unreal retargeting and Blender add-ons recognise.

Markers are the Mixamo-style handles a user can adjust (chin, shoulders, elbows, wrists, groin,
knees, ankles); every joint position is derived from them plus the mesh itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

# (name, parent) in topological order
JOINTS: list[tuple[str, str | None]] = [
    ("Hips", None),
    ("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
    ("Neck", "Spine2"), ("Head", "Neck"), ("HeadTop_End", "Head"),
]
for _side in ("Left", "Right"):
    JOINTS += [
        (f"{_side}Shoulder", "Spine2"), (f"{_side}Arm", f"{_side}Shoulder"), (f"{_side}ForeArm", f"{_side}Arm"),
        (f"{_side}Hand", f"{_side}ForeArm"), (f"{_side}Hand_End", f"{_side}Hand"),
    ]
for _side in ("Left", "Right"):
    JOINTS += [
        (f"{_side}UpLeg", "Hips"), (f"{_side}Leg", f"{_side}UpLeg"), (f"{_side}Foot", f"{_side}Leg"),
        (f"{_side}ToeBase", f"{_side}Foot"), (f"{_side}Toe_End", f"{_side}ToeBase"),
    ]

JOINT_NAMES = [name for name, _ in JOINTS]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
PARENTS = [JOINT_INDEX[p] if p else -1 for _, p in JOINTS]
DEFORM_JOINTS = [n for n in JOINT_NAMES if not n.endswith("_End")]

# Each deforming joint owns the segment towards this child (where its flesh is).
BONE_CHILD = {
    "Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck", "Neck": "Head", "Head": "HeadTop_End",
}
for _side in ("Left", "Right"):
    BONE_CHILD.update({
        f"{_side}Shoulder": f"{_side}Arm", f"{_side}Arm": f"{_side}ForeArm", f"{_side}ForeArm": f"{_side}Hand",
        f"{_side}Hand": f"{_side}Hand_End", f"{_side}UpLeg": f"{_side}Leg", f"{_side}Leg": f"{_side}Foot",
        f"{_side}Foot": f"{_side}ToeBase", f"{_side}ToeBase": f"{_side}Toe_End",
    })

MARKERS = ["chin", "groin", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
           "knee_l", "knee_r", "ankle_l", "ankle_r"]
MARKER_LABELS = {
    "chin": "Barbilla", "groin": "Ingle", "shoulder_l": "Hombro izq.", "shoulder_r": "Hombro der.",
    "elbow_l": "Codo izq.", "elbow_r": "Codo der.", "wrist_l": "Muñeca izq.", "wrist_r": "Muñeca der.",
    "knee_l": "Rodilla izq.", "knee_r": "Rodilla der.", "ankle_l": "Tobillo izq.", "ankle_r": "Tobillo der.",
}


class RigError(Exception):
    """User-facing rigging error."""


@dataclass
class Skeleton:
    positions: np.ndarray  # (J, 3) world rest positions, in JOINTS order

    def __getitem__(self, name: str) -> np.ndarray:
        return self.positions[JOINT_INDEX[name]]

    @property
    def height(self) -> float:
        return float(self["HeadTop_End"][1] - min(self["LeftToe_End"][1], self["RightToe_End"][1]))

    def bone_segments(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        names = DEFORM_JOINTS
        starts = np.array([self[n] for n in names])
        ends = np.array([self[BONE_CHILD[n]] for n in names])
        return names, starts, ends

    def as_dict(self) -> dict:
        return {name: [round(float(v), 5) for v in self.positions[i]] for i, name in enumerate(JOINT_NAMES)}


# ---------------------------------------------------------------- orientation


def yaw_matrix(degrees: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 1, 0])


def guess_yaw(vertices: np.ndarray) -> int:
    """A character in A/T pose is widest from hand to hand; turn it so that span runs along X."""
    ext = np.ptp(vertices, axis=0)
    return 90 if ext[2] > ext[0] * 1.15 else 0


# ---------------------------------------------------------------- markers


def _loops(mesh: trimesh.Trimesh, y: float) -> list[np.ndarray]:
    """Closed outlines of the horizontal cross-section at height y (one per arm, leg, torso…).

    Tiny outlines (internal cavities, loose bits) are noise and are dropped.
    """
    section = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if section is None:
        return []
    min_size = 0.015 * float(np.ptp(mesh.vertices[:, 1]))
    return [lp for lp in section.discrete if len(lp) >= 3 and np.ptp(lp[:, [0, 2]], axis=0).max() >= min_size]


def _central(loops: list[np.ndarray]) -> np.ndarray | None:
    """The outline that contains x=0 (the torso/neck), or the one closest to it."""
    if not loops:
        return None
    containing = [lp for lp in loops if lp[:, 0].min() <= 0 <= lp[:, 0].max()]
    pool = containing or loops
    return min(pool, key=lambda lp: abs(lp[:, 0].mean()))


def _width(loop: np.ndarray | None) -> float:
    return float(np.ptp(loop[:, 0])) if loop is not None else np.inf


def _fingertip(mesh: trimesh.Trimesh, sign: int, shoulder: np.ndarray, groin_y: float, neck_y: float,
               h: float, half_w: float) -> np.ndarray | None:
    """Farthest point of the arm from the shoulder, following the arm's own cross-sections.

    Above the groin every outline on this side apart from the torso is arm. Below it the innermost
    outline is the leg, so only outlines further out (a hanging hand) count.
    """
    above, below = [], []
    for y in np.arange(groin_y - 0.22 * h, neck_y, 0.01 * h):
        loops = _loops(mesh, y)
        torso = _central(loops) if y >= groin_y else None
        side = sorted((lp for lp in loops if sign * lp[:, 0].mean() > 0 and lp is not torso),
                      key=lambda lp: sign * lp[:, 0].mean())
        if y < groin_y:
            below += [lp[:, :2] for lp in side[1:]]  # drop the leg
        else:
            above += [lp[:, :2] for lp in side if sign * lp[:, 0].min() > 0.9 * half_w]  # outside the torso
    # hands below the groin only count when the arm was already seen hanging above it
    points = above + below if above else []
    if points:
        pts = np.vstack(points)
    else:  # T-pose: arms merge with the torso outline; the fingertip is the extreme in x
        v = mesh.vertices
        pts = v[v[:, 1] > groin_y][:, :2]
        if not len(pts):
            return None
        return pts[np.argmax(sign * pts[:, 0])]
    return pts[np.argmax(np.linalg.norm(pts - shoulder, axis=1))]


def estimate_markers(mesh: trimesh.Trimesh) -> dict[str, list[float]]:
    """Mixamo-style markers for a humanoid standing on y=min, facing +Z, centred on x=0."""
    v = mesh.vertices
    ground, top = float(v[:, 1].min()), float(v[:, 1].max())
    h = top - ground
    if h <= 0:
        raise RigError("El modelo no tiene altura")

    def y_at(frac: float) -> float:
        return ground + frac * h

    # neck: narrowest central cross-section between 76 % and 93 % of the height
    neck_y = min((y_at(f) for f in np.linspace(0.76, 0.93, 35)), key=lambda y: _width(_central(_loops(mesh, y))))
    chin_y = neck_y + 0.012 * h

    # groin: highest height (below 62 %) where no outline crosses the centre line, i.e. between the legs
    groin_y = None
    for frac in np.linspace(0.15, 0.62, 48):
        loops = _loops(mesh, y_at(frac))
        if loops and not any(lp[:, 0].min() <= 0 <= lp[:, 0].max() for lp in loops):
            groin_y = y_at(frac)
    groin_y = (groin_y if groin_y is not None else y_at(0.46)) + 0.015 * h

    # torso half-width: narrowest central outline between 60 % and 72 % (arms held close to the body
    # can merge with the torso outline at some heights, which only ever makes it wider)
    widths = []
    for frac in np.linspace(0.6, 0.72, 13):
        torso = _central(_loops(mesh, y_at(frac)))
        if torso is not None:
            widths.append(float(max(abs(torso[:, 0].min()), abs(torso[:, 0].max()))))
    half_w = min(widths) if widths else 0.12 * h
    shoulder_y = neck_y - 0.06 * h

    def leg_center(y: float, sign: int, default: float) -> float:
        side = [lp for lp in _loops(mesh, y) if sign * lp[:, 0].mean() > 0]
        if not side:
            return default
        leg = min(side, key=lambda lp: abs(lp[:, 0].mean()))  # innermost: an arm may hang further out
        return float(leg[:, 0].mean())

    markers: dict[str, list[float]] = {"chin": [0.0, chin_y], "groin": [0.0, groin_y]}
    for suffix, sign in (("l", 1), ("r", -1)):
        shoulder = np.array([sign * half_w * 1.0, shoulder_y])
        far = _fingertip(mesh, sign, shoulder, groin_y, neck_y, h, half_w)
        if far is None:
            far = shoulder + np.array([sign * 0.1 * h, -0.42 * h])
        # anthropometric ratios along shoulder→fingertip: elbow 43 %, wrist 79 %
        elbow = shoulder + 0.43 * (far - shoulder)
        wrist = shoulder + 0.79 * (far - shoulder)
        knee_y = ground + 0.57 * (groin_y - ground)
        ankle_y = ground + 0.055 * h
        markers[f"shoulder_{suffix}"] = [float(shoulder[0]), float(shoulder[1])]
        markers[f"elbow_{suffix}"] = [float(elbow[0]), float(elbow[1])]
        markers[f"wrist_{suffix}"] = [float(wrist[0]), float(wrist[1])]
        markers[f"knee_{suffix}"] = [leg_center(knee_y, sign, sign * 0.08 * h), float(knee_y)]
        markers[f"ankle_{suffix}"] = [leg_center(ankle_y + 0.03 * h, sign, sign * 0.08 * h), float(ankle_y)]
    return {k: [round(float(x), 5) for x in val] for k, val in markers.items()}


def mirror_markers(markers: dict[str, list[float]], source: str = "l") -> dict[str, list[float]]:
    """Copy one side onto the other (x negated); centre markers snap to x=0."""
    other = "r" if source == "l" else "l"
    out = dict(markers)
    for name in MARKERS:
        if name.endswith(f"_{source}"):
            x, y = markers[name][:2]
            out[name[:-1] + other] = [-x, y]
    out["chin"] = [0.0, markers["chin"][1]]
    out["groin"] = [0.0, markers["groin"][1]]
    return out


# ---------------------------------------------------------------- skeleton from markers


def _depth_at(v: np.ndarray, x: float, y: float, radius: float, fallback: float) -> float:
    """Middle of the body's thickness (z) around a front-view point."""
    near = v[(np.abs(v[:, 0] - x) < radius) & (np.abs(v[:, 1] - y) < radius)]
    return float((near[:, 2].min() + near[:, 2].max()) / 2) if len(near) else fallback


def build_skeleton(vertices: np.ndarray, markers: dict[str, list[float]]) -> Skeleton:
    missing = [m for m in MARKERS if m not in markers]
    if missing:
        raise RigError(f"Faltan marcadores: {', '.join(missing)}")
    v = vertices
    ground, top = float(v[:, 1].min()), float(v[:, 1].max())
    h = top - ground
    zc = float(np.median(v[:, 2]))
    r = 0.04 * h

    def p3(name: str, radius: float = r) -> np.ndarray:
        x, y = markers[name][:2]
        return np.array([x, y, _depth_at(v, x, y, radius, zc)])

    pos: dict[str, np.ndarray] = {}
    groin, chin = p3("groin", 0.06 * h), p3("chin", 0.05 * h)
    shoulder_y = (markers["shoulder_l"][1] + markers["shoulder_r"][1]) / 2
    neck_base_y = shoulder_y + 0.3 * (chin[1] - shoulder_y)
    hips = np.array([0.0, groin[1] + 0.02 * h, groin[2]])
    pos["Hips"] = hips
    for name, frac in (("Spine", 0.22), ("Spine1", 0.48), ("Spine2", 0.74)):
        y = hips[1] + frac * (neck_base_y - hips[1])
        pos[name] = np.array([0.0, y, _depth_at(v, 0.0, y, 0.05 * h, zc)])
    pos["Neck"] = np.array([0.0, neck_base_y, _depth_at(v, 0.0, neck_base_y, 0.04 * h, zc)])
    pos["Head"] = np.array([0.0, chin[1], chin[2]])
    pos["HeadTop_End"] = np.array([0.0, top, chin[2]])

    for suffix, side in (("l", "Left"), ("r", "Right")):
        shoulder, elbow, wrist = p3(f"shoulder_{suffix}"), p3(f"elbow_{suffix}"), p3(f"wrist_{suffix}")
        knee, ankle = p3(f"knee_{suffix}"), p3(f"ankle_{suffix}")
        pos[f"{side}Shoulder"] = np.array([shoulder[0] * 0.28, shoulder[1] - 0.005 * h, pos["Spine2"][2]])
        pos[f"{side}Arm"] = shoulder
        pos[f"{side}ForeArm"] = elbow
        pos[f"{side}Hand"] = wrist
        forearm = wrist - elbow
        pos[f"{side}Hand_End"] = wrist + forearm / max(np.linalg.norm(forearm), 1e-6) * 0.55 * np.linalg.norm(forearm)

        pos[f"{side}UpLeg"] = np.array([knee[0], hips[1] - 0.03 * h, hips[2]])
        pos[f"{side}Leg"] = knee
        pos[f"{side}Foot"] = ankle
        foot = v[(np.abs(v[:, 0] - ankle[0]) < 0.06 * h) & (v[:, 1] < ankle[1] + 0.02 * h)]
        toe_z = float(foot[:, 2].max()) if len(foot) else ankle[2] + 0.12 * h
        toe_y = ground + 0.02 * h
        pos[f"{side}ToeBase"] = np.array([ankle[0], toe_y, ankle[2] + 0.65 * (toe_z - ankle[2])])
        pos[f"{side}Toe_End"] = np.array([ankle[0], toe_y, toe_z])

    return Skeleton(np.array([pos[name] for name in JOINT_NAMES], dtype=float))
