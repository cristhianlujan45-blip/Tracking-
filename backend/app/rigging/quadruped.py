"""Quadruped rig (dogs, cats, horses, wolves…): template, side-view markers, skeleton and gaits.

Conventions: Y up, the animal faces +Z, its left side is +X. Markers are edited in a side view, so
they are (z, y) pairs; left and right legs share them and sit at the legs' measured width.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .animations import PRESET_SETS, Pose, Preset, _s
from .skeleton import RigError, Skeleton, Template

JOINTS: list[tuple[str, str | None]] = [
    ("Hips", None), ("Spine", "Hips"), ("Spine1", "Spine"), ("Chest", "Spine1"),
    ("Neck", "Chest"), ("Head", "Neck"), ("Nose_End", "Head"),
    ("Tail", "Hips"), ("Tail1", "Tail"), ("Tail_End", "Tail1"),
]
for _side in ("Left", "Right"):
    JOINTS += [(f"Front{_side}UpLeg", "Chest"), (f"Front{_side}Leg", f"Front{_side}UpLeg"),
               (f"Front{_side}Foot", f"Front{_side}Leg"), (f"Front{_side}Toe_End", f"Front{_side}Foot")]
    JOINTS += [(f"Back{_side}UpLeg", "Hips"), (f"Back{_side}Leg", f"Back{_side}UpLeg"),
               (f"Back{_side}Foot", f"Back{_side}Leg"), (f"Back{_side}Toe_End", f"Back{_side}Foot")]

BONE_CHILD = {"Hips": "Spine", "Spine": "Spine1", "Spine1": "Chest", "Chest": "Neck", "Neck": "Head",
              "Head": "Nose_End", "Tail": "Tail1", "Tail1": "Tail_End"}
for _side in ("Left", "Right"):
    for _end in ("Front", "Back"):
        BONE_CHILD.update({f"{_end}{_side}UpLeg": f"{_end}{_side}Leg", f"{_end}{_side}Leg": f"{_end}{_side}Foot",
                           f"{_end}{_side}Foot": f"{_end}{_side}Toe_End"})

MARKERS = ["nose", "head", "withers", "rump", "tail_tip", "front_shoulder", "front_elbow", "front_paw",
           "back_hip", "back_knee", "back_hock"]
MARKER_LABELS = {
    "nose": "Hocico", "head": "Nuca", "withers": "Cruz (lomo delantero)", "rump": "Grupa", "tail_tip": "Punta cola",
    "front_shoulder": "Hombro", "front_elbow": "Codo", "front_paw": "Muñeca delantera",
    "back_hip": "Cadera", "back_knee": "Rodilla", "back_hock": "Corvejón",
}
FEET = [f"{e}{s}{j}" for e in ("Front", "Back") for s in ("Left", "Right") for j in ("Foot", "Toe_End")]

QUADRUPED = Template("quadruped", JOINTS, BONE_CHILD, MARKERS, MARKER_LABELS, FEET, "side")


# ---------------------------------------------------------------- detection & orientation


def looks_quadruped(vertices: np.ndarray) -> bool:
    """Longer than tall and longer than wide: a body on four legs (after facing it along Z)."""
    ext = np.ptp(vertices, axis=0)
    length = max(ext[0], ext[2])
    return length > ext[1] * 0.95 and length > min(ext[0], ext[2]) * 1.4


def guess_yaw(vertices: np.ndarray) -> int:
    """Turn the body to run along Z, head (the higher end) towards +Z."""
    ext = np.ptp(vertices, axis=0)
    v = vertices
    if ext[0] > ext[2]:  # body along X: decide which X end is the head
        lo, hi = v[:, 0].min(), v[:, 0].max()
        front_hi = v[v[:, 0] > hi - 0.2 * ext[0], 1].max()
        front_lo = v[v[:, 0] < lo + 0.2 * ext[0], 1].max()
        return 270 if front_hi >= front_lo else 90  # +X head → turn -90° (=270) so it faces +Z
    lo, hi = v[:, 2].min(), v[:, 2].max()
    head_pos = v[v[:, 2] > hi - 0.2 * ext[2], 1].max()
    head_neg = v[v[:, 2] < lo + 0.2 * ext[2], 1].max()
    return 0 if head_pos >= head_neg else 180


# ---------------------------------------------------------------- markers


def _slab(v: np.ndarray, z: float, half: float, x_limit: float | None = None) -> np.ndarray:
    sel = np.abs(v[:, 2] - z) < half
    if x_limit is not None:
        sel &= np.abs(v[:, 0]) < x_limit
    return v[sel]


def _leg_columns(mesh: trimesh.Trimesh, h: float) -> tuple[float, float, float]:
    """z of the front and back legs and their x offset, from a low horizontal cross-section."""
    for frac in (0.15, 0.1, 0.22, 0.3):
        section = mesh.section(plane_origin=[0, frac * h, 0], plane_normal=[0, 1, 0])
        if section is None:
            continue
        loops = [lp for lp in section.discrete if len(lp) >= 3]
        if len(loops) >= 2:
            centers = np.array([lp.mean(axis=0) for lp in loops])
            mid = (centers[:, 2].max() + centers[:, 2].min()) / 2
            front = centers[centers[:, 2] >= mid]
            back = centers[centers[:, 2] < mid]
            if len(front) and len(back):
                return float(front[:, 2].mean()), float(back[:, 2].mean()), float(np.abs(centers[:, 0]).mean())
    zs = mesh.vertices[:, 2]
    length = float(np.ptp(zs))
    return float(zs.max() - 0.3 * length), float(zs.min() + 0.25 * length), float(0.1 * np.ptp(mesh.vertices[:, 0]))


def estimate_markers(mesh: trimesh.Trimesh) -> dict[str, list[float]]:
    v = mesh.vertices
    h = float(np.ptp(v[:, 1]))
    length = float(np.ptp(v[:, 2]))
    if h <= 0 or length <= 0:
        raise RigError("El modelo no tiene volumen")
    front_z, back_z, leg_x = _leg_columns(mesh, h)
    body_x = max(leg_x * 1.6, 0.05 * length)

    def spine_at(z: float) -> tuple[float, float]:
        """(back top, belly) of the trunk around z."""
        sl = _slab(v, z, 0.04 * length, body_x)
        sl = sl[sl[:, 1] > 0.3 * h] if len(sl[sl[:, 1] > 0.3 * h]) else sl
        if not len(sl):
            return 0.8 * h, 0.5 * h
        return float(sl[:, 1].max()), float(sl[:, 1].min())

    nose = v[np.argmax(v[:, 2])]
    tail = v[np.argmin(v[:, 2])]
    top_f, belly_f = spine_at(front_z)
    top_b, belly_b = spine_at(back_z)
    withers = [front_z, belly_f + 0.65 * (top_f - belly_f)]
    rump = [back_z, belly_b + 0.6 * (top_b - belly_b)]
    head = [withers[0] + 0.7 * (nose[2] - withers[0]), max(withers[1], nose[1]) + 0.05 * h]
    head_top = v[np.abs(v[:, 2] - head[0]) < 0.04 * length]
    if len(head_top):
        head[1] = float(head_top[:, 1].max() - 0.35 * (head_top[:, 1].max() - head_top[:, 1].min()))

    shoulder_y = belly_f + 0.35 * (top_f - belly_f)
    hip_y = belly_b + 0.4 * (top_b - belly_b)
    markers = {
        "nose": [float(nose[2]), float(nose[1])],
        "head": head,
        "withers": withers,
        "rump": rump,
        "tail_tip": [float(tail[2]), float(tail[1])],
        "front_shoulder": [front_z + 0.02 * length, shoulder_y],
        "front_elbow": [front_z - 0.01 * length, 0.55 * shoulder_y],
        "front_paw": [front_z, 0.07 * h],
        "back_hip": [back_z + 0.03 * length, hip_y],
        "back_knee": [back_z + 0.06 * length, 0.6 * hip_y],
        "back_hock": [back_z - 0.04 * length, 0.22 * hip_y],
        "_leg_x": [leg_x, 0.0],
    }
    return {k: [round(float(a), 5) for a in val] for k, val in markers.items()}


def build_skeleton(vertices: np.ndarray, markers: dict[str, list[float]]) -> Skeleton:
    missing = [m for m in MARKERS if m not in markers]
    if missing:
        raise RigError(f"Faltan marcadores: {', '.join(missing)}")
    v = vertices
    floor = float(v[:, 1].min())
    leg_x = float(markers.get("_leg_x", [0.1 * np.ptp(v[:, 0])])[0])

    def p(name: str, x: float = 0.0) -> np.ndarray:
        z, y = markers[name][:2]
        return np.array([x, y, z], dtype=float)

    pos: dict[str, np.ndarray] = {}
    hips, chest = p("rump"), p("withers")
    pos["Hips"] = hips
    pos["Spine"] = hips + (chest - hips) / 3
    pos["Spine1"] = hips + 2 * (chest - hips) / 3
    pos["Chest"] = chest
    pos["Head"] = p("head")
    pos["Neck"] = chest + 0.45 * (pos["Head"] - chest)
    pos["Nose_End"] = p("nose")
    tail_tip = p("tail_tip")
    pos["Tail"] = hips + 0.12 * (tail_tip - hips)
    pos["Tail1"] = pos["Tail"] + 0.5 * (tail_tip - pos["Tail"])
    pos["Tail_End"] = tail_tip
    for side, sign in (("Left", 1), ("Right", -1)):
        x = sign * leg_x
        paw = p("front_paw", x)
        pos[f"Front{side}UpLeg"] = p("front_shoulder", x)
        pos[f"Front{side}Leg"] = p("front_elbow", x)
        pos[f"Front{side}Foot"] = paw
        pos[f"Front{side}Toe_End"] = np.array([x, floor, paw[2] + 0.04 * np.ptp(v[:, 2])])
        hock = p("back_hock", x)
        pos[f"Back{side}UpLeg"] = p("back_hip", x)
        pos[f"Back{side}Leg"] = p("back_knee", x)
        pos[f"Back{side}Foot"] = hock
        pos[f"Back{side}Toe_End"] = np.array([x, floor, hock[2] + 0.06 * np.ptp(v[:, 2])])
    return Skeleton(np.array([pos[n] for n in QUADRUPED.names]), QUADRUPED)


def mirror_markers(markers: dict, source: str = "l") -> dict:
    return dict(markers)  # side-view markers are shared by both sides already


# ---------------------------------------------------------------- gaits

def _front(p: Pose, side: str, swing: float, lift: float) -> None:
    p.flex(f"Front{side}UpLeg", swing)
    p.flex(f"Front{side}Leg", -0.4 * lift)
    p.flex(f"Front{side}Foot", -1.1 * lift)


def _back(p: Pose, side: str, swing: float, lift: float) -> None:
    p.flex(f"Back{side}UpLeg", swing)
    p.flex(f"Back{side}Leg", -0.9 * lift)
    p.flex(f"Back{side}Foot", 0.9 * lift)


def _gait(p: Pose, t: float, phases: dict[str, float], swing: float, lift: float) -> None:
    for leg, phase in phases.items():
        s = _s(t, shift=phase)
        raise_ = lift * max(0.0, np.cos(2 * np.pi * (t + phase)))  # lift while the leg swings forward
        (_front if leg.startswith("F") else _back)(p, "Left" if leg.endswith("L") else "Right", swing * s, raise_)


def _tail(p: Pose, t: float, wag: float, k: float = 2.0) -> None:
    p.turn("Tail", wag * _s(t, k))
    p.turn("Tail1", 1.3 * wag * _s(t, k, -0.1))


def q_idle(p: Pose, t: float) -> None:
    p.lean("Spine1", 1.2 * _s(t))
    p.turn("Head", 12 * _s(t, 0.5))
    p.lean("Neck", -3 * _s(t))
    _tail(p, t, 8, 1.0)


def q_walk(p: Pose, t: float) -> None:
    # four-beat walk: LH, LF, RH, RF a quarter cycle apart
    _gait(p, t, {"BL": 0.0, "FL": 0.25, "BR": 0.5, "FR": 0.75}, 22, 45)
    p.turn("Spine", 3 * _s(t))
    p.lean("Neck", 4 * _s(t, 2.0))
    _tail(p, t, 10)


def q_trot(p: Pose, t: float) -> None:
    # diagonal pairs move together
    _gait(p, t, {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}, 30, 70)
    p.lean("Neck", 5 * _s(t, 2.0))
    _tail(p, t, 12, 2.0)
    p.offset[1] = 0.02 * p.sk.height * abs(_s(t))


def q_gallop(p: Pose, t: float) -> None:
    _gait(p, t, {"FL": 0.0, "FR": 0.08, "BL": 0.5, "BR": 0.58}, 42, 90)
    flex = 12 * _s(t)
    p.lean("Spine", flex)
    p.lean("Chest", -flex)
    p.lean("Neck", -6 * _s(t, 1.0, 0.25))
    p.lean("Tail", -15)
    p.offset[1] = 0.06 * p.sk.height * max(0.0, _s(t, 1.0, 0.25))


def q_sit(p: Pose, t: float) -> None:
    for side in ("Left", "Right"):
        p.flex(f"Back{side}UpLeg", 70)
        p.flex(f"Back{side}Leg", -120)
        p.flex(f"Back{side}Foot", 60)
    p.lean("Hips", -30)
    p.lean("Spine", -10)
    p.lean("Neck", -10)
    p.turn("Head", 10 * _s(t, 0.5))
    _tail(p, t, 6, 1.0)


def q_happy(p: Pose, t: float) -> None:
    _tail(p, t, 35, 4.0)
    p.tilt("Head", 12 * _s(t))
    p.lean("Neck", -8)
    for side in ("Left", "Right"):
        _front(p, side, 0, 0)
    p.offset[1] = 0.015 * p.sk.height * abs(_s(t, 2.0))


QUAD_PRESETS: dict[str, Preset] = {
    "q_idle": Preset("Reposo", 3.0, q_idle, tags=["básico"]),
    "q_walk": Preset("Paso", 1.2, q_walk, tags=["locomoción"]),
    "q_trot": Preset("Trote", 0.7, q_trot, tags=["locomoción"]),
    "q_gallop": Preset("Galope", 0.5, q_gallop, tags=["locomoción"], ground="clamp"),
    "q_sit": Preset("Sentado", 3.0, q_sit, tags=["pose"], ground="plant"),
    "q_happy": Preset("Contento (cola)", 1.5, q_happy, tags=["gesto"]),
}
DEFAULT_QUAD_PRESETS = ["q_idle", "q_walk", "q_trot", "q_gallop"]
PRESET_SETS["quadruped"] = QUAD_PRESETS
