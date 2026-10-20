"""Animation clips for the humanoid skeleton.

* A built-in library of hand-keyed motions (idle, walk, run, jump, wave, dance…), defined in terms
  of body mechanics (hip flexion, knee bend, arm elevation…) so they adapt to each character's
  proportions and rest pose (A-pose or T-pose). Hands take named shapes (relaxed, open, fist,
  point) when the skeleton has fingers; face clips drive blendshape weights.
* BVH motion-capture import with retargeting (CMU, Mixamo-converted, Rokoko, etc.).

Rotations are local, relative to the parent joint, with the rest pose as identity, which is how
the glTF writer stores the skeleton.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import FINGERS, HUMANOID, JOINT_INDEX, RigError, Skeleton

FPS = 30
Z = np.array([0.0, 0.0, 1.0])


@dataclass
class Clip:
    name: str
    times: np.ndarray  # (k,)
    rotations: np.ndarray  # (k, J, 4) local quaternions xyzw
    hips_offset: np.ndarray  # (k, 3) translation of Hips relative to its rest position
    loop: bool = True
    morphs: dict[str, np.ndarray] | None = None  # blendshape name → (k,) weight


# ---------------------------------------------------------------- posing helpers


# finger curl per joint (1, 2, 3) in degrees, plus spread of the base joint (towards the thumb +)
HandShape = dict[str, tuple[float, float, float, float]]
HAND_SHAPES: dict[str, HandShape] = {
    "relaxed": {"Thumb": (5, 10, 8, 0), "Index": (12, 16, 10, 2), "Middle": (15, 20, 12, 0),
                "Ring": (18, 22, 14, -2), "Pinky": (20, 25, 15, -4)},
    "open": {"Thumb": (0, 0, 0, 12), "Index": (0, 0, 0, 8), "Middle": (0, 0, 0, 0),
             "Ring": (0, 0, 0, -7), "Pinky": (0, 0, 0, -14)},
    "fist": {"Thumb": (20, 35, 40, -5), "Index": (80, 95, 60, 0), "Middle": (85, 100, 65, 0),
             "Ring": (88, 100, 65, 0), "Pinky": (90, 100, 65, 0)},
}
HAND_SHAPES["point"] = {**HAND_SHAPES["fist"], "Index": (0, 0, 0, 0), "Thumb": (25, 30, 35, -8)}


class Pose:
    """Local joint rotations built from anatomical motions, for one character."""

    def __init__(self, skeleton: Skeleton):
        self.sk = skeleton
        self.rot: dict[str, Rotation] = {}
        self.offset = np.zeros(3)
        self.hands: dict[str, HandShape] = {}  # side → shape; unset hands are relaxed
        self.morph: dict[str, float] = {}  # blendshape weights 0..1

    # every new motion is applied after the ones already set on that joint
    def _apply(self, joint: str, r: Rotation) -> None:
        self.rot[joint] = r * self.rot.get(joint, Rotation.identity())

    def rest_dir(self, joint: str) -> np.ndarray:
        d = self.sk[self.sk.template.bone_child[joint]] - self.sk[joint]
        return d / max(np.linalg.norm(d), 1e-9)

    def about(self, joint: str, axis, degrees: float) -> None:
        axis = np.asarray(axis, float)
        self._apply(joint, Rotation.from_rotvec(np.radians(degrees) * axis / np.linalg.norm(axis)))

    def flex(self, joint: str, degrees: float, direction: np.ndarray | None = None) -> None:
        """Swing a limb forward (+) / backward (−) in its own plane of motion."""
        d = self.rest_dir(joint) if direction is None else direction
        axis = np.cross(d, Z)
        if np.linalg.norm(axis) < 1e-6:  # bone already points forward (toes)
            axis = np.array([-1.0, 0.0, 0.0])
        self.about(joint, axis, degrees)

    # --- trunk
    def lean(self, joint: str, degrees: float) -> None:  # forward +
        self.about(joint, [1, 0, 0], degrees)

    def turn(self, joint: str, degrees: float) -> None:  # towards the character's left +
        self.about(joint, [0, 1, 0], degrees)

    def tilt(self, joint: str, degrees: float) -> None:  # towards the character's left +
        self.about(joint, [0, 0, -1], degrees)

    # --- limbs
    def rest_arm_elevation(self, side: str) -> float:
        d = self.rest_dir(f"{side}Arm")
        return float(np.degrees(np.arctan2(abs(d[0]), -d[1])))

    def arm(self, side: str, elevation: float, forward: float = 0.0, elbow: float = 0.0) -> None:
        """elevation: 0 = hanging down, 90 = horizontal (T-pose), 180 = straight up."""
        sign = 1 if side == "Left" else -1
        raise_deg = elevation - self.rest_arm_elevation(side)
        raise_rot = Rotation.from_rotvec(np.radians(sign * raise_deg) * Z)
        self._apply(f"{side}Arm", raise_rot)
        if forward:
            self.flex(f"{side}Arm", forward, direction=raise_rot.apply(self.rest_dir(f"{side}Arm")))
        if elbow:
            self.flex(f"{side}ForeArm", elbow)

    def leg(self, side: str, hip: float = 0.0, knee: float = 0.0, ankle: float = 0.0, spread: float = 0.0) -> None:
        """hip: thigh forward +; knee: bend (+, shin goes back); ankle: toes up +; spread: outwards +."""
        if spread:
            self.about(f"{side}UpLeg", Z, spread if side == "Left" else -spread)
        if hip:
            self.flex(f"{side}UpLeg", hip)
        if knee:
            self.flex(f"{side}Leg", -knee)
        if ankle:
            self.about(f"{side}Foot", [1, 0, 0], -ankle)

    # --- hands (no-ops without finger bones)
    def hand(self, side: str, shape: str, amount: float = 1.0, base: str = "relaxed") -> None:
        """Blend a hand from `base` to `shape` (0 → base, 1 → shape)."""
        a, b = HAND_SHAPES[base], HAND_SHAPES[shape]
        self.hands[side] = {f: tuple((1 - amount) * x + amount * y for x, y in zip(a[f], b[f])) for f in FINGERS}

    def _palm(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(palm normal, across-the-palm direction towards the thumb) in the rest pose."""
        sk = self.sk
        d = sk[f"{side}HandMiddle1"] - sk[f"{side}Hand"]
        across = sk[f"{side}HandIndex1"] - sk[f"{side}HandPinky1"]
        across = across / max(np.linalg.norm(across), 1e-9)
        n = np.cross(d, across) * (1 if side == "Left" else -1)  # left/right hands are mirror images
        return n / max(np.linalg.norm(n), 1e-9), across

    def _pose_fingers(self) -> None:
        if "LeftHandIndex1" not in self.sk.template.index:
            return
        for side in ("Left", "Right"):
            shape = self.hands.get(side, HAND_SHAPES["relaxed"])
            normal, across = self._palm(side)
            for finger in FINGERS:
                curls, spread = shape[finger][:3], shape[finger][3]
                # fingers close towards the palm; the thumb also crosses it towards the little finger
                target = normal if finger != "Thumb" else (normal - across) / np.sqrt(2)
                for i, curl in enumerate(curls, 1):
                    joint = f"{side}Hand{finger}{i}"
                    b = self.rest_dir(joint)
                    r = Rotation.identity()
                    axis = np.cross(b, target)
                    if curl and np.linalg.norm(axis) > 1e-6:
                        r = Rotation.from_rotvec(np.radians(curl) * axis / np.linalg.norm(axis))
                    if i == 1 and spread:
                        axis = np.cross(b, across)
                        if np.linalg.norm(axis) > 1e-6:
                            r = r * Rotation.from_rotvec(np.radians(spread) * axis / np.linalg.norm(axis))
                    self.rot[joint] = r

    def matrix(self) -> np.ndarray:
        self._pose_fingers()
        index = self.sk.template.index
        quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(index), 1))
        for joint, r in self.rot.items():
            quats[index[joint]] = r.as_quat()
        return quats


Motion = Callable[[Pose, float], None]  # (pose, phase 0..1)


@dataclass
class Preset:
    label: str
    duration: float
    motion: Motion
    loop: bool = True
    tags: list[str] = field(default_factory=list)
    # "plant": the lowest foot always touches the floor (hips height follows the legs)
    # "clamp": feet may leave the floor (jumps) but never go through it
    ground: str = "plant"


def _s(phase: float, k: float = 1.0, shift: float = 0.0) -> float:
    return float(np.sin(2 * np.pi * (k * phase + shift)))


def _relaxed_arms(p: Pose, sway: float = 0.0) -> None:
    p.arm("Left", 12, forward=sway, elbow=12)
    p.arm("Right", 12, forward=-sway, elbow=12)


# ---------------------------------------------------------------- library


def idle(p: Pose, t: float) -> None:
    breath = _s(t)
    p.lean("Spine1", 1.5 * breath)
    p.lean("Neck", -1.0 * breath)
    p.turn("Head", 3 * _s(t, 0.5))
    _relaxed_arms(p, sway=2 * breath)
    p.offset[1] = 0.003 * p.sk.height * breath


def walk(p: Pose, t: float) -> None:
    for side, shift in (("Left", 0.0), ("Right", 0.5)):
        swing = _s(t, shift=shift)
        p.leg(side, hip=24 * swing, knee=max(0.0, 55 * _s(t, shift=shift + 0.75)) + 6, ankle=8 * swing)
    p.arm("Left", 10, forward=-20 * _s(t), elbow=18)
    p.arm("Right", 10, forward=20 * _s(t), elbow=18)
    p.turn("Spine", 5 * _s(t))
    p.turn("Spine2", -7 * _s(t))
    p.lean("Spine1", 3)
    p.tilt("Hips", 3 * _s(t))
    p.offset[1] = -0.012 * p.sk.height * abs(_s(t, 1.0, 0.25))


def run(p: Pose, t: float) -> None:
    for side, shift in (("Left", 0.0), ("Right", 0.5)):
        swing = _s(t, shift=shift)
        p.leg(side, hip=40 * swing + 10, knee=max(0.0, 100 * _s(t, shift=shift + 0.7)) + 15, ankle=12 * swing)
    p.arm("Left", 15, forward=-40 * _s(t), elbow=85)
    p.arm("Right", 15, forward=40 * _s(t), elbow=85)
    p.lean("Spine", 10)
    p.turn("Spine1", 8 * _s(t))
    p.offset[1] = 0.03 * p.sk.height * abs(_s(t, 1.0)) - 0.02 * p.sk.height


def jump(p: Pose, t: float) -> None:
    # 0–.25 crouch, .25–.4 push, .4–.7 air, .7–.85 land, .85–1 recover
    def env(a, b):
        return float(np.clip((t - a) / (b - a), 0, 1))

    crouch = np.sin(np.pi * env(0.0, 0.4)) if t < 0.4 else 0.0
    land = np.sin(np.pi * env(0.7, 1.0)) if t > 0.7 else 0.0
    bend = max(crouch, land)
    air = np.sin(np.pi * env(0.38, 0.72))
    for side in ("Left", "Right"):
        p.leg(side, hip=60 * bend + 15 * air, knee=110 * bend + 30 * air, ankle=-20 * air)
    p.lean("Spine", 25 * bend)
    arms_up = 40 + 120 * air
    p.arm("Left", arms_up - 30 * bend, forward=-40 * bend + 20 * air, elbow=20)
    p.arm("Right", arms_up - 30 * bend, forward=-40 * bend + 20 * air, elbow=20)
    p.offset[1] = p.sk.height * (0.22 * air - 0.16 * bend)


def wave(p: Pose, t: float) -> None:
    p.arm("Left", 12, elbow=12)
    p.arm("Right", 145, forward=15, elbow=35)
    p.about("RightForeArm", Z, 22 * _s(t, 2.0))
    p.about("RightHand", Z, 10 * _s(t, 2.0, 0.1))
    p.hand("Right", "open")
    p.turn("Head", -10)
    p.tilt("Spine1", -4)
    p.offset[1] = 0.002 * p.sk.height * _s(t)


def dance(p: Pose, t: float) -> None:
    beat = _s(t, 2.0)
    p.offset[0] = 0.035 * p.sk.height * _s(t)
    p.offset[1] = -0.025 * p.sk.height * abs(beat)
    p.tilt("Spine", 8 * _s(t))
    p.tilt("Spine2", -10 * _s(t))
    p.turn("Spine1", 12 * _s(t, 1.0, 0.25))
    p.tilt("Head", 8 * _s(t, 2.0))
    p.arm("Left", 100 + 50 * _s(t, 1.0), forward=20, elbow=60 + 30 * beat)
    p.arm("Right", 100 - 50 * _s(t, 1.0), forward=20, elbow=60 - 30 * beat)
    for side, sh in (("Left", 0.0), ("Right", 0.5)):
        p.leg(side, hip=12 + 10 * _s(t, 2.0, sh), knee=20 + 25 * abs(beat), spread=6)


def punch(p: Pose, t: float) -> None:
    # guard stance with a right jab then a left jab
    jab_r = np.exp(-((t - 0.25) / 0.07) ** 2)
    jab_l = np.exp(-((t - 0.65) / 0.07) ** 2)
    p.leg("Left", hip=15, knee=20, spread=8)
    p.leg("Right", hip=-5, knee=15, spread=8)
    p.lean("Spine", 8)
    p.turn("Spine1", 18 * jab_r - 18 * jab_l)
    p.arm("Right", 30 + 50 * jab_r, forward=40 + 50 * jab_r, elbow=120 * (1 - jab_r) + 5)
    p.arm("Left", 30 + 50 * jab_l, forward=40 + 50 * jab_l, elbow=120 * (1 - jab_l) + 5)
    p.offset[1] = -0.03 * p.sk.height


def victory(p: Pose, t: float) -> None:
    pump = abs(_s(t, 2.0))
    p.arm("Left", 150 + 15 * pump, forward=10, elbow=25 * (1 - pump))
    p.arm("Right", 150 + 15 * pump, forward=10, elbow=25 * (1 - pump))
    p.lean("Spine1", -6)
    p.lean("Head", -12)
    for side in ("Left", "Right"):
        p.leg(side, hip=10 * pump, knee=20 * pump)
    p.offset[1] = p.sk.height * (0.05 * pump - 0.02)


def nod(p: Pose, t: float) -> None:
    _relaxed_arms(p)
    p.lean("Head", 18 * max(0.0, _s(t, 2.0)))
    p.lean("Neck", 6 * max(0.0, _s(t, 2.0)))


def shake(p: Pose, t: float) -> None:
    _relaxed_arms(p)
    p.turn("Head", 28 * _s(t, 2.0))
    p.turn("Neck", 8 * _s(t, 2.0))


def look_around(p: Pose, t: float) -> None:
    _relaxed_arms(p)
    p.turn("Spine2", 15 * _s(t))
    p.turn("Neck", 15 * _s(t))
    p.turn("Head", 25 * _s(t))
    p.lean("Head", -6 * _s(t, 2.0))


def sit(p: Pose, t: float) -> None:
    for side in ("Left", "Right"):
        p.leg(side, hip=90, knee=90, spread=4)
    thigh = np.linalg.norm(p.sk["LeftUpLeg"] - p.sk["LeftLeg"])
    p.offset[1] = -thigh * 0.95
    p.offset[2] = -thigh * 0.15
    p.lean("Spine", 3 + 1.5 * _s(t))
    p.arm("Left", 15, forward=45, elbow=55)
    p.arm("Right", 15, forward=45, elbow=55)


def t_pose(p: Pose, t: float) -> None:
    p.arm("Left", 90)
    p.arm("Right", 90)


def a_pose(p: Pose, t: float) -> None:
    p.arm("Left", 45)
    p.arm("Right", 45)


def _hold(t: float, rise: float = 0.2) -> float:
    """0 → 1 → 0 over a loop, holding at 1 in the middle."""
    return float(np.clip(min(t, 1 - t) / rise, 0, 1))


def fist(p: Pose, t: float) -> None:
    close = _hold(t, 0.25)
    p.arm("Left", 12, elbow=12)
    p.arm("Right", 30 + 20 * close, forward=40 + 25 * close, elbow=60 + 40 * close)
    p.hand("Right", "fist", close)
    p.hand("Left", "fist", 0.6 * close)
    p.lean("Head", -4 * close)
    p.offset[1] = 0.002 * p.sk.height * _s(t)


def point(p: Pose, t: float) -> None:
    aim = _hold(t, 0.2)
    p.arm("Left", 12, elbow=12)
    p.arm("Right", 12 + 68 * aim, forward=75 * aim, elbow=15 * (1 - aim) + 5)
    p.hand("Right", "point", aim)
    p.turn("Spine1", -8 * aim)
    p.turn("Head", -12 * aim)
    p.offset[1] = 0.002 * p.sk.height * _s(t)


def grab(p: Pose, t: float) -> None:
    # 0–.35 reach out opening the hand, .35–.5 close it, .5–.8 pull it in, .8–1 let go
    reach = float(np.sin(np.pi * min(t / 0.7, 1.0)))
    p.arm("Left", 12, elbow=12)
    p.arm("Right", 20 + 25 * reach, forward=20 + 50 * reach, elbow=70 - 55 * reach)
    if t < 0.35:
        p.hand("Right", "open", min(t / 0.25, 1.0))
    elif t < 0.8:
        p.hand("Right", "fist", min((t - 0.35) / 0.15, 1.0), base="open")
    else:
        p.hand("Right", "fist", (1.0 - t) / 0.2)
    p.lean("Spine1", 8 * reach)
    p.turn("Spine1", -6 * reach)
    p.lean("Head", 6 * reach)


def talk(p: Pose, t: float) -> None:
    # syllables: the jaw opens and closes at uneven rates, with small head nods
    open_ = 0.5 + 0.5 * _s(t, 6.0) * (0.6 + 0.4 * _s(t, 1.0, 0.2))
    p.morph["jawOpen"] = float(np.clip(open_ * (0.4 + 0.6 * abs(_s(t, 2.0, 0.1))), 0, 1))
    p.morph["smile"] = 0.15 + 0.1 * _s(t, 1.0)
    _relaxed_arms(p, sway=2 * _s(t))
    p.arm("Right", 15, forward=20 + 8 * _s(t, 2.0), elbow=50)
    p.lean("Head", 3 * _s(t, 3.0))
    p.turn("Head", 4 * _s(t, 1.0))


def blink(p: Pose, t: float) -> None:
    # two blinks per loop (the second one double), relaxed breathing in between
    lids = max(np.exp(-((t - 0.2) / 0.03) ** 2), np.exp(-((t - 0.7) / 0.03) ** 2), np.exp(-((t - 0.8) / 0.03) ** 2))
    p.morph["blink_L"] = p.morph["blink_R"] = float(lids)
    idle(p, t)


PRESETS: dict[str, Preset] = {
    "idle": Preset("Reposo (respirar)", 3.0, idle, tags=["básico"]),
    "walk": Preset("Caminar", 1.1, walk, tags=["locomoción"]),
    "run": Preset("Correr", 0.7, run, tags=["locomoción"]),
    "jump": Preset("Saltar", 1.4, jump, loop=False, tags=["acción"], ground="clamp"),
    "wave": Preset("Saludar", 2.0, wave, tags=["gesto"]),
    "dance": Preset("Bailar", 2.0, dance, tags=["diversión"]),
    "punch": Preset("Puñetazos", 1.2, punch, tags=["combate"]),
    "victory": Preset("Victoria", 2.0, victory, tags=["gesto"], ground="clamp"),
    "nod": Preset("Asentir (sí)", 1.5, nod, tags=["gesto"]),
    "shake": Preset("Negar (no)", 1.5, shake, tags=["gesto"]),
    "look_around": Preset("Mirar alrededor", 4.0, look_around, tags=["básico"]),
    "sit": Preset("Sentado", 3.0, sit, tags=["pose"]),
    "t_pose": Preset("T-pose", 1.0, t_pose, tags=["pose"]),
    "a_pose": Preset("A-pose", 1.0, a_pose, tags=["pose"]),
    "fist": Preset("Puño", 2.0, fist, tags=["manos"]),
    "point": Preset("Señalar", 2.5, point, tags=["manos"]),
    "grab": Preset("Agarrar", 2.4, grab, tags=["manos"]),
    "talk": Preset("Hablar", 3.0, talk, tags=["cara"]),
    "blink": Preset("Parpadear", 3.0, blink, tags=["cara"]),
}
DEFAULT_PRESETS = ["idle", "walk", "run", "jump", "wave"]
PRESET_SETS: dict[str, dict[str, Preset]] = {"humanoid": PRESETS}  # other templates register theirs


def preset_clip(skeleton: Skeleton, name: str) -> Clip:
    presets = PRESET_SETS.get(skeleton.template.name, {})
    if name not in presets:
        raise RigError(f"Animación desconocida para este tipo de esqueleto: {name}")
    preset = presets[name]
    frames = max(2, int(round(preset.duration * FPS)) + 1)
    times = np.linspace(0, preset.duration, frames)
    rotations, offsets, morphs = [], [], []
    for i in range(frames):
        pose = Pose(skeleton)
        preset.motion(pose, (i / (frames - 1)) % 1.0 if preset.loop else i / (frames - 1))
        rotations.append(pose.matrix())
        offsets.append(pose.offset.copy())
        morphs.append(pose.morph)
    if preset.loop:  # make the last frame exactly the first for seamless loops
        rotations[-1], offsets[-1], morphs[-1] = rotations[0], offsets[0], morphs[0]
    rotations, offsets = np.array(rotations), np.array(offsets)
    ground_feet(skeleton, rotations, offsets, preset.ground)
    names = sorted({n for m in morphs for n in m})
    weights = {n: np.array([m.get(n, 0.0) for m in morphs]) for n in names} or None
    return Clip(name, times, rotations, offsets, preset.loop, weights)


FOOT_JOINTS = [JOINT_INDEX[n] for n in HUMANOID.feet]  # humanoid default, kept for callers and tests


def ground_feet(skeleton: Skeleton, rotations: np.ndarray, offsets: np.ndarray, mode: str) -> None:
    """Adjust the hips height per frame so feet meet the floor (in place)."""
    feet = skeleton.foot_indices
    floor = skeleton.positions[feet, 1].min()
    for i in range(len(rotations)):
        lowest = forward_kinematics(skeleton, rotations[i], offsets[i], max(feet) + 1)[feet, 1].min()
        if mode == "plant" or lowest < floor:
            offsets[i, 1] += floor - lowest


# ---------------------------------------------------------------- BVH motion capture


@dataclass
class BvhJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]


def parse_bvh(text: str) -> tuple[list[BvhJoint], np.ndarray, float]:
    """Joints, (frames, channels) motion values and frame time."""
    tokens = text.replace("{", " { ").replace("}", " } ").split()
    if not tokens or tokens[0].upper() != "HIERARCHY":
        raise RigError("El archivo no es un BVH válido (falta HIERARCHY)")
    joints: list[BvhJoint] = []
    stack: list[int] = []
    i = 1
    pending: tuple[str, int] | None = None
    try:
        while tokens[i].upper() != "MOTION":
            tok = tokens[i].upper()
            if tok in ("ROOT", "JOINT"):
                pending = (tokens[i + 1], stack[-1] if stack else -1)
                i += 2
            elif tok == "END":  # End Site
                parent = stack[-1]
                pending = (joints[parent].name + "_End", parent)
                i += 2
            elif tok == "{":
                name, parent = pending  # type: ignore[misc]
                joints.append(BvhJoint(name, parent, np.zeros(3), []))
                stack.append(len(joints) - 1)
                i += 1
            elif tok == "}":
                stack.pop()
                i += 1
            elif tok == "OFFSET":
                joints[stack[-1]].offset = np.array([float(x) for x in tokens[i + 1 : i + 4]])
                i += 4
            elif tok == "CHANNELS":
                n = int(tokens[i + 1])
                joints[stack[-1]].channels = tokens[i + 2 : i + 2 + n]
                i += 2 + n
            else:
                i += 1
        i += 1  # MOTION
        frames = int(tokens[i + 1])  # Frames: N
        frame_time = float(tokens[i + 4])  # Frame Time: x
        values = np.array([float(x) for x in tokens[i + 5 :]])
    except (IndexError, ValueError) as exc:
        raise RigError("No se pudo leer el BVH (formato inesperado)") from exc
    n_channels = sum(len(j.channels) for j in joints)
    if frames < 1 or n_channels == 0 or len(values) < frames * n_channels:
        raise RigError("El BVH no contiene fotogramas de movimiento")
    return joints, values[: frames * n_channels].reshape(frames, n_channels), frame_time


_SIDE = re.compile(r"^(left|right|l|r)(?=[a-z])")
_CANON = {
    "hips": "Hips", "pelvis": "Hips", "root": "Hips", "hip": "Hips",
    "spine": "Spine", "abdomen": "Spine", "spine0": "Spine", "lowerback": "Spine",
    "spine1": "Spine1", "chest": "Spine1", "spine2": "Spine2", "chest2": "Spine2", "upperchest": "Spine2",
    "neck": "Neck", "neck1": "Neck", "head": "Head",
    "shoulder": "Shoulder", "collar": "Shoulder", "clavicle": "Shoulder",
    "arm": "Arm", "shldr": "Arm", "upperarm": "Arm", "humerus": "Arm",
    "forearm": "ForeArm", "lowerarm": "ForeArm", "elbow": "ForeArm", "radius": "ForeArm",
    "hand": "Hand", "wrist": "Hand",
    "upleg": "UpLeg", "thigh": "UpLeg", "upperleg": "UpLeg", "femur": "UpLeg",
    "leg": "Leg", "shin": "Leg", "lowerleg": "Leg", "calf": "Leg", "knee": "Leg", "tibia": "Leg",
    "foot": "Foot", "ankle": "Foot", "toebase": "ToeBase", "toe": "ToeBase", "toes": "ToeBase",
}


def canonical_name(name: str) -> str | None:
    """Map common rig naming schemes (Mixamo, CMU, 3ds Max Biped, Rokoko…) to our joint names."""
    n = re.sub(r"^.*[:|]", "", name)  # mixamorig:LeftArm, Armature|LeftArm
    n = re.sub(r"[^a-z0-9]", "", n.lower().replace("bip01", ""))
    side = ""
    m = _SIDE.match(n)
    if m and n[m.end():] in _CANON and _CANON[n[m.end():]] not in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"):
        side = "Left" if m.group(1).startswith("l") else "Right"
        n = n[m.end():]
    else:  # suffix style: upperarm_l, thigh.r
        m2 = re.match(r"^(.*?)(l|r|left|right)$", n)
        if m2 and m2.group(1) in _CANON and _CANON[m2.group(1)] not in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"):
            side = "Left" if m2.group(2).startswith("l") else "Right"
            n = m2.group(1)
    base = _CANON.get(n)
    if base is None:
        return None
    if base in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"):
        return base if not side else None
    return f"{side}{base}" if side else None


def _euler_order(channels: list[str]) -> str:
    return "".join(c[0].upper() for c in channels if c.lower().endswith("rotation"))


def bvh_global(joints: list[BvhJoint], motion: np.ndarray) -> tuple[list[Rotation], np.ndarray, np.ndarray]:
    """Per joint: global rotations (frames), global positions (frames, J, 3), rest positions (J, 3)."""
    frames = motion.shape[0]
    rest = np.zeros((len(joints), 3))
    for j, joint in enumerate(joints):
        rest[j] = joint.offset + (rest[joint.parent] if joint.parent >= 0 else 0)
    glob: list[Rotation] = []
    pos = np.zeros((frames, len(joints), 3))
    col = 0
    for j, joint in enumerate(joints):
        n = len(joint.channels)
        vals = motion[:, col : col + n]
        col += n
        rot_idx = [k for k, c in enumerate(joint.channels) if c.lower().endswith("rotation")]
        pos_idx = {c[0].upper(): k for k, c in enumerate(joint.channels) if c.lower().endswith("position")}
        local = (Rotation.from_euler(_euler_order(joint.channels), vals[:, rot_idx], degrees=True)
                 if rot_idx else Rotation.identity(frames))
        if len(pos_idx) == 3:
            translation = vals[:, [pos_idx["X"], pos_idx["Y"], pos_idx["Z"]]]
        else:
            translation = np.tile(joint.offset, (frames, 1))
        if joint.parent < 0:
            glob.append(local)
            pos[:, j] = translation
        else:
            parent_rot = glob[joint.parent]
            glob.append(parent_rot * local)
            pos[:, j] = pos[:, joint.parent] + parent_rot.apply(translation)
    return glob, pos, rest


_FINGER = re.compile(r"^(left|right|l|r)(?:hand)?(thumb|index|middle|ring|pinky)([123])$")


def finger_name(name: str) -> str | None:
    """Finger joints in Mixamo-like names (LeftHandIndex1, mixamorig:RightHandThumb2, l_index3…)."""
    n = re.sub(r"^.*[:|]", "", name)
    m = _FINGER.match(re.sub(r"[^a-z0-9]", "", n.lower()))
    if not m:
        return None
    side = "Left" if m.group(1).startswith("l") else "Right"
    return f"{side}Hand{m.group(2).capitalize()}{m.group(3)}"


def retarget_bvh(skeleton: Skeleton, text: str, name: str = "mocap", in_place: bool = True,
                 max_seconds: float = 120.0) -> Clip:
    """Fingers follow the mocap only when it has full 3-joint chains for them; otherwise (and the
    jaw always) they stay in the rest pose."""
    if skeleton.template.name != HUMANOID.name:
        raise RigError("La captura de movimiento BVH solo se aplica a personajes humanoides")
    joints, motion, frame_time = parse_bvh(text)
    frame_time = frame_time if frame_time > 0 else 1 / FPS
    step = max(1, int(round((1 / FPS) / frame_time)))  # resample to ~30 fps
    motion = motion[::step][: int(max_seconds * FPS) + 1]
    src_glob, src_pos, src_rest = bvh_global(joints, motion)
    frames = motion.shape[0]

    src_index: dict[str, int] = {}
    fingers: dict[str, int] = {}
    for j, joint in enumerate(joints):
        canon = canonical_name(joint.name)
        if canon and canon not in src_index:
            src_index[canon] = j
        finger = finger_name(joint.name) if joint.channels else None
        if finger and finger not in fingers:
            fingers[finger] = j
    tpl = skeleton.template
    for base in {f[:-1] for f in fingers}:
        if all(f"{base}{i}" in fingers for i in (1, 2, 3)) and f"{base}1" in tpl.index:
            src_index.update({f"{base}{i}": fingers[f"{base}{i}"] for i in (1, 2, 3)})
    needed = ["Hips", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm"]
    if any(n not in src_index for n in needed):
        raise RigError("El BVH no tiene un esqueleto humanoide reconocible (caderas, piernas y brazos)")

    # face the same way: the source's left-hip direction must become +X
    left = src_rest[src_index["LeftUpLeg"]] - src_rest[src_index["RightUpLeg"]]
    yaw = np.arctan2(-left[2], left[0])
    fix = Rotation.from_rotvec([0, -yaw, 0]) if abs(yaw) > 1e-3 else Rotation.identity()
    fix_inv = fix.inv()
    src_glob = [fix * g * fix_inv for g in src_glob]
    src_pos = fix.apply(src_pos.reshape(-1, 3)).reshape(src_pos.shape)
    src_rest = fix.apply(src_rest)

    tgt_glob: dict[str, Rotation] = {}
    align: dict[str, Rotation] = {}
    for name_j, parent_name in tpl.joints:
        child = tpl.bone_child.get(name_j)
        # a finger's last bone points like the source's tip (its End Site), when the source has one
        src_child = src_index.get(child)
        if src_child is None and name_j in src_index and child and child.endswith("_End") and name_j[-1] == "3":
            src_child = next((k for k, jt in enumerate(joints) if jt.parent == src_index[name_j]), None)
        if name_j in src_index and src_child is not None:
            t_dir = skeleton[child] - skeleton[name_j]
            s_dir = src_rest[src_child] - src_rest[src_index[name_j]]
            if np.linalg.norm(t_dir) > 1e-9 and np.linalg.norm(s_dir) > 1e-9:
                align[name_j] = Rotation.align_vectors(s_dir[None], t_dir[None])[0]
        if name_j not in align:
            align[name_j] = align.get(parent_name, Rotation.identity()) if parent_name else Rotation.identity()
        if name_j in src_index:
            tgt_glob[name_j] = src_glob[src_index[name_j]] * align[name_j]
        else:  # joint the mocap lacks (e.g. extra spine): follow the parent rigidly
            tgt_glob[name_j] = tgt_glob[parent_name] if parent_name else Rotation.identity(frames)

    rotations = np.zeros((frames, len(tpl.names), 4))
    for name_j, parent_name in tpl.joints:
        local = tgt_glob[name_j] if not parent_name else tgt_glob[parent_name].inv() * tgt_glob[name_j]
        rotations[:, tpl.index[name_j]] = local.as_quat()

    # root motion scaled by leg length
    def leg_len(rest_lookup) -> float:
        return float(np.linalg.norm(rest_lookup("LeftUpLeg") - rest_lookup("LeftFoot")))

    src_leg = leg_len(lambda n: src_rest[src_index[n]]) if "LeftFoot" in src_index else 0.0
    tgt_leg = leg_len(lambda n: skeleton[n])
    scale = tgt_leg / src_leg if src_leg > 1e-9 else skeleton.height / max(np.ptp(src_rest[:, 1]), 1e-9)
    hips = src_index["Hips"]
    offsets = (src_pos[:, hips] - src_pos[0, hips]) * scale
    # vertical: hips height above the floor (y=0 in BVH space) versus the source's standing height,
    # so crouches, jumps and sitting keep their real height whatever the root OFFSET convention
    feet = [src_index[n] for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase") if n in src_index]
    standing = src_rest[hips, 1] - (src_rest[feet, 1].min() if feet else 0.0)
    offsets[:, 1] = (src_pos[:, hips, 1] - standing) * scale
    if in_place:
        offsets[:, [0, 2]] = 0.0
    times = np.arange(frames) / FPS
    calibrate_floor(skeleton, rotations, offsets)
    ground_feet(skeleton, rotations, offsets, "clamp")
    return Clip(name, times, rotations, offsets, loop=False)


def calibrate_floor(skeleton: Skeleton, rotations: np.ndarray, offsets: np.ndarray) -> None:
    """Shift the clip vertically so the feet's contact level (5th percentile of the lowest foot)
    matches the character's floor: mocap floors are rarely exactly at y=0."""
    feet = skeleton.foot_indices
    floor = skeleton.positions[feet, 1].min()
    sample = range(0, len(rotations), max(1, len(rotations) // 60))
    lowest = [forward_kinematics(skeleton, rotations[i], offsets[i], max(feet) + 1)[feet, 1].min() for i in sample]
    offsets[:, 1] -= np.percentile(lowest, 5) - floor


# ---------------------------------------------------------------- forward kinematics (tests, previews)


def forward_kinematics(skeleton: Skeleton, rotations: np.ndarray, hips_offset: np.ndarray,
                       count: int | None = None) -> np.ndarray:
    """World joint positions (J, 3) for one frame of local rotations. With `count`, only the first
    `count` joints are solved (joints are in topological order); the rest stay at zero."""
    parents = skeleton.template.parents
    n = len(parents) if count is None else count
    local = Rotation.from_quat(rotations[:n]).as_matrix()
    world = np.empty_like(local)
    rest = skeleton.positions
    pos = np.zeros_like(rest)
    for j in range(n):
        parent = parents[j]
        if parent < 0:
            world[j] = local[j]
            pos[j] = rest[j] + hips_offset
        else:
            world[j] = world[parent] @ local[j]
            pos[j] = pos[parent] + world[parent] @ (rest[j] - rest[parent])
    return pos
