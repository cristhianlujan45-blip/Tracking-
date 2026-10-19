import io
import json
import os
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import geometry
from app.config import Settings
from app.main import create_app
from app.providers.mock import build_humanoid
from app.rigging import rig
from app.rigging.animations import PRESETS, canonical_name, forward_kinematics, preset_clip, retarget_bvh
from app.rigging.skeleton import JOINT_INDEX, JOINTS, build_skeleton, estimate_markers

BLENDER_PY = os.getenv("TEST_BLENDER_PATH", "/tmp/claude-0/bpyenv/bin/python")


def gltf_json(glb: bytes) -> dict:
    length = struct.unpack("<I", glb[12:16])[0]
    return json.loads(glb[20:20 + length])


def humanoid(arm_angle=45.0):
    mesh = build_humanoid(np.random.default_rng(0), np.array([0.2, 0.4, 0.8]), arm_angle=arm_angle)
    mesh.apply_translation([0, -mesh.vertices[:, 1].min(), 0])
    return mesh


def make_bvh(skeleton, raise_left_arm=True, names=lambda n: f"mixamorig:{n}", walk_z=0.0) -> str:
    """BVH in T-pose (cm) of our own joint layout; frame 1 lifts the left arm straight up."""
    tpose = {"LeftArm": [1, 0, 0], "LeftForeArm": [1, 0, 0], "LeftHand": [1, 0, 0], "LeftHand_End": [1, 0, 0],
             "RightArm": [-1, 0, 0], "RightForeArm": [-1, 0, 0], "RightHand": [-1, 0, 0], "RightHand_End": [-1, 0, 0]}
    parent_of = dict(JOINTS)
    pos = {}
    for n, parent in JOINTS:
        if parent is None:
            pos[n] = np.array([0, 100.0, 0])
            continue
        d = skeleton[n] - skeleton[parent]
        if n in tpose:
            d = np.array(tpose[n], float) * np.linalg.norm(d)
        pos[n] = pos[parent] + d * 100
    children = {n: [c for c, p in JOINTS if p == n] for n, _ in JOINTS}
    lines, order = ["HIERARCHY"], []

    def emit(n, depth):
        ind = "  " * depth
        off = pos[n] - (pos[parent_of[n]] if parent_of[n] else 0)
        if n.endswith("_End"):
            lines.extend([f"{ind}End Site", ind + "{", f"{ind}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}", ind + "}"])
            return
        lines.extend([f"{ind}{'ROOT' if parent_of[n] is None else 'JOINT'} {names(n)}", ind + "{",
                      f"{ind}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}",
                      f"{ind}  CHANNELS " + ("6 Xposition Yposition Zposition " if parent_of[n] is None else "3 ")
                      + "Zrotation Xrotation Yrotation"])
        order.append(n)
        for c in children[n]:
            emit(c, depth + 1)
        lines.append(ind + "}")

    emit("Hips", 0)
    frames = []
    for f in range(2):
        values = []
        for n in order:
            if n == "Hips":
                values += [0, 100, walk_z * f]
            values += [90 if (f == 1 and raise_left_arm and n == "LeftArm") else 0, 0, 0]
        frames.append(" ".join(map(str, values)))
    return "\n".join(lines + ["MOTION", "Frames: 2", "Frame Time: 0.0333333"] + frames)


# ---------------------------------------------------------------- core


@pytest.mark.parametrize("angle", [20, 45, 85])
def test_markers_close_to_truth(angle):
    mesh = humanoid(angle)
    markers = estimate_markers(mesh)
    a = np.radians(angle)
    d = np.array([np.sin(a), -np.cos(a)])
    shoulder = np.array([0.19, 1.38])
    truth = {"elbow_l": shoulder + 0.29 * d, "wrist_l": shoulder + 0.55 * d, "ankle_l": [0.11, 0.1], "chin": [0, 1.52]}
    for name, expected in truth.items():
        assert np.linalg.norm(np.array(markers[name]) - expected) < 0.06, (name, markers[name], expected)
    assert markers["wrist_r"][0] == pytest.approx(-markers["wrist_l"][0], abs=0.02)


def test_skin_weights_follow_anatomy():
    mesh = humanoid()
    result = rig(geometry.mesh_to_glb(mesh), presets=["t_pose"])
    gltf = gltf_json(result.glb)
    assert len(gltf["skins"][0]["joints"]) == len(JOINTS)
    assert [a["name"] for a in gltf["animations"]] == ["t_pose"]
    # T-pose clip: hands go up to shoulder height (forward kinematics on the stored skeleton)
    sk = build_skeleton(mesh.vertices, result.markers)
    clip = preset_clip(sk, "t_pose")
    pos = forward_kinematics(sk, clip.rotations[0], clip.hips_offset[0])
    assert pos[JOINT_INDEX["LeftHand"], 1] == pytest.approx(sk["LeftArm"][1], abs=0.02)


def test_presets_keep_feet_on_floor():
    sk = build_skeleton(humanoid().vertices, estimate_markers(humanoid()))
    floor = sk.positions[[JOINT_INDEX[n] for n in ("LeftToe_End", "RightToe_End")], 1].min()
    for name in PRESETS:
        clip = preset_clip(sk, name)
        lowest = min(forward_kinematics(sk, clip.rotations[i], clip.hips_offset[i])[:, 1].min()
                     for i in range(0, len(clip.times), 4))
        assert lowest >= floor - 1e-6, name
        assert np.allclose(np.linalg.norm(clip.rotations, axis=-1), 1, atol=1e-6)


def test_bvh_retarget_from_tpose_to_apose():
    mesh = humanoid(45)
    sk = build_skeleton(mesh.vertices, estimate_markers(mesh))
    clip = retarget_bvh(sk, make_bvh(sk))
    first = forward_kinematics(sk, clip.rotations[0], clip.hips_offset[0])
    raised = forward_kinematics(sk, clip.rotations[1], clip.hips_offset[1])
    shoulder = sk["LeftArm"]
    assert first[JOINT_INDEX["LeftHand"], 1] == pytest.approx(shoulder[1], abs=0.03)  # T-pose arms
    assert raised[JOINT_INDEX["LeftHand"], 0] == pytest.approx(shoulder[0], abs=0.03)  # straight up
    assert raised[JOINT_INDEX["LeftHand"], 1] > shoulder[1] + 0.4
    assert raised[JOINT_INDEX["RightHand"], 1] == pytest.approx(shoulder[1], abs=0.03)


def test_bvh_root_motion_in_place_and_not():
    sk = build_skeleton(humanoid().vertices, estimate_markers(humanoid()))
    moving = retarget_bvh(sk, make_bvh(sk, walk_z=50), in_place=False)
    assert moving.hips_offset[1, 2] > 0.2
    still = retarget_bvh(sk, make_bvh(sk, walk_z=50), in_place=True)
    assert still.hips_offset[1, 2] == 0


@pytest.mark.parametrize("name,expected", [
    ("mixamorig:LeftForeArm", "LeftForeArm"), ("lShldr", "LeftArm"), ("RightUpLeg", "RightUpLeg"),
    ("Bip01 L Thigh", "LeftUpLeg"), ("upperarm_r", "RightArm"), ("Hips", "Hips"), ("Chest", "Spine1"),
    ("LeftHandIndex1", None), ("Armature|Head", "Head"),
])
def test_bone_name_mapping(name, expected):
    assert canonical_name(name) == expected


def test_bad_bvh_is_rejected():
    from app.rigging import RigError

    sk = build_skeleton(humanoid().vertices, estimate_markers(humanoid()))
    with pytest.raises(RigError):
        retarget_bvh(sk, "not a bvh")
    with pytest.raises(RigError):  # no humanoid joints
        retarget_bvh(sk, make_bvh(sk, names=lambda n: f"bone_{abs(hash(n)) % 1000}"))


# ---------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0, blender_path="/nonexistent")
    with TestClient(create_app(settings)) as c:
        yield c


def wait_done(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def character(client):
    job = client.post("/api/jobs/text", json={"prompt": "un personaje guerrero", "quality": "draft"}).json()[0]
    return wait_done(client, job["id"])


def test_presets_endpoint(client):
    body = client.get("/api/rig/presets").json()
    assert {p["id"] for p in body["presets"]} >= {"idle", "walk", "run", "jump", "wave", "dance"}
    assert len(body["markers"]) == 12


def test_rig_flow(client, character):
    markers = client.post(f"/api/jobs/{character['id']}/rig/markers").json()
    assert markers["yaw"] == 0 and markers["view_yaw"] == 0 and len(markers["markers"]) == 12

    job = client.post(f"/api/jobs/{character['id']}/rig", json={"animations": ["idle", "walk", "dance"]}).json()
    assert job["mode"] == "rig" and job["parent_id"] == character["id"]
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["params"]["clips"] == ["idle", "walk", "dance"] and len(job["params"]["markers"]) == 12
    glb = client.get(job["model_url"]).content
    gltf = gltf_json(glb)
    assert gltf["skins"] and [a["name"] for a in gltf["animations"]] == ["idle", "walk", "dance"]
    exported = client.get(f"/api/jobs/{job['id']}/export", params={"format": "glb", "size_mm": 50}).content
    assert gltf_json(exported)["skins"]  # GLB export keeps the rig

    # editing markers of a rigged character returns the stored ones, viewer already aligned
    again = client.post(f"/api/jobs/{job['id']}/rig/markers").json()
    assert again["markers"] == job["params"]["markers"] and again["view_yaw"] == 0


def test_rig_with_custom_markers_and_symmetry(client, character):
    auto = client.post(f"/api/jobs/{character['id']}/rig/markers").json()["markers"]
    custom = dict(auto, wrist_l=[auto["wrist_l"][0] + 0.02, auto["wrist_l"][1]], wrist_r=[0, 0])
    job = wait_done(client, client.post(f"/api/jobs/{character['id']}/rig",
                                        json={"markers": custom, "symmetric": True, "animations": []}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    used = job["params"]["markers"]
    assert used["wrist_r"] == [-custom["wrist_l"][0], custom["wrist_l"][1]]
    assert "animations" not in gltf_json(client.get(job["model_url"]).content)


def test_rig_validation(client, character):
    assert client.post(f"/api/jobs/{character['id']}/rig", json={"animations": ["moonwalk"]}).status_code == 422
    assert client.post(f"/api/jobs/{character['id']}/rig", json={"markers": {"chin": [0, 1]}}).status_code == 422
    assert client.post(f"/api/jobs/{character['id']}/animate", data={"animations": "walk"}).status_code == 400


def test_animate_with_bvh(client, character):
    rigged = wait_done(client, client.post(f"/api/jobs/{character['id']}/rig", json={"animations": ["idle"]}).json()["id"])
    sk_positions = rigged["params"]["skeleton"]
    from app.rigging.skeleton import JOINT_NAMES, Skeleton

    sk = Skeleton(np.array([sk_positions[n] for n in JOINT_NAMES]))
    bvh = make_bvh(sk).encode()
    resp = client.post(f"/api/jobs/{rigged['id']}/animate", data={"animations": "idle,wave"},
                       files=[("bvh", ("brazo arriba.bvh", bvh, "text/plain"))])
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["params"]["clips"] == ["idle", "wave", "brazo_arriba"]
    assert job["params"]["source_id"] == character["id"]  # always rigs the original static mesh

    # a later re-bake keeps the uploaded mocap when asked to
    keep = wait_done(client, client.post(f"/api/jobs/{job['id']}/animate",
                                         data={"animations": "run", "keep_bvh": "brazo_arriba"}).json()["id"])
    assert keep["params"]["clips"] == ["run", "brazo_arriba"]

    bad = wait_done(client, client.post(f"/api/jobs/{rigged['id']}/animate",
                                        files=[("bvh", ("x.bvh", b"garbage", "text/plain"))]).json()["id"])
    assert bad["status"] == "failed" and "BVH" in bad["error"]


def test_fbx_needs_blender(client, character):
    assert "fbx" not in client.get("/api/config").json()["export_formats"]
    resp = client.get(f"/api/jobs/{character['id']}/export", params={"format": "fbx"})
    assert resp.status_code == 400 and "Blender" in resp.json()["detail"]


@pytest.mark.skipif(not os.path.exists(BLENDER_PY), reason="Blender (bpy) not available")
def test_fbx_export_with_blender(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0, blender_path=BLENDER_PY)
    with TestClient(create_app(settings)) as c:
        assert "fbx" in c.get("/api/config").json()["export_formats"]
        job = wait_done(c, c.post("/api/jobs/text", json={"prompt": "un robot", "quality": "draft"}).json()[0]["id"])
        rigged = wait_done(c, c.post(f"/api/jobs/{job['id']}/rig", json={"animations": ["walk"]}).json()["id"])
        fbx = c.get(f"/api/jobs/{rigged['id']}/export", params={"format": "fbx"})
        assert fbx.status_code == 200, fbx.text
        assert fbx.content.startswith(b"Kaydara FBX Binary")
        assert fbx.headers["content-disposition"].endswith('.fbx"')


def test_markers_auto_on_rigged_character(client, character):
    rigged = wait_done(client, client.post(f"/api/jobs/{character['id']}/rig", json={
        "markers": dict(client.post(f"/api/jobs/{character['id']}/rig/markers").json()["markers"], chin=[0, 1.0]),
        "animations": []}).json()["id"])
    assert client.post(f"/api/jobs/{rigged['id']}/rig/markers").json()["markers"]["chin"] == [0, 1.0]
    fresh = client.post(f"/api/jobs/{rigged['id']}/rig/markers", json={"auto": True}).json()
    assert fresh["markers"]["chin"][1] > 1.3 and fresh["view_yaw"] == 0
    turned = client.post(f"/api/jobs/{rigged['id']}/rig/markers", json={"yaw": 90, "auto": True}).json()
    assert turned["yaw"] == 90 and turned["view_yaw"] == 90
