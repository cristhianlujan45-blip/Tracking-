import io
import json
import os
import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

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


def make_bvh(skeleton, raise_left_arm=True, names=lambda n: f"mixamorig:{n}", walk_z=0.0, joints=JOINTS,
             bend=None) -> str:
    """BVH in T-pose (cm) of our own joint layout; frame 1 lifts the left arm straight up (and
    applies `bend`: joint → Z rotation in degrees)."""
    tpose = {"LeftArm": [1, 0, 0], "LeftForeArm": [1, 0, 0], "LeftHand": [1, 0, 0], "LeftHand_End": [1, 0, 0],
             "RightArm": [-1, 0, 0], "RightForeArm": [-1, 0, 0], "RightHand": [-1, 0, 0], "RightHand_End": [-1, 0, 0]}
    parent_of = dict(joints)
    pos = {}
    for n, parent in joints:
        if parent is None:
            pos[n] = np.array([0, 100.0, 0])
            continue
        d = skeleton[n] - skeleton[parent]
        if n in tpose:
            d = np.array(tpose[n], float) * np.linalg.norm(d)
        pos[n] = pos[parent] + d * 100
    children = {n: [c for c, p in joints if p == n] for n, _ in joints}
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
            z = 90 if (f == 1 and raise_left_arm and n == "LeftArm") else (bend or {}).get(n, 0) * f
            values += [z, 0, 0]
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
    # 27 body joints + jaw + 3-bone fingers (with tips) on both hands
    assert len(gltf["skins"][0]["joints"]) == len(JOINTS) + 2 + 40 == len(result.skeleton)
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


def test_mocap_library_clips(client, character):
    library = client.get("/api/rig/presets").json()["library"]
    assert len(library) >= 600 and {"Caminar", "Baile", "Combate"} <= {c["category"] for c in library}
    walk = next(c for c in library if c["label"] == "Caminar triste")
    assert walk["label_en"] == "Sad walk"
    job = wait_done(client, client.post(f"/api/jobs/{character['id']}/rig",
                                        json={"animations": ["idle", walk["id"], "cmu_17_10"]}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["params"]["clips"] == ["idle", "Caminar triste", "Boxeo"]
    names = [a["name"] for a in gltf_json(client.get(job["model_url"]).content)["animations"]]
    assert names == ["idle", "Caminar triste", "Boxeo"]


def test_every_library_clip_retargets_with_feet_on_floor():
    from app.rigging import mocap
    from app.rigging.animations import FOOT_JOINTS

    mesh = humanoid()
    sk = build_skeleton(mesh.vertices, estimate_markers(mesh))
    floor = sk.positions[FOOT_JOINTS, 1].min()
    for clip_id, entry in mocap.catalogue().items():
        clip = retarget_bvh(sk, mocap.clip_text(clip_id), name=entry["label"])
        lows = [forward_kinematics(sk, clip.rotations[i], clip.hips_offset[i])[FOOT_JOINTS, 1].min()
                for i in range(0, len(clip.times), max(1, len(clip.times) // 60))]
        assert min(lows) >= floor - 1e-6, clip_id
        assert np.percentile(lows, 5) == pytest.approx(floor, abs=0.03 * sk.height), clip_id


# ---------------------------------------------------------------- quadrupeds


def dog():
    from app.providers.mock import build_quadruped

    mesh = build_quadruped(np.random.default_rng(0), np.array([0.7, 0.5, 0.3]))
    mesh.apply_translation([0, -mesh.vertices[:, 1].min(), 0])
    return mesh


def test_quadruped_detection_orientation_and_markers():
    import trimesh as _trimesh

    from app.rigging import auto_markers

    for yaw in (0, 90, 180, 270):
        mesh = dog()
        mesh.apply_transform(_trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0]))
        result = auto_markers(geometry.mesh_to_glb(mesh))
        assert result["template"] == "quadruped" and result["view"] == "side"
        assert (result["yaw"] + yaw) % 360 == 0, (yaw, result["yaw"])  # turned back to face +Z
        m = result["markers"]
        assert m["nose"][0] > m["withers"][0] > m["rump"][0] > m["tail_tip"][0]  # head → tail along Z
        assert m["front_paw"][1] < 0.1 and m["back_hock"][1] < m["back_knee"][1] < m["back_hip"][1]
    assert auto_markers(geometry.mesh_to_glb(humanoid()))["template"] == "humanoid"


def test_quadruped_gaits_keep_paws_on_floor():
    from app.rigging import quadruped
    from app.rigging.animations import forward_kinematics as fk

    mesh = dog()
    sk = quadruped.build_skeleton(mesh.vertices, quadruped.estimate_markers(mesh))
    assert sk.template.name == "quadruped" and len(sk.template.names) == 26
    feet = sk.foot_indices
    floor = sk.positions[feet, 1].min()
    for name in quadruped.QUAD_PRESETS:
        clip = preset_clip(sk, name)
        lows = [fk(sk, clip.rotations[i], clip.hips_offset[i])[feet, 1].min() for i in range(len(clip.times))]
        assert min(lows) >= floor - 1e-6, name
    trot = preset_clip(sk, "q_trot")
    paw = [fk(sk, trot.rotations[i], trot.hips_offset[i])[sk.template.index["FrontLeftFoot"], 2]
           for i in range(len(trot.times))]
    assert np.ptp(paw) > 0.1  # the paw actually travels


def test_rig_quadruped_api(client):
    job = wait_done(client, client.post("/api/jobs/text", json={"prompt": "un perro", "quality": "draft"}).json()[0]["id"])
    markers = client.post(f"/api/jobs/{job['id']}/rig/markers").json()
    assert markers["template"] == "quadruped" and markers["view"] == "side"
    assert set(markers["labels"]) == set(client.get("/api/rig/presets").json()["quadruped"]["markers"][i]["id"]
                                         for i in range(11))
    rigged = wait_done(client, client.post(f"/api/jobs/{job['id']}/rig",
                                           json={"template": "quadruped", "animations": ["q_walk", "q_trot"]}).json()["id"])
    assert rigged["status"] == "succeeded", rigged["error"]
    assert rigged["params"]["template"] == "quadruped" and rigged["params"]["clips"] == ["q_walk", "q_trot"]
    gltf = gltf_json(client.get(rigged["model_url"]).content)
    assert len(gltf["skins"][0]["joints"]) == 26
    # humanoid-only things are refused clearly
    bad = wait_done(client, client.post(f"/api/jobs/{job['id']}/rig",
                                        json={"template": "quadruped", "animations": ["cmu_16_15"]}).json()["id"])
    assert bad["status"] == "failed" and "no disponible" in bad["error"]
    # re-rig with edited side-view markers
    edited = dict(markers["markers"], nose=[markers["markers"]["nose"][0] - 0.02, markers["markers"]["nose"][1]])
    again = wait_done(client, client.post(f"/api/jobs/{rigged['id']}/rig",
                                          json={"markers": edited, "animations": ["q_idle"]}).json()["id"])
    assert again["status"] == "succeeded", again["error"] and again["params"]["template"] == "quadruped"


def test_auto_template_ignores_clips_for_the_other_body(client):
    dog_job = wait_done(client, client.post("/api/jobs/text", json={"prompt": "un perro", "quality": "draft"}).json()[0]["id"])
    rigged = wait_done(client, client.post(f"/api/jobs/{dog_job['id']}/rig",
                                           json={"animations": ["idle", "walk", "cmu_16_15", "q_trot"]}).json()["id"])
    assert rigged["status"] == "succeeded", rigged["error"]
    assert rigged["params"]["template"] == "quadruped" and rigged["params"]["clips"] == ["q_trot"]
    only_human = wait_done(client, client.post(f"/api/jobs/{dog_job['id']}/rig",
                                               json={"animations": ["idle", "walk"]}).json()["id"])
    assert only_human["status"] == "succeeded" and only_human["params"]["clips"]  # falls back to the gaits


def _textured_humanoid_glb(with_engine_maps: bool = False) -> bytes:
    import trimesh as _trimesh
    from PIL import Image

    mesh = humanoid()
    xy = mesh.vertices[:, :2]
    uv = (xy - xy.min(0)) / np.ptp(xy, axis=0)
    noise = np.random.default_rng(0).random((128, 128, 3))
    maps = {}
    if with_engine_maps:  # what Hunyuan3D / Tripo / Meshy deliver: real normal and metal-rough maps
        maps = {"normalTexture": Image.new("RGB", (64, 64), (128, 128, 255)),
                "metallicRoughnessTexture": Image.new("RGB", (64, 64), (0, 77, 200))}
    material = _trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray((noise * 255).astype(np.uint8)), metallicFactor=1.0, roughnessFactor=1.0, **maps)
    mesh.visual = _trimesh.visual.TextureVisuals(uv=uv, material=material)
    return geometry.mesh_to_glb(mesh)


def test_hd_textures_keep_engine_maps_and_rig_keeps_pbr():
    import asyncio

    from PIL import Image

    from app import textures

    hd, report = asyncio.run(textures.enhance(_textured_humanoid_glb(True), "2k", 4, True, None, "cpu"))
    gltf = gltf_json(hd)
    material = gltf["materials"][0]
    assert {"normalTexture", "occlusionTexture"} <= set(material)
    mr_index = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    view = gltf["bufferViews"][gltf["images"][gltf["textures"][mr_index]["source"]]["bufferView"]]
    n = struct.unpack_from("<I", hd, 12)[0]
    png = hd[20 + n + 8 + view.get("byteOffset", 0):][:view["byteLength"]]
    pixel = Image.open(io.BytesIO(png)).convert("RGB").getpixel((5, 5))
    assert pixel[1:] == (77, 200), "the engine's metal-rough map was replaced"
    assert report.after[0] == (512, 512)

    rigged = gltf_json(rig(hd, presets=["idle"]).glb)
    material = rigged["materials"][0]
    assert "baseColorTexture" in material["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    assert {"normalTexture", "occlusionTexture"} <= set(material)
    assert len(rigged["images"]) == 4  # base colour, normal, metal-rough, estimated occlusion


def test_full_character_one_click(client):
    up = client.post("/api/jobs/upload", files={"model": ("hero.glb", _textured_humanoid_glb(), "model/gltf-binary")})
    assert up.status_code == 201, up.text
    src = wait_done(client, up.json()["id"])
    jobs = client.post(f"/api/jobs/{src['id']}/character",
                       json={"target": "2k", "animations": ["idle", "walk", "cmu_16_15"]}).json()
    assert [j["mode"] for j in jobs] == ["enhance", "rig"]
    hd, rigged = (wait_done(client, j["id"]) for j in jobs)
    assert hd["status"] == "succeeded" and rigged["status"] == "succeeded", rigged["error"]
    assert rigged["parent_id"] == hd["id"] and rigged["params"]["source_id"] == hd["id"]
    assert len(rigged["params"]["clips"]) == 3
    material = gltf_json(client.get(rigged["model_url"]).content)["materials"][0]
    assert "normalTexture" in material and "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    # re-rigging later starts from the HD model, not the original
    again = wait_done(client, client.post(f"/api/jobs/{rigged['id']}/rig", json={"animations": ["idle"]}).json()["id"])
    assert again["params"]["source_id"] == hd["id"]
    # untextured model: no texture step, straight to the rig
    plain = wait_done(client, client.post("/api/jobs/text", json={"prompt": "un personaje", "quality": "draft"}).json()[0]["id"])
    only = client.post(f"/api/jobs/{plain['id']}/character", json={}).json()
    assert [j["mode"] for j in only] == ["rig"] and wait_done(client, only[0]["id"])["status"] == "succeeded"
    assert client.post(f"/api/jobs/{rigged['id']}/character", json={}).status_code == 400


# ---------------------------------------------------------------- fingers, face, creatures


def _hand_parts(wrist):
    """T-pose left hand at `wrist`: palm down, four separate fingers along +X, thumb forward (+Z)."""
    import trimesh as _trimesh

    from app.providers.mock import _limb

    w = np.asarray(wrist, float)
    palm = _trimesh.creation.box((0.09, 0.025, 0.085))
    palm.apply_translation(w + [0.045, 0, 0])
    parts, axes = [palm], {}
    for name, z, length in (("Index", 0.03, 0.075), ("Middle", 0.01, 0.085), ("Ring", -0.01, 0.08),
                            ("Pinky", -0.03, 0.065)):
        a, b = w + [0.09, 0, z], w + [0.09 + length, 0, z * 1.15]
        parts.append(_limb(a, b, 0.0085))
        axes[name] = (a, b)
    a, b = w + [0.02, -0.003, 0.045], w + [0.075, -0.006, 0.08]
    parts.append(_limb(a, b, 0.01))
    axes["Thumb"] = (a, b)
    hand = _trimesh.util.concatenate(parts)
    v, f = _trimesh.remesh.subdivide_to_size(hand.vertices, hand.faces, 0.006)
    return _trimesh.Trimesh(v, f), axes


def handed_humanoid():
    """T-pose mock humanoid whose left mitten is replaced by a hand with separate fingers."""
    import trimesh as _trimesh

    mesh = humanoid(90)
    markers = estimate_markers(mesh)
    wrist = build_skeleton(mesh.vertices, markers, details=False)["LeftHand"]
    keep = ~(mesh.vertices[:, 0] > wrist[0] + 0.005)[mesh.faces].any(axis=1)
    body = _trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    body.remove_unreferenced_vertices()
    hand, axes = _hand_parts(wrist)
    return _trimesh.util.concatenate([body, hand]), markers, axes


def test_fingers_follow_separated_fingers_and_get_their_own_weights():
    from app.rigging.skinning import compute_skin

    mesh, markers, axes = handed_humanoid()
    sk = build_skeleton(mesh.vertices, markers)
    assert "LeftHandIndex1" in sk.template.index and "RightHandIndex1" in sk.template.index
    for name, (a, b) in axes.items():
        axis = (b - a) / np.linalg.norm(b - a)
        joints = [sk[f"LeftHand{name}{i}"] for i in (2, 3)] + [sk[f"LeftHand{name}_End"]]
        if name != "Thumb":
            joints.append(sk[f"LeftHand{name}1"])  # the thumb's first joint sits inside the palm
        for p in joints:
            assert np.linalg.norm((p - a) - ((p - a) @ axis) * axis) < (0.01 if name == "Thumb" else 0.004), name
        assert (sk[f"LeftHand{name}_End"] - a) @ axis > 0.8 * np.linalg.norm(b - a), name  # reaches the tip
    assert sk["LeftHand_End"] == pytest.approx(sk["LeftHandMiddle1"])  # the hand bone ends at the knuckles

    weights = compute_skin(mesh, sk)
    names = sk.template.deform
    v = np.asarray(mesh.vertices)
    for name, (a, b) in axes.items():
        axis, length = (b - a) / np.linalg.norm(b - a), np.linalg.norm(b - a)
        along = (v - a) @ axis
        radial = np.linalg.norm((v - a) - along[:, None] * axis, axis=1)
        finger = (along > 0.25 * length) & (along < length) & (radial < 0.012)
        own = [i for i, n in enumerate(names) if n.startswith(f"LeftHand{name}")]
        assert finger.sum() > 100 and weights[finger][:, own].sum(axis=1).mean() > 0.9, name
    fingers = [i for i, n in enumerate(names) if n.startswith("LeftHand") and n != "LeftHand"]
    forearm = (v[:, 0] > 0.3) & (v[:, 0] < sk["LeftHand"][0] - 0.02)
    assert weights[forearm][:, fingers].max() < 1e-6  # fingers never take the arm


def test_mitten_hands_get_fingers_and_degenerate_hands_fall_back():
    mesh = humanoid(45)
    markers = estimate_markers(mesh)
    sk = build_skeleton(mesh.vertices, markers)
    assert len(sk.template.names) == len(JOINTS) + 2 + 40
    for name in ("Thumb", "Index", "Pinky"):  # both mittens: four even chains + thumb, mirror images
        for j in ("1", "_End"):
            left, right = sk[f"LeftHand{name}{j}"], sk[f"RightHand{name}{j}"]
            assert np.abs(left * [-1, 1, 1] - right).max() < 0.01, (name, j)
    assert sk["LeftHandThumb_End"][2] > 0.02 and sk["LeftHandPinky1"][2] < -0.02  # thumb forward
    # wrists past the fingertips leave no hand to measure: no fingers, the rest of the rig is intact
    bad = dict(markers, wrist_l=[markers["wrist_l"][0] * 1.6, markers["wrist_l"][1]],
               wrist_r=[markers["wrist_r"][0] * 1.6, markers["wrist_r"][1]])
    plain = build_skeleton(mesh.vertices, bad)
    assert "LeftHandIndex1" not in plain.template.index and "Jaw" in plain.template.index
    result = rig(geometry.mesh_to_glb(mesh), markers=bad, presets=["fist", "wave"])
    assert len(gltf_json(result.glb)["skins"][0]["joints"]) == len(JOINTS) + 2


def test_hand_clips_and_mocap_fingers():
    from app.rigging.skeleton import FINGERS

    mesh = humanoid(45)
    sk = build_skeleton(mesh.vertices, estimate_markers(mesh))
    index = sk.template.index

    def reach(clip_name, side="Right"):  # fingertip distance to the wrist, relative to the rest pose
        clip = preset_clip(sk, clip_name)
        k = len(clip.times) // 2
        pos = forward_kinematics(sk, clip.rotations[k], clip.hips_offset[k])
        return {f: np.linalg.norm(pos[index[f"{side}Hand{f}_End"]] - pos[index[f"{side}Hand"]])
                / np.linalg.norm(sk[f"{side}Hand{f}_End"] - sk[f"{side}Hand"]) for f in FINGERS}

    fist, point, wave = reach("fist"), reach("point"), reach("wave")
    assert all(fist[f] < 0.7 for f in ("Index", "Middle", "Ring", "Pinky"))
    assert point["Index"] > 0.97 and point["Middle"] < 0.7
    assert all(wave[f] > 0.97 for f in FINGERS)  # waving opens the hand
    grab = preset_clip(sk, "grab")
    curl = [np.linalg.norm(Rotation.from_quat(grab.rotations[i, index["RightHandMiddle2"]]).as_rotvec())
            for i in range(len(grab.times))]
    assert max(curl) > np.radians(80) and min(curl) < np.radians(5)  # opens, then closes on the object

    # mocap without finger channels: fingers and jaw stay in the rest pose
    clip = retarget_bvh(sk, make_bvh(sk))
    still = [index[n] for n in sk.template.names if "Hand" in n and n[-1].isdigit() or n == "Jaw"]
    assert np.allclose(np.abs(clip.rotations[:, still, 3]), 1.0)
    # a full Mixamo-style finger chain is followed
    bent = retarget_bvh(sk, make_bvh(sk, joints=sk.template.joints, bend={"LeftHandIndex2": 60}))
    angle = np.linalg.norm(Rotation.from_quat(bent.rotations[1, index["LeftHandIndex2"]]).as_rotvec())
    assert angle == pytest.approx(np.radians(60), abs=np.radians(3))
    assert np.allclose(np.abs(bent.rotations[:, index["LeftHandMiddle2"], 3]), 1.0)


@pytest.mark.parametrize("name,expected", [
    ("mixamorig:LeftHandIndex1", "LeftHandIndex1"), ("RightHandThumb3", "RightHandThumb3"),
    ("l_pinky2", "LeftHandPinky2"), ("LThumb", None), ("LeftHand", None),
])
def test_finger_name_mapping(name, expected):
    from app.rigging.animations import finger_name

    assert finger_name(name) == expected


def _painted_eyes(mesh, eyes):
    import trimesh as _trimesh

    colors = np.asarray(mesh.visual.vertex_colors).copy()
    for x in eyes:
        colors[np.linalg.norm(mesh.vertices - [x, 1.655, 0.12], axis=1) < 0.03, :3] = 20
    mesh.visual = _trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    return mesh


def test_face_blendshapes_and_clips():
    import trimesh as _trimesh

    plain = rig(geometry.mesh_to_glb(humanoid()), presets=["talk", "blink", "idle"])
    gltf = gltf_json(plain.glb)
    mesh = gltf["meshes"][0]
    assert mesh["extras"]["targetNames"] == ["jawOpen", "smile"]  # no eyes to find: no blink
    assert mesh["weights"] == [0.0, 0.0] and len(mesh["primitives"][0]["targets"]) == 2
    paths = {a["name"]: [c["target"]["path"] for c in a["channels"]] for a in gltf["animations"]}
    assert "weights" in paths["talk"] and "weights" not in paths["blink"] and "weights" not in paths["idle"]
    talk = next(a for a in gltf["animations"] if a["name"] == "talk")
    channel = next(c for c in talk["channels"] if c["target"]["path"] == "weights")
    sampler = talk["samplers"][channel["sampler"]]
    assert channel["target"]["node"] == 1  # the skinned "Character" node
    assert gltf["accessors"][sampler["output"]]["count"] == 2 * gltf["accessors"][sampler["input"]]["count"]
    jaw = gltf["accessors"][mesh["primitives"][0]["targets"][0]["POSITION"]]
    assert "sparse" in jaw and jaw["sparse"]["count"] < jaw["count"] and jaw["min"][1] < -0.02  # the chin goes down
    assert "Jaw" in plain.skeleton and plain.skeleton["Jaw"][1] > plain.skeleton["Head"][1]
    _trimesh.load(io.BytesIO(plain.glb), file_type="glb")  # still a readable GLB

    # the jawOpen offsets: lower face only, never above the mouth or on the body
    sk = build_skeleton(humanoid().vertices, plain.markers)
    from app.rigging.face import analyse_face

    body = humanoid()
    offsets = analyse_face(body.vertices, body.vertex_normals, None, sk).targets(body.vertices, body.vertex_normals)
    moved = np.linalg.norm(offsets["jawOpen"], axis=1) > 1e-4
    assert moved.sum() > 20 and body.vertices[moved, 1].min() > sk["Head"][1] - 0.05
    assert body.vertices[moved, 1].max() < sk["Jaw"][1]

    eyed = rig(geometry.mesh_to_glb(_painted_eyes(humanoid(), [0.04, -0.04])), presets=["blink"])
    gltf = gltf_json(eyed.glb)
    assert gltf["meshes"][0]["extras"]["targetNames"] == ["jawOpen", "blink_L", "blink_R", "smile"]
    blink = gltf["animations"][0]
    assert blink["channels"][-1]["target"]["path"] == "weights"


def creature():
    """One-eyed monster: a big round belly, short legs and long thin arms hanging close to it,
    fused into one surface like generated meshes."""
    import trimesh as _trimesh

    from app.providers.mock import _limb

    belly = _trimesh.creation.icosphere(4, radius=1.0)
    belly.apply_scale((0.42, 0.42, 0.38))
    belly.apply_translation((0, 0.98, 0))
    head = _trimesh.creation.icosphere(3, radius=0.2)
    head.apply_translation((0, 1.5, 0.02))
    parts = [belly, head]
    for side in (-1, 1):
        parts += [_limb((0.16 * side, 0.7, 0), (0.18 * side, 0.32, 0), 0.08),
                  _limb((0.18 * side, 0.32, 0), (0.18 * side, 0.07, 0), 0.065),
                  _limb((0.3 * side, 1.28, 0), (0.55 * side, 1.02, 0), 0.06),
                  _limb((0.55 * side, 1.02, 0), (0.58 * side, 0.6, 0), 0.05),
                  _limb((0.58 * side, 0.6, 0), (0.6 * side, 0.45, 0), 0.045)]
    fused = geometry.voxelize(_trimesh.util.concatenate(parts), 96)
    fused.merge_vertices()
    _trimesh.smoothing.filter_taubin(fused, iterations=12)
    fused.apply_translation([0, -fused.vertices[:, 1].min(), 0])
    markers = {"chin": [0, 1.33], "groin": [0, 0.68], "shoulder_l": [0.33, 1.26], "elbow_l": [0.55, 1.02],
               "wrist_l": [0.58, 0.6], "knee_l": [0.18, 0.32], "ankle_l": [0.18, 0.07]}
    markers |= {k[:-1] + "r": [-x, y] for k, (x, y) in markers.items() if k.endswith("_l")}
    return fused, markers


def test_wide_belly_is_not_dragged_by_the_arms():
    from app.rigging.skinning import compute_skin

    mesh, markers = creature()
    sk = build_skeleton(mesh.vertices, markers)
    weights = compute_skin(mesh, sk)
    names = sk.template.deform
    arm = [i for i, n in enumerate(names) if ("Arm" in n or "Hand" in n)]
    trunk = [i for i, n in enumerate(names) if n in ("Hips", "Spine", "Spine1", "Spine2")]
    legs = [i for i, n in enumerate(names) if "Leg" in n]
    v = mesh.vertices
    belly = (np.abs(v[:, 0]) > 0.2) & (np.abs(v[:, 0]) < 0.45) & (v[:, 1] > 0.7) & (v[:, 1] < 1.0)
    arms = (np.abs(v[:, 0]) > 0.5) & (v[:, 1] > 0.5) & (v[:, 1] < 1.0)
    from_arms = weights[belly][:, arm].sum(axis=1)
    assert belly.sum() > 1000 and np.percentile(from_arms, 99) < 0.01 and (from_arms > 0.05).mean() < 0.005
    assert weights[arms][:, trunk].sum(axis=1).max() < 0.02  # and the arms are not held by the spine
    assert weights[belly][:, trunk].sum(axis=1).mean() > weights[belly][:, legs].sum(axis=1).mean()  # nor the legs
    assert (weights > 0).sum(axis=1).max() <= 4
