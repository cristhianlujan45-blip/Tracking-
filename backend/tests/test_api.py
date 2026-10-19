import asyncio
import io
import time
import zipfile

import pytest
import trimesh
from fastapi.testclient import TestClient

from app import geometry
from app.config import Settings
from app.main import create_app
from app.providers.base import GenerationRequest
from app.providers.mock import KEYWORDS, MockProvider, pick_color, pick_kind

def _png(color=(180, 40, 40)):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(out, format="PNG")
    return out.getvalue()


PNG_1PX = _png()


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0, max_upload_bytes=1024 * 1024)
    with TestClient(create_app(settings)) as c:
        yield c


def wait_done(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def make_job(client, prompt="un hongo rojo", **opts):
    resp = client.post("/api/jobs/text", json={"prompt": prompt, "quality": "draft", **opts})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()[0]["id"])
    assert job["status"] == "succeeded", job
    return job


def load_glb(content):
    return trimesh.load(io.BytesIO(content), file_type="glb", force="scene")


# ---------------------------------------------------------------- basics


def test_health_and_config(client):
    assert client.get("/api/health").json() == {"status": "ok", "provider": "mock"}
    cfg = client.get("/api/config").json()
    assert set(cfg["capabilities"]) == {"text", "image", "multi_image", "concepts", "retexture", "depth"}
    assert {s["id"] for s in cfg["styles"]} >= {"auto", "lowpoly", "voxel", "sculpture"}
    assert set(cfg["export_formats"]) == {"glb", "obj", "stl", "3mf", "ply"}


def test_text_job_produces_valid_glb(client):
    job = make_job(client, "un árbol de navidad")
    assert job["progress"] == 100
    assert job["title"] == "un árbol de navidad"
    assert job["params"]["style"] == "auto" and isinstance(job["params"]["seed"], int)

    model = client.get(job["model_url"])
    assert model.status_code == 200
    assert model.headers["content-type"] == "model/gltf-binary"
    assert len(load_glb(model.content).geometry) >= 1
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job["id"]]


def test_variants_get_consecutive_seeds(client):
    resp = client.post("/api/jobs/text", json={"prompt": "una roca", "variants": 3, "seed": 10})
    jobs = resp.json()
    assert len(jobs) == 3
    assert [j["params"]["seed"] for j in jobs] == [10, 11, 12]
    for j in jobs:
        assert wait_done(client, j["id"])["status"] == "succeeded"


@pytest.mark.parametrize("style", ["lowpoly", "voxel", "sculpture", "cartoon"])
def test_styles(client, style):
    job = make_job(client, "una dona", style=style)
    scene = load_glb(client.get(job["model_url"]).content)
    assert len(scene.geometry) >= 1


def test_image_job(client):
    resp = client.post("/api/jobs/image", files={"images": ("cat.png", PNG_1PX, "image/png")}, data={"quality": "high"})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()[0]["id"])
    assert job["status"] == "succeeded" and job["mode"] == "image"
    assert client.get(job["image_url"]).content == PNG_1PX


def test_multi_image_job(client):
    files = [("images", (f"v{i}.png", PNG_1PX, "image/png")) for i in range(3)]
    job = client.post("/api/jobs/image", files=files).json()[0]
    assert job["mode"] == "multi_image"
    assert len(job["params"]["inputs"]) == 3
    assert wait_done(client, job["id"])["status"] == "succeeded"


def test_image_validation(client):
    bad_type = client.post("/api/jobs/image", files={"images": ("a.gif", b"GIF89a", "image/gif")})
    assert bad_type.status_code == 415
    too_big = client.post("/api/jobs/image", files={"images": ("a.png", b"x" * (1024 * 1024 + 1), "image/png")})
    assert too_big.status_code == 413
    assert client.post("/api/jobs/image", data={"quality": "draft"}).status_code == 400
    corrupt = client.post("/api/jobs/image", files={"images": ("a.png", b"\x89PNG broken", "image/png")})
    assert corrupt.status_code == 400
    five = [("images", (f"v{i}.png", PNG_1PX, "image/png")) for i in range(5)]
    assert client.post("/api/jobs/image", files=five).status_code == 400


def test_prompt_validation(client):
    assert client.post("/api/jobs/text", json={"prompt": "a"}).status_code == 422
    assert client.post("/api/jobs/text", json={"prompt": "silla", "quality": "ultra"}).status_code == 422
    assert client.post("/api/jobs/text", json={"prompt": "silla", "style": "nope"}).status_code == 422
    assert client.post("/api/jobs/text", json={"prompt": "silla", "variants": 9}).status_code == 422


# ---------------------------------------------------------------- concepts


def test_concepts_then_generate(client):
    resp = client.post("/api/concepts", json={"prompt": "un muñeco de nieve", "count": 4, "seed": 7})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["seed"] == 7 and len(body["concepts"]) == 4
    img = client.get(body["concepts"][0]["url"])
    assert img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"

    job = client.post(
        "/api/jobs/image", data={"concept_id": body["concepts"][2]["id"], "prompt": "un muñeco de nieve"}
    ).json()[0]
    assert job["params"]["concept_id"] == body["concepts"][2]["id"]
    assert wait_done(client, job["id"])["status"] == "succeeded"


def test_concept_ids_are_validated(client):
    assert client.get("/api/concepts/..%2Fapp.db").status_code == 404
    assert client.post("/api/jobs/image", data={"concept_id": "0" * 32}).status_code == 404


# ---------------------------------------------------------------- tools


def test_remesh_modes(client):
    parent = make_job(client)
    results = {}
    for mode, extra in [("decimate", {"target_faces": 500}), ("lowpoly", {"target_faces": 300}),
                        ("voxel", {"resolution": 16}), ("smooth", {})]:
        resp = client.post(f"/api/jobs/{parent['id']}/remesh", json={"mode": mode, **extra})
        assert resp.status_code == 201, resp.text
        job = wait_done(client, resp.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["parent_id"] == parent["id"] and job["mode"] == "remesh"
        results[mode] = job
    assert client.get(f"/api/jobs/{results['decimate']['id']}/analysis").json()["faces"] <= 500


def test_tools_need_a_finished_model(client):
    job = client.post("/api/jobs/text", json={"prompt": "una roca"}).json()[0]
    client.delete(f"/api/jobs/{job['id']}")
    assert client.post(f"/api/jobs/{job['id']}/remesh", json={}).status_code == 404


def test_analysis_repair_and_split(client):
    parent = make_job(client, "un muñeco de nieve")
    report = client.get(f"/api/jobs/{parent['id']}/analysis").json()
    assert report["bodies"] > 1 and report["faces"] > 0

    repaired = wait_done(client, client.post(f"/api/jobs/{parent['id']}/repair").json()["id"])
    assert repaired["status"] == "succeeded"
    assert client.get(f"/api/jobs/{repaired['id']}/analysis").json()["watertight"] is True

    split = client.post(f"/api/jobs/{repaired['id']}/split", json={"mode": "height", "parts": 3}).json()
    split = wait_done(client, split["id"])
    assert split["status"] == "succeeded", split
    assert len(load_glb(client.get(split["model_url"]).content).geometry) == 3

    comps = wait_done(client, client.post(f"/api/jobs/{parent['id']}/split", json={"mode": "components"}).json()["id"])
    assert len(load_glb(client.get(comps["model_url"]).content).geometry) == report["bodies"]


def test_repair_closes_holes():
    sphere = trimesh.creation.icosphere(3)
    sphere.update_faces(sphere.faces[:, 0] != 0)  # punch a hole
    sphere.remove_unreferenced_vertices()
    sphere.visual = trimesh.visual.ColorVisuals(sphere, vertex_colors=[200, 50, 50, 255])
    glb = geometry.mesh_to_glb(sphere)
    assert not geometry.analyze(glb).watertight
    assert geometry.analyze(geometry.repair(glb)).watertight


def test_retexture(client):
    parent = make_job(client, "un cubo")
    resp = client.post(f"/api/jobs/{parent['id']}/retexture", json={"prompt": "metal azul con rayas"})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded" and job["mode"] == "retexture"
    old = load_glb(client.get(parent["model_url"]).content).to_geometry()
    new = load_glb(client.get(job["model_url"]).content).to_geometry()
    assert len(old.faces) == len(new.faces)  # same geometry, new colors
    assert (old.visual.vertex_colors != new.visual.vertex_colors).any()


def test_refine_and_variations(client):
    parent = make_job(client, "una roca", seed=42, style="cartoon")
    refined = client.post(f"/api/jobs/{parent['id']}/refine").json()
    assert len(refined) == 1
    assert refined[0]["quality"] == "high" and refined[0]["params"]["seed"] == 42
    assert refined[0]["params"]["style"] == "cartoon" and refined[0]["parent_id"] == parent["id"]

    variations = client.post(f"/api/jobs/{parent['id']}/variations", json={"count": 2}).json()
    assert len(variations) == 2 and variations[0]["params"]["seed"] != 42
    for j in refined + variations:
        assert wait_done(client, j["id"])["status"] == "succeeded"

    remeshed = wait_done(client, client.post(f"/api/jobs/{parent['id']}/remesh", json={}).json()["id"])
    assert client.post(f"/api/jobs/{remeshed['id']}/refine").status_code == 400


@pytest.mark.parametrize("fmt", ["glb", "obj", "stl", "3mf", "ply"])
def test_export_formats(client, fmt):
    job = make_job(client, "un árbol")
    resp = client.get(f"/api/jobs/{job['id']}/export", params={"format": fmt, "height_mm": 80})
    assert resp.status_code == 200, resp.text
    assert "attachment" in resp.headers["content-disposition"]
    if fmt == "obj":
        assert "model.obj" in zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
    if fmt == "stl":
        mesh = trimesh.load(io.BytesIO(resp.content), file_type="stl")
        assert mesh.extents[2] == pytest.approx(80, rel=1e-3)  # Z-up, millimetres
        assert mesh.bounds[0][2] == pytest.approx(0, abs=1e-6)  # resting on the bed


def test_upload_model_roundtrip(client):
    stl = trimesh.creation.box((10, 20, 30)).export(file_type="stl")
    resp = client.post("/api/jobs/upload", files={"model": ("caja.stl", stl, "model/stl")})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded" and job["title"] == "caja"
    report = client.get(f"/api/jobs/{job['id']}/analysis").json()
    assert report["extents"][1] == pytest.approx(30)  # STL Z-up became glTF Y-up
    assert client.post("/api/jobs/upload", files={"model": ("x.fbx", b"x", "application/octet-stream")}).status_code == 415
    bad = wait_done(client, client.post("/api/jobs/upload", files={"model": ("x.stl", b"junk", "model/stl")}).json()["id"])
    assert bad["status"] == "failed"


# ---------------------------------------------------------------- gallery


def test_patch_thumbnail_and_delete(client):
    job = make_job(client, "una roca")
    patched = client.patch(f"/api/jobs/{job['id']}", json={"name": "Mi roca", "favorite": True}).json()
    assert patched["title"] == "Mi roca" and patched["favorite"] is True

    put = client.put(f"/api/jobs/{job['id']}/thumbnail", content=PNG_1PX, headers={"content-type": "image/png"})
    assert put.status_code == 204
    thumb_url = client.get(f"/api/jobs/{job['id']}").json()["thumbnail_url"]
    assert client.get(thumb_url).content == PNG_1PX
    bad = client.put(f"/api/jobs/{job['id']}/thumbnail", content=b"x", headers={"content-type": "text/plain"})
    assert bad.status_code == 415

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/model.glb").status_code == 404


def test_unknown_job(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_frontend_served(client):
    for page in ("/", "/share.html"):
        resp = client.get(page)
        assert resp.status_code == 200
        assert "Forja3D" in resp.text


def test_migrates_old_database(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "app.db")
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, mode TEXT NOT NULL, prompt TEXT, quality TEXT NOT NULL DEFAULT"
        " 'standard', status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, stage TEXT, error TEXT, provider"
        " TEXT NOT NULL, input_image TEXT, model_file TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.execute("INSERT INTO jobs VALUES ('old', 'text', 'vieja', 'standard', 'failed', 0, NULL, 'x', 'mock',"
                 " NULL, NULL, 1, 1)")
    conn.commit()
    conn.close()
    with TestClient(create_app(Settings(provider="mock", data_dir=tmp_path))) as c:
        job = c.get("/api/jobs/old").json()
        assert job["params"] == {} and job["favorite"] is False


# ---------------------------------------------------------------- mock provider


@pytest.mark.parametrize("kind", list(KEYWORDS))
def test_every_mock_shape_exports(kind):
    assert pick_kind(f"quiero {KEYWORDS[kind][0]} bonito", 0) == kind

    async def noop(*_):
        pass

    req = GenerationRequest(mode="text", prompt=KEYWORDS[kind][0], quality="draft")
    result = asyncio.run(MockProvider(step_delay=0).generate(req, noop))
    assert result.model_glb[:4] == b"glTF"
    assert len(load_glb(result.model_glb).geometry) >= 1
    concepts = asyncio.run(MockProvider(step_delay=0).generate_images(req, 2))
    assert len(concepts) == 2 and concepts[0] != concepts[1]


def test_color_words():
    assert tuple(pick_color("un hongo rojo de cuento")) == (0.85, 0.15, 0.12)
    assert pick_color("una silla") is None


def test_voxel_model_analyzes_as_one_solid(client):
    parent = make_job(client, "una dona")
    voxel = wait_done(client, client.post(f"/api/jobs/{parent['id']}/remesh", json={"mode": "voxel"}).json()["id"])
    report = client.get(f"/api/jobs/{voxel['id']}/analysis").json()
    assert report["bodies"] == 1 and report["watertight"] is True


def test_geometry_ops_keep_vertex_colors(client):
    job = make_job(client, "un hongo rojo")
    glb = client.get(job["model_url"]).content
    original = geometry.to_color_mesh(geometry.load_scene(glb)).visual.vertex_colors
    assert (original[:, 0] > original[:, 2] + 50).any()  # red cap survives loading
    for out in (geometry.remesh(glb, "lowpoly", 500), geometry.remesh(glb, "voxel", resolution=16), geometry.repair(glb)):
        colors = geometry.to_color_mesh(geometry.load_scene(out)).visual.vertex_colors
        assert (colors[:, 0] > colors[:, 2] + 50).any()


def test_glb_outputs_include_normals(client):
    import json as _json
    import struct

    job = make_job(client, "un cubo")
    for content in (client.get(job["model_url"]).content,
                    client.get(f"/api/jobs/{job['id']}/export", params={"format": "glb"}).content):
        length = struct.unpack("<I", content[12:16])[0]
        gltf = _json.loads(content[20:20 + length])
        assert all("NORMAL" in prim["attributes"] for mesh in gltf["meshes"] for prim in mesh["primitives"])


# ---------------------------------------------------------------- scenes, reliefs, text


def _photo():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 120), (120, 170, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 80, 160, 120), fill=(90, 140, 60))
    draw.ellipse((60, 40, 100, 85), fill=(200, 60, 40))
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


@pytest.mark.parametrize("kind", ["scene", "relief", "lithophane"])
def test_scene_kinds(client, kind):
    resp = client.post("/api/jobs/scene", files={"image": ("foto.jpg", _photo(), "image/jpeg")},
                       data={"kind": kind, "resolution": 96})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["mode"] == kind and job["image_url"]
    report = client.get(f"/api/jobs/{job['id']}/analysis").json()
    if kind == "scene":
        assert job["depth_url"] and client.get(job["depth_url"]).content[:4] == b"\x89PNG"
        scene = load_glb(client.get(job["model_url"]).content)
        assert next(iter(scene.geometry.values())).visual.kind == "texture"  # photo as texture
    else:
        assert report["watertight"] and report["printable"] and report["bodies"] == 1
    if kind == "lithophane":
        assert job["depth_url"] is None


def test_relief_prints_flat_at_requested_width(client):
    job = wait_done(client, client.post("/api/jobs/scene", files={"image": ("f.jpg", _photo(), "image/jpeg")},
                                        data={"kind": "relief", "resolution": 64}).json()["id"])
    stl = client.get(f"/api/jobs/{job['id']}/export", params={"format": "stl", "size_mm": 120, "fit": "width"})
    mesh = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert mesh.extents[0] == pytest.approx(120, rel=1e-3)
    assert mesh.extents[2] < mesh.extents[1] / 5  # thin plaque lying on the bed
    assert mesh.bounds[0][2] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("style", ["letters", "plate", "keychain"])
def test_text3d(client, style):
    resp = client.post("/api/jobs/text3d", json={"text": "Hola\nÑandú 3D", "style": style, "color": "#ff0000"})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded", job
    report = client.get(f"/api/jobs/{job['id']}/analysis").json()
    assert report["watertight"]
    assert job["depth_url"] is None  # "depth" here is letter thickness, not a depth map
    colors = load_glb(client.get(job["model_url"]).content).to_geometry().visual.vertex_colors
    assert (colors[:, :3] == [255, 0, 0]).all(axis=1).any()


def test_text3d_validation(client):
    assert client.post("/api/jobs/text3d", json={"text": ""}).status_code == 422
    assert client.post("/api/jobs/text3d", json={"text": "hola", "color": "red"}).status_code == 422
    assert client.post("/api/jobs/text3d", json={"text": "a\nb\nc\nd\ne"}).status_code == 422
    job = wait_done(client, client.post("/api/jobs/text3d", json={"text": "   "}).json()["id"])
    assert job["status"] == "failed"


def test_blender_addon_download(client):
    resp = client.get("/api/integrations/blender")
    assert resp.status_code == 200
    assert 'filename="forja3d_blender.py"' in resp.headers["content-disposition"]
    assert 'DEFAULT_SERVER = "http://testserver"' in resp.text
    compile(resp.text, "forja3d_blender.py", "exec")  # valid Python
