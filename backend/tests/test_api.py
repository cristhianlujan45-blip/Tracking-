import asyncio
import base64
import io
import time
import zipfile

import numpy as np
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


def wait_done(client, job_id, timeout=60):
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
    assert set(cfg["export_formats"]) >= {"glb", "obj", "stl", "3mf", "ply", "3mf_color", "stl_colors"}


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


def test_enhance_textures_to_hd_with_pbr(client):
    scene = wait_done(client, client.post("/api/jobs/scene", files={"image": ("f.jpg", _photo(), "image/jpeg")},
                                          data={"kind": "scene", "resolution": 64}).json()["id"])
    resp = client.post(f"/api/jobs/{scene['id']}/enhance", json={"target": "2k", "pbr": True})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["mode"] == "enhance" and job["params"]["method"].startswith("CPU")
    assert max(job["params"]["sizes"][0]) == 640  # 160 px photo ×4, under the 2K cap
    gltf = __import__("json").loads(client.get(job["model_url"]).content[20:20 + int.from_bytes(
        client.get(job["model_url"]).content[12:16], "little")])
    assert "normalTexture" in gltf["materials"][0]
    assert "metallicRoughnessTexture" in gltf["materials"][0]["pbrMetallicRoughness"]

    plain = make_job(client, "una roca")  # vertex colours only
    failed = wait_done(client, client.post(f"/api/jobs/{plain['id']}/enhance", json={}).json()["id"])
    assert failed["status"] == "failed" and "texturas" in failed["error"]


def test_quad_remesh_exports_real_quads(client):
    parent = make_job(client, "un muñeco de nieve")
    job = wait_done(client, client.post(f"/api/jobs/{parent['id']}/remesh",
                                        json={"mode": "quads", "target_faces": 2000}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    data = client.get(f"/api/jobs/{job['id']}/export", params={"format": "obj"}).content
    zf = zipfile.ZipFile(io.BytesIO(data))
    obj = zf.read("model.obj").decode()
    faces = [line.split()[1:] for line in obj.splitlines() if line.startswith("f ")]
    assert faces and all(len(f) == 4 for f in faces)
    assert 1000 < len(faces) < 4000
    # with its material and texture, and the model in the viewer is textured too
    assert "mtllib model.mtl" in obj and b"map_Kd model.png" in zf.read("model.mtl")
    assert zf.read("model.png")[:4] == b"\x89PNG"
    scene = geometry.load_scene(client.get(job["model_url"]).content)
    assert all(g.visual.kind == "texture" for g in scene.geometry.values())


def test_multicolor_print_exports(client):
    import xml.etree.ElementTree as ET

    job = make_job(client, "un muñeco de nieve")  # white body, orange nose, coloured scarf in demo mode
    data = client.get(f"/api/jobs/{job['id']}/export", params={"format": "3mf_color", "colors": 3, "size_mm": 80}).content
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = ET.fromstring(zf.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    bases = root.findall(".//m:basematerials/m:base", ns)
    tris = root.findall(".//m:triangle", ns)
    assert 1 <= len(bases) <= 3 and len(tris) > 100
    assert {t.get("paint_color") for t in tris} <= {"4", "8", "0C"}
    zs = [float(v.get("z")) for v in root.findall(".//m:vertex", ns)]
    assert max(zs) - min(zs) == pytest.approx(80, rel=1e-3)
    # the plain 3MF still loads in trimesh as geometry
    assert len(trimesh.load(io.BytesIO(data), file_type="3mf", force="mesh").faces) == len(tris)

    parts = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job['id']}/export",
                                                  params={"format": "stl_colors", "colors": 3}).content))
    total = sum(len(trimesh.load(io.BytesIO(parts.read(n)), file_type="stl").faces) for n in parts.namelist())
    assert total == len(tris) and all(n.startswith("color_") for n in parts.namelist())


def test_paint_code_encoding():
    from app.geometry import _paint_code

    assert [_paint_code(i) for i in range(6)] == ["4", "8", "0C", "1C", "2C", "3C"]


def test_character_sheet_split(client):
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (900, 420), (27, 28, 30))
    d = ImageDraw.Draw(sheet)
    # front (wide), side (narrow), back (wide): blue figures with a dark shaded belly
    for x0, w in ((40, 200), (330, 110), (560, 200)):
        d.rounded_rectangle([x0, 40, x0 + w, 380], radius=40, fill=(70, 120, 140))
        d.ellipse([x0 + w // 4, 200, x0 + 3 * w // 4, 300], fill=(45, 60, 70))
    d.rectangle([20, 395, 120, 405], fill=(200, 200, 200))  # a caption: too small to be a view
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    resp = client.post("/api/sheet/split", files={"image": ("sheet.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200, resp.text
    views = resp.json()["views"]
    assert [v["role"] for v in views] == ["front", "left", "back"]
    img = Image.open(io.BytesIO(base64.b64decode(views[0]["url"].split(",", 1)[1])))
    assert img.mode == "RGBA"
    alpha = np.asarray(img.getchannel("A"))
    assert alpha[0, 0] == 0 and alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255  # cut out, belly kept
    blank = io.BytesIO()
    Image.new("RGB", (200, 200), (27, 28, 30)).save(blank, format="PNG")
    assert client.post("/api/sheet/split", files={"image": ("b.png", blank.getvalue(), "image/png")}).status_code == 422


def test_print_report_bed_split_and_plate(client):
    import xml.etree.ElementTree as ET

    job = make_job(client, "un muñeco de nieve")
    report = client.get(f"/api/jobs/{job['id']}/analysis", params={"size_mm": 400, "bed": "bambu_a1mini"}).json()["print"]
    assert report["size_mm"][2] == 400 and report["fits_bed"] is False and report["bed_mm"] == [180, 180, 180]
    assert {"holes", "non_manifold_edges", "watertight", "volume_cm3", "weight_g"} <= set(report)
    beds = client.get("/api/printers").json()["beds"]
    assert any(b["id"] == "prusa_mk4" for b in beds)

    split = wait_done(client, client.post(f"/api/jobs/{job['id']}/split",
                                          json={"mode": "bed", "bed": "bambu_a1mini", "size_mm": 400}).json()["id"])
    assert split["status"] == "succeeded", split["error"]
    data = client.get(f"/api/jobs/{split['id']}/export", params={"format": "3mf", "size_mm": 400, "bed": "bambu_a1mini"}).content
    model = ET.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    objects = model.findall(".//m:object", ns)
    assert len(objects) >= 3  # 400 mm tall on a 180 mm printer needs several pieces
    for obj in objects:  # every piece fits the printer
        v = np.array([[float(x.get(a)) for a in "xyz"] for x in obj.findall(".//m:vertex", ns)])
        assert np.ptp(v[:, 2]) <= 180 and sorted(np.ptp(v[:, :2], axis=0))[1] <= 180
    pieces = trimesh.load(io.BytesIO(client.get(split["model_url"]).content), file_type="glb")
    assert all(g.is_watertight for g in pieces.geometry.values())


def test_send_to_slicer(client, monkeypatch):
    from app import printing

    opened = {}
    monkeypatch.setattr(printing, "open_in_slicer", lambda path, slicer: opened.update(path=path, slicer=slicer) or "Bambu Studio")
    job = make_job(client, "un cofre")
    resp = client.post(f"/api/jobs/{job['id']}/slicer", json={"slicer": "bambu", "size_mm": 60})
    assert resp.status_code == 200, resp.text
    assert resp.json()["opened_with"] == "Bambu Studio" and opened["slicer"] == "bambu"
    assert opened["path"].suffix == ".3mf" and zipfile.is_zipfile(opened["path"])


def test_glb_vertex_colours_are_linear_and_read_back_as_srgb():
    """glTF COLOR_0 is linear: viewers and Blender must show the same colour the app painted."""
    box = trimesh.creation.box()
    box.visual.vertex_colors = np.tile([47, 56, 59, 255], (len(box.vertices), 1)).astype(np.uint8)
    glb = geometry.mesh_to_glb(box)
    raw = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh").visual.vertex_colors[0][:3]
    assert list(raw) == [7, 10, 11]  # linear values in the file
    back = next(iter(geometry.load_scene(glb).geometry.values())).visual.vertex_colors[0][:3]
    assert abs(np.asarray(back, int) - [47, 56, 59]).max() <= 1
    assert list(box.visual.vertex_colors[0][:3]) == [47, 56, 59]  # the scene itself is untouched


def test_publish_kit(client):
    job = make_job(client, "un dragón de juguete")
    resp = client.get(f"/api/jobs/{job['id']}/publish", params={"size_mm": 80, "bed": "prusa_mk4"})
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert any(n.endswith(".3mf") for n in names) and any(n.endswith(".stl") for n in names)
    assert sum(n.startswith("fotos/") and n.endswith(".png") for n in names) == 4
    es = zf.read("descripcion_es.txt").decode()
    assert "un dragón de juguete" in es and "Prusa MK4" in es and "mm" in es
    stl = trimesh.load(io.BytesIO(zf.read(next(n for n in names if n.endswith(".stl")))), file_type="stl")
    assert abs(stl.extents[2] - 80) < 1  # printed at the chosen height, Z up


def test_clipboard_endpoint_answers(client):
    """Copiar diagnóstico asks the app to copy when the window blocks the browser clipboard."""
    resp = client.post("/api/clipboard", json={"text": "Forja3D diagnóstico"})
    assert resp.status_code == 200 and isinstance(resp.json()["ok"], bool)


def test_desktop_downloads_are_saved_by_the_app(client, tmp_path, monkeypatch):
    """The desktop window can mangle downloads (an error saved as a .json file): the app saves them."""
    monkeypatch.setenv("FORJA_DOWNLOADS", str(tmp_path / "dl"))
    job = make_job(client, "un cubo")
    saved = client.get(f"/api/jobs/{job['id']}/export", params={"format": "glb", "save": True}).json()["saved"]
    assert saved.endswith(".glb") and open(saved, "rb").read(4) == b"glTF"
    again = client.get(f"/api/jobs/{job['id']}/export", params={"format": "glb", "save": True}).json()["saved"]
    assert again != saved and again.endswith("(2).glb")  # never overwrites
    assert client.post("/api/open-folder", json={"url": "/etc/passwd"}).status_code == 400


def test_fbx_without_blender_downloads_it_first(client):
    """FBX/.blend for Unity, Unreal or Blender: with no Blender the app fetches it and the page waits."""
    started = []

    class Portable:
        supported = True

        def path(self):
            return None

        def start(self):
            started.append(1)

        def status(self):
            return {"state": "downloading", "progress": 12, "error": "", "size_mb": 366}

    client.app.state.blender = None
    client.app.state.portable_blender = Portable()
    job = make_job(client, "un cubo")
    resp = client.get(f"/api/jobs/{job['id']}/export", params={"format": "fbx"})
    assert resp.status_code == 202 and resp.json()["blender"]["progress"] == 12 and started
    assert "fbx" in client.get("/api/config").json()["export_formats"]


def test_portable_blender_download_is_checked_and_unpacked(tmp_path, monkeypatch):
    import asyncio
    import hashlib

    import httpx

    from app import blender

    monkeypatch.setattr(blender.shutil, "disk_usage", lambda p: type("U", (), {"free": 50 * 1024**3})())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"blender-{blender.PORTABLE_VERSION}-windows-x64/blender.exe", b"MZ")
    data = buf.getvalue()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=data))
    portable = blender.PortableBlender(tmp_path / "b", client=httpx.AsyncClient(transport=transport))

    async def run():
        portable.start()
        await portable.task

    asyncio.run(run())  # wrong checksum: refused, nothing kept
    assert portable.status()["state"] == "error" and "dañada" in portable.error and not portable.path()
    monkeypatch.setattr(blender, "PORTABLE_SHA256", hashlib.sha256(data).hexdigest())
    portable.client = httpx.AsyncClient(transport=transport)
    asyncio.run(run())
    assert portable.status()["state"] == "ready" and portable.path().endswith("blender.exe")


def test_production_options_are_stored_and_applied(client):
    """Target faces and PBR run after generation; the job keeps what was asked for."""
    body = {"prompt": "a robot", "quality": "draft", "target_faces": 3000, "pbr": True}
    job = client.post("/api/jobs/text", json=body).json()[0]
    assert job["params"]["target_faces"] == 3000 and job["params"]["pbr"] is True
    assert job["params"]["topology"] == "triangle"
    done = wait_done(client, job["id"])
    assert done["status"] == "succeeded", done.get("error")
    assert client.post("/api/jobs/text", json={**body, "target_faces": 10}).status_code == 422
    assert client.post("/api/jobs/text", json={**body, "topology": "hexa"}).status_code == 422
