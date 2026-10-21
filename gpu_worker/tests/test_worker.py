import base64
import os
import sys
import io
import time

import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from worker.backends import DummyBackend
from worker.server import create_app


def b64_png(color=(200, 30, 30)):
    out = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


@pytest.fixture
def client():
    with TestClient(create_app(DummyBackend(delay=0.01), token="")) as c:
        yield c


def wait(client, task_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/tasks/{task_id}").json()
        if task["status"] in ("succeeded", "failed"):
            return task
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def test_health(client):
    body = client.get("/health").json()
    assert body["backend"] == "dummy"
    assert set(body["capabilities"]) == {"text", "image", "concepts", "multi_image", "retexture", "depth", "upscale",
                                        "views", "cutout"}


def test_images(client):
    resp = client.post("/images", json={"prompt": "una silla", "count": 3, "seed": 5})
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert len(images) == 3 and base64.b64decode(images[0])[:4] == b"\x89PNG"


def test_mesh_task_lifecycle(client):
    task_id = client.post("/tasks", json={"images": [b64_png()], "seed": 3}).json()["id"]
    task = wait(client, task_id)
    assert task["status"] == "succeeded" and task["progress"] == 100
    glb = client.get(f"/tasks/{task_id}/result")
    assert glb.headers["content-type"] == "model/gltf-binary"
    scene = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="scene")
    assert len(scene.geometry) == 1
    assert client.delete(f"/tasks/{task_id}").status_code == 204
    assert client.get(f"/tasks/{task_id}").status_code == 404


def test_texture_task(client):
    mesh_id = client.post("/tasks", json={"images": [b64_png()]}).json()["id"]
    wait(client, mesh_id)
    glb = base64.b64encode(client.get(f"/tasks/{mesh_id}/result").content).decode()
    task_id = client.post("/tasks", json={"kind": "texture", "images": [b64_png((0, 0, 255))], "mesh_glb": glb}).json()["id"]
    assert wait(client, task_id)["status"] == "succeeded"
    assert client.post("/tasks", json={"kind": "texture", "images": [b64_png()]}).status_code == 400


def test_bad_input(client):
    assert client.post("/tasks", json={"images": [base64.b64encode(b"nope").decode()]}).status_code == 400
    assert client.post("/tasks", json={"images": []}).status_code == 422
    assert client.get("/tasks/missing/result").status_code == 404


def test_failures_are_reported():
    class Broken(DummyBackend):
        def image_to_mesh(self, job, progress):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    with TestClient(create_app(Broken(delay=0), token="")) as c:
        task = wait(c, c.post("/tasks", json={"images": [b64_png()]}).json()["id"])
        assert task["status"] == "failed" and "VRAM" in task["error"]
        assert c.get(f"/tasks/{task['id']}/result").status_code == 409


def test_token_required():
    with TestClient(create_app(DummyBackend(delay=0), token="secreto")) as c:
        assert c.get("/health").status_code == 401
        assert c.get("/health", headers={"Authorization": "Bearer secreto"}).status_code == 200


def test_disabled_capabilities():
    with TestClient(create_app(DummyBackend(delay=0, texture=False, multiview=False), token="")) as c:
        assert "retexture" not in c.get("/health").json()["capabilities"]
        two = [b64_png(), b64_png((0, 255, 0))]
        task = wait(c, c.post("/tasks", json={"images": two}).json()["id"])
        assert task["status"] == "succeeded"  # extra views are ignored, not an error


def test_depth(client):
    resp = client.post("/depth", json={"image": b64_png()})
    assert resp.status_code == 200
    depth = Image.open(io.BytesIO(base64.b64decode(resp.json()["depth"])))
    assert depth.size == (64, 64) and depth.mode == "L"


def test_upscale(client):
    resp = client.post("/upscale", json={"image": b64_png(), "scale": 4})
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(base64.b64decode(resp.json()["image"])))
    assert img.size == (256, 256)
    assert client.post("/upscale", json={"image": b64_png(), "scale": 8}).status_code == 422


def test_novel_views(client):
    resp = client.post("/views", json={"image": b64_png(), "quality": "draft"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["views"]) == {"left", "back", "right"}
    left = Image.open(io.BytesIO(base64.b64decode(body["views"]["left"])))
    assert left.mode == "RGBA" and min(left.size) >= 512
    assert body["poses"]["left"] == [90.0, -10.0] and len(body["poses"]["back"]) == 2


def test_novel_views_disabled():
    with TestClient(create_app(DummyBackend(delay=0, novel_views=False), token="")) as c:
        assert "views" not in c.get("/health").json()["capabilities"]
        assert c.post("/views", json={"image": b64_png()}).status_code == 400


def test_mesh_task_roles():
    backend = DummyBackend(delay=0)
    with TestClient(create_app(backend, token="")) as c:
        three = [b64_png(), b64_png((0, 255, 0)), b64_png((0, 0, 255))]
        task = wait(c, c.post("/tasks", json={"images": three, "roles": ["front", "left", "right"]}).json()["id"])
        assert task["status"] == "succeeded"
        assert backend.last_job.roles == ["front", "left", "right"] and len(backend.last_job.images) == 3
        assert c.post("/tasks", json={"images": three, "roles": ["front", "left"]}).status_code == 400
        assert c.post("/tasks", json={"images": three[:2], "roles": ["front", "front"]}).status_code == 400
        assert c.post("/tasks", json={"images": three[:1], "roles": ["top"]}).status_code == 422


def test_frame_centres_the_cut_out():
    from worker.backends import _frame

    img = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (150, 10, 190, 90))  # an off-centre 40×80 box
    framed = _frame(img, 320)
    assert framed.size == (320, 320)
    x0, y0, x1, y1 = framed.getchannel("A").point(lambda a: 255 if a > 128 else 0).getbbox()
    assert abs((x0 + x1) / 2 - 160) <= 2 and abs((y0 + y1) / 2 - 160) <= 2
    assert abs((y1 - y0) - 0.85 * 320) <= 3


def test_clean_cutout_drops_specks_and_fills_pinholes():
    import numpy as np
    from PIL import Image, ImageDraw

    from worker.backends import clean_cutout

    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 30, 160, 190], fill=(200, 50, 50, 255))   # the subject
    d.ellipse([95, 100, 99, 104], fill=(0, 0, 0, 0))        # a pinhole in the mask
    d.rectangle([5, 5, 8, 8], fill=(90, 90, 90, 255))       # a speck of background
    alpha = np.asarray(clean_cutout(img))[:, :, 3]
    assert alpha[6, 6] == 0 and alpha[102, 97] == 255 and alpha[110, 100] == 255


def test_model_files_are_copied_not_linked_on_windows(monkeypatch):
    """Some Windows PCs refuse the links Hugging Face's cache makes (WinError 1314)."""
    import types

    from worker import __main__ as entry

    fake = types.ModuleType("huggingface_hub.file_download")
    fake.are_symlinks_supported = lambda *a, **k: True
    package = types.ModuleType("huggingface_hub")
    package.file_download = fake
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", fake)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.setattr(entry.os, "name", "nt")
    entry.copy_instead_of_links()
    monkeypatch.setattr(entry.os, "name", "posix")
    assert fake.are_symlinks_supported() is False and os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
