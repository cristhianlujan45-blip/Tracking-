import base64
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
    assert set(body["capabilities"]) == {"text", "image", "concepts", "multi_image", "retexture", "depth"}


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
