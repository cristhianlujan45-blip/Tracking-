import asyncio
import io
import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.base import GenerationRequest
from app.providers.mock import KEYWORDS, MockProvider, pick_color, pick_kind

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360f8cf00000301010018dd8db00000000049454e44ae426082"
)


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0, max_upload_bytes=1024)
    with TestClient(create_app(settings)) as c:
        yield c


def wait_done(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "provider": "mock"}


def test_text_job_produces_valid_glb(client):
    resp = client.post("/api/jobs/text", json={"prompt": "un árbol de navidad", "quality": "draft"})
    assert resp.status_code == 201
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded", job
    assert job["progress"] == 100

    model = client.get(job["model_url"])
    assert model.status_code == 200
    assert model.headers["content-type"] == "model/gltf-binary"
    scene = trimesh.load(io.BytesIO(model.content), file_type="glb")
    assert len(scene.geometry) >= 1

    assert [j["id"] for j in client.get("/api/jobs").json()] == [job["id"]]


def test_image_job(client):
    resp = client.post(
        "/api/jobs/image", files={"image": ("cat.png", PNG_1PX, "image/png")}, data={"quality": "high"}
    )
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()["id"])
    assert job["status"] == "succeeded"
    assert client.get(job["image_url"]).content == PNG_1PX


def test_image_validation(client):
    bad_type = client.post("/api/jobs/image", files={"image": ("a.gif", b"GIF89a", "image/gif")})
    assert bad_type.status_code == 415
    too_big = client.post("/api/jobs/image", files={"image": ("a.png", b"x" * 2048, "image/png")})
    assert too_big.status_code == 413


def test_prompt_validation(client):
    assert client.post("/api/jobs/text", json={"prompt": "a"}).status_code == 422
    assert client.post("/api/jobs/text", json={"prompt": "silla", "quality": "ultra"}).status_code == 422


def test_delete_job(client):
    job_id = client.post("/api/jobs/text", json={"prompt": "una roca"}).json()["id"]
    wait_done(client, job_id)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/model.glb").status_code == 404


def test_unknown_job(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_frontend_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<canvas" in resp.text or "viewer" in resp.text


@pytest.mark.parametrize("kind", list(KEYWORDS))
def test_every_mock_shape_exports(kind):
    assert pick_kind(f"quiero {KEYWORDS[kind][0]} bonito", 0) == kind

    async def noop(*_):
        pass

    req = GenerationRequest(mode="text", prompt=KEYWORDS[kind][0], quality="draft")
    result = asyncio.run(MockProvider(step_delay=0).generate(req, noop))
    assert result.model_glb[:4] == b"glTF"
    assert len(trimesh.load(io.BytesIO(result.model_glb), file_type="glb").geometry) >= 1


def test_color_words():
    assert tuple(pick_color("un hongo rojo de cuento")) == (0.85, 0.15, 0.12)
    assert pick_color("una silla") is None
