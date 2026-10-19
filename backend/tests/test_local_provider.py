"""The app talking to a real GPU worker process (dummy backend, in-process transport)."""

import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gpu_worker"))

from worker.backends import DummyBackend  # noqa: E402
from worker.server import create_app as create_worker  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.providers.local import LocalProvider  # noqa: E402

def _png():
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 40, 40)).save(out, format="PNG")
    return out.getvalue()


PNG = _png()


def make_client(tmp_path, **backend_opts):
    worker = create_worker(DummyBackend(delay=0.01, **backend_opts), token="t0k")
    provider = LocalProvider(url="http://worker", token="t0k", poll_interval=0.02,
                             transport=httpx.ASGITransport(app=worker))
    settings = Settings(provider="local", data_dir=tmp_path, mock_step_delay=0)
    return TestClient(create_app(settings, provider=provider))


def wait_done(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_config_reports_worker(client):
    cfg = client.get("/api/config").json()
    assert cfg["provider"] == "local"
    assert cfg["provider_info"]["gpu"] == "ninguna (dummy)"
    assert set(cfg["capabilities"]) == {"text", "image", "concepts", "multi_image", "retexture", "depth"}


def test_text_to_3d_keeps_reference_image(client):
    job = client.post("/api/jobs/text", json={"prompt": "una silla", "quality": "draft"}).json()[0]
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job
    assert job["image_url"]  # concept generated on the worker is kept for refine
    assert client.get(job["model_url"]).content[:4] == b"glTF"
    refined = client.post(f"/api/jobs/{job['id']}/refine").json()[0]
    assert refined["mode"] == "image"
    assert wait_done(client, refined["id"])["status"] == "succeeded"


def test_image_concepts_and_retexture(client):
    concepts = client.post("/api/concepts", json={"prompt": "un barco", "count": 2}).json()["concepts"]
    assert len(concepts) == 2
    job = client.post("/api/jobs/image", files={"images": ("a.png", PNG, "image/png")}).json()[0]
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job["error"]
    retex = client.post(f"/api/jobs/{job['id']}/retexture", json={"prompt": "madera"}).json()
    assert wait_done(client, retex["id"])["status"] == "succeeded"


def test_capabilities_follow_worker(tmp_path):
    with make_client(tmp_path, texture=False, multiview=False) as c:
        caps = c.get("/api/config").json()["capabilities"]
        assert "retexture" not in caps and "multi_image" not in caps
        job = wait_done(c, c.post("/api/jobs/text", json={"prompt": "una roca"}).json()[0]["id"])
        assert c.post(f"/api/jobs/{job['id']}/retexture", json={"prompt": "metal"}).status_code == 400


def test_worker_down_gives_clear_error(tmp_path):
    provider = LocalProvider(url="http://127.0.0.1:9", poll_interval=0.01)
    with TestClient(create_app(Settings(provider="local", data_dir=tmp_path), provider=provider)) as c:
        assert c.get("/api/config").json()["provider_info"] == {}
        job = wait_done(c, c.post("/api/jobs/text", json={"prompt": "una roca"}).json()[0]["id"])
        assert job["status"] == "failed" and "worker" in job["error"]


def test_scene_uses_worker_depth(client):
    job = client.post("/api/jobs/scene", files={"image": ("f.png", PNG, "image/png")}, data={"kind": "relief"}).json()
    assert job["params"]["depth_source"] == "ia"
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job["error"]
    assert client.get(job["depth_url"]).status_code == 200
