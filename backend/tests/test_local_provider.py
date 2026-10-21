"""The app talking to a real GPU worker process (dummy backend, in-process transport)."""

import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gpu_worker"))

from worker.backends import DummyBackend  # noqa: E402
from worker.server import create_app as create_worker  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.providers.base import GenerationError  # noqa: E402
from app.providers.local import LocalProvider  # noqa: E402

def _png():
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (32, 32), (180, 40, 40)).save(out, format="PNG")
    return out.getvalue()


PNG = _png()


def make_client(tmp_path, backend=None, **backend_opts):
    worker = create_worker(backend or DummyBackend(delay=0.01, **backend_opts), token="t0k")
    provider = LocalProvider(url="http://worker", token="t0k", poll_interval=0.02,
                             transport=httpx.ASGITransport(app=worker))
    settings = Settings(provider="local", data_dir=tmp_path, mock_step_delay=0)
    return TestClient(create_app(settings, provider=provider))


def wait_done(client, job_id, timeout=90):
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
    assert set(cfg["capabilities"]) == {"text", "image", "concepts", "multi_image", "retexture", "depth", "upscale",
                                        "views", "cutout"}


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
        assert job["status"] == "failed" and "IA local" in job["error"]


def test_scene_uses_worker_depth(client):
    job = client.post("/api/jobs/scene", files={"image": ("f.png", PNG, "image/png")}, data={"kind": "relief"}).json()
    assert job["params"]["depth_source"] == "ia"
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job["error"]
    assert client.get(job["depth_url"]).status_code == 200


def test_enhance_uses_worker_upscaler(client):
    import io as _io

    from PIL import Image as _Image

    out = _io.BytesIO()
    _Image.new("RGB", (64, 48), (90, 140, 200)).save(out, format="PNG")
    scene = wait_done(client, client.post("/api/jobs/scene", files={"image": ("f.png", out.getvalue(), "image/png")},
                                          data={"kind": "scene", "resolution": 32}).json()["id"])
    job = wait_done(client, client.post(f"/api/jobs/{scene['id']}/enhance", json={"target": "2k"}).json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["params"]["method"] == "IA local (Real-ESRGAN)" and job["params"]["sizes"] == [[256, 192]]


def test_bare_shapes_are_painted_from_the_input_image(client):
    from app.geometry import load_scene

    job = client.post("/api/jobs/image", files={"images": ("r.png", PNG, "image/png")}, data={"quality": "draft"}).json()[0]
    job = wait_done(client, job["id"])
    assert job["status"] == "succeeded", job
    scene = load_scene(client.get(job["model_url"]).content)  # COLOR_0 is linear: read back as sRGB
    mesh = next(iter(scene.geometry.values()))
    assert mesh.visual.kind == "texture"  # baked into a real UV texture
    colors = mesh.visual.to_color().vertex_colors[:, :3].astype(int)
    z = mesh.vertices[:, 2]
    front = z > (z.mean() + z.max()) / 2  # the sides now come from the generated views
    assert colors[front].mean(0)[0] > 150 and colors[front].mean(0)[1] < 90  # the red of the picture


def _front_and_side_colours(glb):
    from app.geometry import load_scene

    mesh = next(iter(load_scene(glb).geometry.values()))
    colors = mesh.visual.to_color().vertex_colors[:, :3].astype(int)
    v = mesh.vertices - mesh.vertices.mean(0)
    reach = np.abs(v).max(0)
    return colors[v[:, 2] > 0.6 * reach[2]].mean(0), colors[v[:, 0] > 0.6 * reach[0]].mean(0)


def test_single_picture_gets_ai_side_and_back_views(tmp_path):
    backend = DummyBackend(delay=0.01)
    with make_client(tmp_path, backend=backend) as c:
        job = c.post("/api/jobs/image", files={"images": ("r.png", PNG, "image/png")}, data={"quality": "draft"}).json()[0]
        job = wait_done(c, job["id"])
        assert job["status"] == "succeeded", job["error"]
        # the shape is built from the front and the square side views; the 30°-off back only paints
        assert backend.last_job.roles == ["front", "left", "right"] and len(backend.last_job.images) == 3
        assert all((tmp_path / "files" / job["id"] / f"view_{r}.png").is_file() for r in ("left", "back", "right"))
        front, side = _front_and_side_colours(c.get(job["model_url"]).content)
        assert front[0] > 150 and front[2] < 90  # the picture's red
        assert side[2] > front[2] + 30  # the generated (blue-tinted) side view painted the side

        text = wait_done(c, c.post("/api/jobs/text", json={"prompt": "una roca", "quality": "draft"}).json()[0]["id"])
        assert text["status"] == "succeeded", text["error"]
        assert backend.last_job.roles == ["front", "left", "right"]  # also after the reference image


def test_without_views_a_single_picture_goes_alone(tmp_path):
    backend = DummyBackend(delay=0.01, novel_views=False)
    with make_client(tmp_path, backend=backend) as c:
        job = c.post("/api/jobs/image", files={"images": ("r.png", PNG, "image/png")}, data={"quality": "draft"}).json()[0]
        assert wait_done(c, job["id"])["status"] == "succeeded"
        assert backend.last_job.roles is None and len(backend.last_job.images) == 1


def test_failed_views_fall_back_to_the_single_picture(tmp_path):
    class NoViews(DummyBackend):
        def novel_views(self, image, quality="standard", seed=0):
            raise RuntimeError("CUDA out of memory")

    backend = NoViews(delay=0.01)
    with make_client(tmp_path, backend=backend) as c:
        job = c.post("/api/jobs/image", files={"images": ("r.png", PNG, "image/png")}, data={"quality": "draft"}).json()[0]
        assert wait_done(c, job["id"])["status"] == "succeeded"
        assert len(backend.last_job.images) == 1


def test_view_slots_keep_their_roles(client):
    """Front + back pictures reach the worker and the painter as front + back, not front + side."""
    files = [("images", ("b.png", PNG, "image/png")), ("images", ("f.png", PNG, "image/png"))]
    resp = client.post("/api/jobs/image", files=files, data={"roles": ["back", "front"], "quality": "draft"})
    assert resp.status_code == 201, resp.text
    job = wait_done(client, resp.json()[0]["id"])
    assert job["status"] == "succeeded" and job["params"]["roles"] == ["front", "back"]
    bad = client.post("/api/jobs/image", files=files, data={"roles": ["front", "front"]})
    assert bad.status_code == 422


class Recording(DummyBackend):
    """Remembers every shape job, view generation and cut-out the app asked for."""

    def __init__(self, **kwargs):
        super().__init__(delay=0.01, **kwargs)
        self.jobs, self.view_seeds, self.cutouts = [], [], 0

    def image_to_mesh(self, job, progress):
        self.jobs.append(job)
        return super().image_to_mesh(job, progress)

    def novel_views(self, image, quality="standard", seed=0):
        self.view_seeds.append(seed)
        return super().novel_views(image, quality, seed)

    def cutout(self, image):
        self.cutouts += 1
        return super().cutout(image)


def test_shape_variants_are_built_differently_and_one_is_kept(tmp_path):
    backend = Recording()
    with make_client(tmp_path, backend=backend) as c:
        resp = c.post("/api/jobs/image", files={"images": ("r.png", PNG, "image/png")},
                      data={"quality": "draft", "variants": 3, "seed": 40})
        assert resp.status_code == 201, resp.text
        jobs = [wait_done(c, j["id"], timeout=120) for j in resp.json()]
        assert all(j["status"] == "succeeded" for j in jobs), [j["error"] for j in jobs]
        assert len({j["params"]["group"] for j in jobs}) == 1
        assert [j["params"]["variant_label"] for j in jobs] == ["Fiel a la foto", "Forma libre", "Otras vistas"]
        assert all("auto=1" in j["thumbnail_url"] for j in jobs)  # a preview to compare them in the gallery
        assert c.get(jobs[0]["thumbnail_url"]).content[:4] == b"\x89PNG"
        # one cut-out and one default view set shared by all; the third variant imagines other views
        assert backend.cutouts == 1 and sorted(backend.view_seeds) == [0, 42]
        shapes = {tuple(job.roles or ()): len(job.images) for job in backend.jobs}
        assert shapes == {("front", "left", "right"): 3, (): 1}  # the free shape uses the photo alone
        # every variant is still painted with the generated views
        assert all((tmp_path / "files" / j["id"] / "view_back.png").is_file() for j in jobs)

        kept = c.post(f"/api/jobs/{jobs[1]['id']}/keep")
        assert kept.status_code == 200, kept.text
        assert kept.json()["params"]["group"] is None and kept.json()["params"]["variant"] == 1
        assert [j["id"] for j in c.get("/api/jobs").json()] == [jobs[1]["id"]]
        assert c.post(f"/api/jobs/{jobs[1]['id']}/keep").status_code == 400

        # three more from the kept one: never the way it was built, grouped with it
        more = c.post(f"/api/jobs/{jobs[1]['id']}/variations", json={"count": 3}).json()
        assert [m["params"]["variant"] for m in more] == [2, 3, 4]
        assert all(wait_done(c, m["id"], timeout=120)["status"] == "succeeded" for m in more)
        parent = c.get(f"/api/jobs/{jobs[1]['id']}").json()
        assert parent["params"]["group"] == more[0]["params"]["group"] == jobs[1]["id"]
        assert parent["params"]["variant_label"] == "Forma libre"
        refined = c.post(f"/api/jobs/{jobs[1]['id']}/refine").json()[0]
        assert refined["params"]["variant"] == 1 and "group" not in refined["params"]


def test_worker_crash_mid_generation_retries_in_light_mode(tmp_path):
    """The worker dying while generating (memory) restarts it with the light model and retries once."""
    import asyncio

    from app.providers.base import GenerationRequest, InputImage

    worker = create_worker(DummyBackend(delay=0.01), token="t0k")
    inner = httpx.ASGITransport(app=worker)

    class Dropping(httpx.AsyncBaseTransport):
        drops = 1

        async def handle_async_request(self, request):
            if request.url.path == "/tasks" and request.method == "POST" and self.drops:
                self.drops -= 1
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return await inner.handle_async_request(request)

    class Launcher:
        light_mode = False
        reasons = []

        async def ensure_running(self):
            return "http://worker"

        def worker_log_tail(self):
            return "CUDA error: out of memory"

        def use_light_mode(self, reason):
            self.light_mode = True
            self.reasons.append(reason)

        def stop(self):
            pass

    launcher = Launcher()
    provider = LocalProvider(url="http://worker", token="t0k", poll_interval=0.02, transport=Dropping(),
                             launcher=launcher)
    stages = []

    async def progress(pct, stage):
        stages.append(stage)

    req = GenerationRequest(mode="image", prompt=None, quality="draft", images=[InputImage(PNG)], seed=3)
    result = asyncio.run(provider.generate(req, progress))
    assert result.model_glb[:4] == b"glTF"
    assert launcher.light_mode and "RemoteProtocolError" in launcher.reasons[0] and "out of memory" in launcher.reasons[0]
    assert "La IA local se cerró: reintentando en modo ligero" in stages

    # already in light mode: the error is reported, not retried forever
    provider.transport = Dropping()
    with pytest.raises(GenerationError, match="se cerró mientras generaba"):
        asyncio.run(provider.generate(req, progress))


def test_a_hung_task_restarts_the_worker_in_light_mode(tmp_path):
    """No progress for too long (e.g. a full GPU spilling into RAM): restarted lighter and retried."""
    import asyncio

    from app.providers.base import GenerationRequest, InputImage

    class Slow(DummyBackend):
        hangs = 1

        def image_to_mesh(self, job, progress):
            if self.hangs:
                self.hangs -= 1
                progress(15, "Cargando modelo multi-vista")
                time.sleep(1.5)  # stuck, as far as the app can tell
            return super().image_to_mesh(job, progress)

    worker = create_worker(Slow(delay=0.01), token="t0k")

    class Launcher:
        light_mode = False
        stopped = 0

        async def ensure_running(self):
            return "http://worker"

        def worker_log_tail(self):
            return ""

        def use_light_mode(self, reason):
            self.light_mode, self.reason = True, reason

        def stop(self):
            self.stopped += 1

    launcher = Launcher()
    provider = LocalProvider(url="http://worker", token="t0k", poll_interval=0.05, stall_s=0.5,
                             transport=httpx.ASGITransport(app=worker), launcher=launcher)

    async def progress(pct, stage):
        pass

    req = GenerationRequest(mode="image", prompt=None, quality="draft", images=[InputImage(PNG)], seed=1)
    result = asyncio.run(provider.generate(req, progress))
    assert result.model_glb[:4] == b"glTF" and launcher.light_mode and "no avanza" in launcher.reason
