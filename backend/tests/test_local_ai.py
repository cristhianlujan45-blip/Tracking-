"""The built-in local AI launcher: starts the GPU worker by itself and the local provider uses it."""

import asyncio
import shutil
import socket
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import local_ai as la
from app.config import Settings
from app.main import create_app
from app.providers.base import GenerationRequest
from app.providers.local import LocalProvider

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def dummy_ai(tmp_path) -> la.LocalAI:
    ai = la.LocalAI(tmp_path, ROOT, backend="dummy", worker_args=["--dummy-delay", "0", "--multiview"], python=sys.executable)
    ai.port = free_port()
    return ai


def test_starts_the_worker_on_demand_and_the_provider_uses_it(tmp_path):
    ai = dummy_ai(tmp_path)

    async def scenario():
        provider = LocalProvider(url="http://127.0.0.1:1", launcher=ai, poll_interval=0.05)
        events = []

        async def progress(pct, stage):
            events.append(pct)

        req = GenerationRequest(mode="text", prompt="a red mushroom", quality="draft", seed=3)
        result = await provider.generate(req, progress)
        return provider, result, events

    try:
        provider, result, events = asyncio.run(scenario())
        assert ai.running and provider.url == ai.url
        assert result.model_glb[:4] == b"glTF" and events
        assert "multi_image" in provider.capabilities  # read from the worker it started
    finally:
        ai.stop()
    assert not ai.running


def test_progress_lines_and_failures(tmp_path):
    ai = la.LocalAI(tmp_path, ROOT)
    ai.root.mkdir(parents=True)

    async def scenario():
        await ai._run(sys.executable, "-c", "print('@@ 50 Descargando: algo'); print('hecho')", start=10, share=40)
        assert ai.progress == 30 and ai.step == "Descargando: algo" and ai.log[-1] == "hecho"
        with pytest.raises(la.LocalAIError, match="boom"):
            await ai._run(sys.executable, "-c", "import sys; print('boom'); sys.exit(2)")

    asyncio.run(scenario())
    assert not ai.installed  # nothing installed yet
    with pytest.raises(la.LocalAIError, match="no está instalada"):
        ai.start()


def test_install_needs_an_nvidia_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(la, "find_nvidia_smi", lambda: None)
    settings = Settings(provider="local", data_dir=tmp_path, mock_step_delay=0)
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/local-ai").json()
        assert status["installed"] is False and status["gpu"] is False and status["worker_up"] is False
        resp = client.post("/api/local-ai/install")
        assert resp.status_code == 400 and "NVIDIA" in resp.json()["detail"]
        assert client.get("/api/config").json()["local_ai"]["installed"] is False
        # generating without the AI installed explains what to do instead of a connection error
        job = client.post("/api/jobs/text", json={"prompt": "un hongo", "quality": "draft"}).json()[0]
        for _ in range(100):
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] == "failed":
                break
            asyncio.run(asyncio.sleep(0.05))
        assert "Instalar" in job["error"]


def test_install_runs_in_a_fresh_data_folder(tmp_path, monkeypatch):
    """First run on a new PC: the AI folder does not exist yet when the first step starts."""
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text(f"#!{sys.executable}\nprint('NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, 580.00')\n")
    smi.chmod(0o755)
    monkeypatch.setattr(la, "find_nvidia_smi", lambda: str(smi))
    ai = la.LocalAI(tmp_path / "nueva carpeta", ROOT)
    assert not ai.root.exists()
    venv_python = ai.python

    async def fake_step(self, start, share):
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()

    for i in range(1, len(la.STEPS)):  # downloads and installs are simulated; step 0 runs for real
        monkeypatch.setattr(la.LocalAI, f"_step{i}", fake_step)

    async def no_start(self, timeout=600):
        return self.url

    async def online(self):
        return True

    monkeypatch.setattr(la.LocalAI, "ensure_running", no_start)
    monkeypatch.setattr(la.LocalAI, "_online", online)
    monkeypatch.setattr(la.shutil, "disk_usage", lambda p: type("U", (), {"free": 300 * 1024**3})())

    async def scenario():
        ai.start_install()
        await ai.task

    asyncio.run(scenario())
    assert ai.error == "" and ai.progress == 100 and ai.installed, (ai.error, list(ai.log))
    assert any("RTX 5060" in line for line in ai.log)

    # a step that fails (a dropped download) is tried again by itself
    ai.marker.unlink()
    failures = []

    async def flaky(self, start, share):
        if len(failures) < 2:
            failures.append(1)
            raise la.LocalAIError("Falló «Descargando»: Connection reset by peer")

    monkeypatch.setattr(la.LocalAI, "_step7", flaky)
    monkeypatch.setattr(la, "RETRY_WAIT_S", 0)
    asyncio.run(scenario())
    assert ai.error == "" and ai.installed and len(failures) == 2
    assert sum("reintentando" in line for line in ai.log) == 2


def test_pre_checks_explain_old_driver_and_full_disk(tmp_path, monkeypatch):
    def fake_smi(driver):
        smi = tmp_path / f"smi-{driver}"
        smi.write_text(f"#!{sys.executable}\nprint('NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, {driver}')\n")
        smi.chmod(0o755)
        return str(smi)

    async def online(self):
        return True

    monkeypatch.setattr(la.LocalAI, "_online", online)
    ai = la.LocalAI(tmp_path / "data", ROOT)
    ai.root.mkdir(parents=True)

    monkeypatch.setattr(la, "find_nvidia_smi", lambda: fake_smi("560.94"))
    with pytest.raises(la.LocalAIError, match="driver"):
        asyncio.run(ai._step0(0, 2))

    monkeypatch.setattr(la, "find_nvidia_smi", lambda: fake_smi("581.57"))
    monkeypatch.setattr(la.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 * 1024**3})())
    with pytest.raises(la.LocalAIError, match="45 GB libres"):
        asyncio.run(ai._step0(0, 2))


def test_light_mode_after_running_out_of_memory(tmp_path):
    ai = la.LocalAI(tmp_path, ROOT)
    assert "--multiview" in ai.args() and "hunyuan3d-dit-v2-0" in ai.args()
    ai.root.mkdir(parents=True)
    (ai.root / "modo_ligero.txt").write_text("sin memoria")
    args = ai.args()
    assert "--multiview" not in args and "hunyuan3d-dit-v2-0" not in args
    assert args[-4:] == ["--shape-model", "tencent/Hunyuan3D-2mini", "--shape-subfolder", "hunyuan3d-dit-v2-mini"]


def test_diagnostics_text(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0)
    with TestClient(create_app(settings)) as client:
        text = client.get("/api/diagnostics").json()["text"]
        assert text.startswith("Forja3D ") and "IA local: instalada=False" in text and "Disco libre" in text
        assert client.get("/api/config").json()["version"]


def test_ai_textures_toggle(tmp_path, monkeypatch):
    downloads = []

    async def fake_setup(self):
        downloads.append(self.args())

    monkeypatch.setattr(la.LocalAI, "_texture_setup", fake_setup)
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0)
    with TestClient(create_app(settings)) as client:
        assert client.put("/api/local-ai/textures", params={"enabled": True}).status_code == 400  # not installed
        ai = client.app.state.local_ai
        ai.root.mkdir(parents=True, exist_ok=True)
        (ai.root / "texturas_ia.txt").write_text("custom_rasterizer.whl")
        body = client.put("/api/local-ai/textures", params={"enabled": True}).json()
        assert body["ai_textures"] and "--texture" in ai.args()
        assert downloads and "--texture" in downloads[0]  # its models are downloaded right away
        assert not client.put("/api/local-ai/textures", params={"enabled": False}).json()["ai_textures"]
        assert "--texture" not in ai.args()


def test_refresh_syncs_worker_code_and_reinstalls_changed_requirements(tmp_path, monkeypatch):
    ai = la.LocalAI(tmp_path, ROOT)
    ai.root.mkdir(parents=True)
    (ai.root / "uv").mkdir()
    (ai.root / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")).write_text("")
    monkeypatch.setattr(la.LocalAI, "installed", property(lambda self: True))
    installs = []

    async def fake_install(self, start=0):
        installs.append(1)
        (self.root / "requirements.sha").write_text(la._digest(ROOT / "gpu_worker" / "requirements.txt"))

    monkeypatch.setattr(la.LocalAI, "_install_packages", fake_install)
    asyncio.run(ai._refresh())
    assert (ai.root / "worker" / "backends.py").is_file() and installs == [1]
    asyncio.run(ai._refresh())  # nothing changed: no reinstall
    assert installs == [1]


def test_stale_worker_is_replaced(tmp_path, monkeypatch):
    """A worker left running by an older app version is shut down and the current one started."""
    ai = dummy_ai(tmp_path)
    old = la.LocalAI(tmp_path / "old", ROOT, backend="dummy", worker_args=["--dummy-delay", "0"],
                     python=sys.executable)
    old.port = ai.port
    old.start()  # an "orphan" from before an update

    async def scenario():
        await old.ensure_running()
        health = await ai.healthy()
        assert health["version"] == ai.worker_version() != ""
        monkeypatch.setattr(la.LocalAI, "worker_version", lambda self: "otra")  # the app ships a newer one
        await ai._shutdown_stale()
        assert not await ai.healthy(1)

    try:
        asyncio.run(scenario())
    finally:
        old.stop()


def test_python_from_a_plain_folder_when_windows_refuses_uvs_link(tmp_path, monkeypatch):
    """On Windows uv's own Python needs a folder link some PCs refuse (os error 448), and once one is
    left behind every look at it fails: the links are removed, the Python uv downloaded is moved to a
    plain folder (or python.org's package is used) and the environment is built from it."""
    ai = la.LocalAI(tmp_path, ROOT)
    managed = ai.root / "python" / "cpython-3.11.17-windows-x86_64-none"
    managed.mkdir(parents=True)
    (managed / "python.exe").write_text("x" * 1000)
    link = ai.root / "python" / "cpython-3.11-windows-x86_64-none"
    link.symlink_to(link, target_is_directory=True)  # a link nothing can look through
    assert la._folder_gb(ai.root) > 0  # the disk check does not trip over it
    calls = []

    async def fake_run(self, *argv, start=0, share=0, cwd=None):
        calls.append(argv)

    monkeypatch.setattr(la.LocalAI, "_run", fake_run)
    monkeypatch.setattr(la.LocalAI, "_uv", lambda self: "uv")
    monkeypatch.setattr(la.sys, "platform", "win32")
    assert ai.env()["UV_PYTHON_DOWNLOADS"] == "never" and ai.env()["HF_HUB_DISABLE_SYMLINKS"] == "1"
    asyncio.run(ai._step2(0, 0))
    own = ai.root / "py311" / "python.exe"
    assert calls == [("uv", "venv", str(ai.root / "venv"), "--python", str(own))]
    assert own.is_file() and not managed.exists() and not link.is_symlink()

    # no copy from uv at all: python.org's package
    shutil.rmtree(ai.root / "py311")
    fetched = []

    async def python_org(self):
        fetched.append(True)
        return self.root / "py311" / "python.exe"

    monkeypatch.setattr(la.LocalAI, "_python_org", python_org)
    asyncio.run(ai._step2(0, 0))
    assert fetched and calls[-1][-1] == str(own)


def test_big_downloads_show_how_much_arrived(tmp_path):
    """uv is silent while it downloads PyTorch: the bar follows the GB reaching its cache."""
    ai = la.LocalAI(tmp_path, ROOT)
    cache = ai.root / "cache"
    cache.mkdir(parents=True)

    async def scenario():
        watch = asyncio.ensure_future(ai._watch_download(cache, 0.001, 10, 30, "Descargando PyTorch con CUDA",
                                                         every_s=0.05, stall_s=0.3))
        await asyncio.sleep(0.1)
        (cache / "torch.part").write_bytes(b"x" * 600_000)
        await asyncio.sleep(0.2)
        moving = ai.step, ai.progress
        await asyncio.sleep(0.6)
        stalled = ai.step
        watch.cancel()
        return moving, stalled

    (step, progress), stalled = asyncio.run(scenario())
    assert step.startswith("Descargando PyTorch con CUDA: 0.0 de") and 10 < progress <= 10 + 30 * 0.95
    assert "sin avance" in stalled


def test_duplicate_model_copies_are_removed(tmp_path):
    """Full Hub copies of a Hunyuan repo go once the worker's own folder has the model."""
    ai = la.LocalAI(tmp_path, ROOT)
    hub = ai.root / "models" / "huggingface" / "hub"
    for repo in ("Hunyuan3D-2", "Hunyuan3D-2mv"):
        blob = hub / f"models--tencent--{repo}" / "blobs" / "x"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"0" * 1000)
    own = ai.root / "models" / "hy3dgen" / "tencent" / "Hunyuan3D-2" / "hunyuan3d-dit-v2-0"
    own.mkdir(parents=True)
    (own / "model.fp16.safetensors").write_bytes(b"1")
    ai._drop_duplicate_models()
    assert not (hub / "models--tencent--Hunyuan3D-2").exists()
    assert (hub / "models--tencent--Hunyuan3D-2mv").exists()  # its own folder is missing: kept
