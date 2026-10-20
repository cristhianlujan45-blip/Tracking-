"""The built-in local AI launcher: starts the GPU worker by itself and the local provider uses it."""

import asyncio
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

    monkeypatch.setattr(la.LocalAI, "ensure_running", no_start)

    async def scenario():
        ai.start_install()
        await ai.task

    asyncio.run(scenario())
    assert ai.error == "" and ai.progress == 100 and ai.installed, (ai.error, list(ai.log))
    assert any("RTX 5060" in line for line in ai.log)
