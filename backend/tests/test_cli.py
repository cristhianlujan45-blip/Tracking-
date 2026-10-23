"""Command line (integrations/cli/forja3d.py) driving a real app with the mock provider."""

import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SCRIPT = Path(__file__).resolve().parents[2] / "integrations" / "cli" / "forja3d.py"


@pytest.fixture
def cli(tmp_path, monkeypatch):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0)
    spec = importlib.util.spec_from_file_location("forja3d_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with TestClient(create_app(settings)) as client:
        class Resp(io.BytesIO):
            def __init__(self, r):
                super().__init__(r.content)
                self.headers = r.headers

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            url = req.full_url.replace("http://test", "")
            r = client.request(req.get_method(), url, content=req.data, headers=dict(req.header_items()))
            if r.status_code >= 400:
                raise HTTPError(req.full_url, r.status_code, "err", r.headers, io.BytesIO(r.content))
            return Resp(r)

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        yield lambda *argv: mod.main(["--server", "http://test", *argv])


def test_text_to_exported_file(cli, tmp_path, capsys):
    out = tmp_path / "robot.glb"
    assert cli("text", "un robot", "--quality", "draft", "--faces", "3000", "--pbr", "--export", "glb", "-o", str(out)) == 0
    assert out.read_bytes()[:4] == b"glTF"
    assert str(out) in capsys.readouterr().out


def test_list_animations_and_errors(cli, capsys):
    assert cli("--json", "animations", "baile", "--limit", "2") == 0
    assert 1 <= len(json.loads(capsys.readouterr().out)) <= 2
    assert cli("get", "no-existe") == 1
    assert "404" in capsys.readouterr().err
    assert cli("image", "/no/existe.png") == 1


def test_integration_downloads(tmp_path):
    import zipfile

    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0)
    with TestClient(create_app(settings)) as client:
        z = zipfile.ZipFile(io.BytesIO(client.get("/api/integrations/unity").content))
        names = z.namelist()
        assert "com.forja3d.bridge/package.json" in names and "com.forja3d.bridge/Editor/Forja3DWindow.cs" in names
        assert b'DefaultServer = "http://testserver"' in z.read("com.forja3d.bridge/Editor/Forja3DWindow.cs")
        cli = client.get("/api/integrations/cli").text
        assert 'DEFAULT_URL = "http://testserver"' in cli
