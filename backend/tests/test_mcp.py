"""MCP server for Claude Code / Cursor (integrations/mcp/forja3d_mcp.py): download, setup and its tools
driving a real app with the mock provider."""

import asyncio
import importlib.util
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

SCRIPT = Path(__file__).resolve().parents[2] / "integrations" / "mcp" / "forja3d_mcp.py"
TOOLS = {"create_concepts", "generate_model", "image_to_model", "get_model", "wait_for_model", "list_models",
         "search_animations", "rig_and_animate", "check_printability", "export_model", "list_printers"}


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0, max_upload_bytes=1024 * 1024)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def mcp_mod(client, monkeypatch):
    """The MCP script loaded as a module, talking to the test app (TestClient is an httpx.Client)."""
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("forja3d_mcp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "CLIENT", client)
    monkeypatch.setattr(mod, "POLL_S", 0.05)
    return mod


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- endpoints


def test_mcp_script_download(client):
    resp = client.get("/api/integrations/mcp")
    assert resp.status_code == 200
    assert 'filename="forja3d_mcp.py"' in resp.headers["content-disposition"]
    assert 'DEFAULT_URL = "http://testserver"' in resp.text
    assert "# /// script" in resp.text and '"mcp>=1.2"' in resp.text
    compile(resp.text, "forja3d_mcp.py", "exec")  # valid Python


def test_mcp_setup_installs_script_locally(client, tmp_path):
    body = client.get("/api/integrations/mcp/setup").json()
    path = Path(body["path"])
    assert body["installed"] and path == (tmp_path / "integrations" / "forja3d_mcp.py").resolve()
    assert path.read_text(encoding="utf-8") == client.get("/api/integrations/mcp").text
    assert body["claude_code"].startswith("claude mcp add forja3d -- uv run ")
    assert body["claude_code"].endswith(str(path) if " " not in str(path) else f'"{path}"')
    assert body["json_config"] == {"mcpServers": {"forja3d": {"command": "uv", "args": ["run", str(path)]}}}
    assert body["script_url"] == "http://testserver/api/integrations/mcp"


def test_server_url_discovery(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("forja3d_mcp_url", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.delenv("FORJA3D_URL", raising=False)
    monkeypatch.setattr(mod, "data_dir", lambda: tmp_path)
    assert mod.server_url() == "http://127.0.0.1:8000"
    (tmp_path / "server.json").write_text(json.dumps({"url": "http://127.0.0.1:54321/"}))
    assert mod.server_url() == "http://127.0.0.1:54321"
    monkeypatch.setenv("FORJA3D_URL", "http://otro:9000/")
    assert mod.server_url() == "http://otro:9000"


# ---------------------------------------------------------------- tools


def test_mcp_lists_all_tools(mcp_mod):
    tools = run(mcp_mod.mcp.list_tools())
    assert {t.name for t in tools} == TOOLS
    assert all(t.description for t in tools)


def test_mcp_generate_wait_list_print_export(mcp_mod, tmp_path):
    job = run(mcp_mod.generate_model("un hongo rojo", quality="draft", a_pose=False))
    assert job["status"] in ("queued", "running", "succeeded") and job["type"] == "text"

    done = run(mcp_mod.wait_for_model(job["job_id"], timeout_s=30))
    assert done["status"] == "succeeded" and not done["timed_out"]
    assert done["model_url"].startswith("http://testserver/api/jobs/")
    assert run(mcp_mod.get_model(job["job_id"]))["progress"] == 100

    models = run(mcp_mod.list_models(limit=5))["models"]
    assert [m["job_id"] for m in models] == [job["job_id"]]

    report = run(mcp_mod.check_printability(job["job_id"], size_mm=80, printer="prusa_mk4"))
    assert report["print"]["bed"] == "prusa_mk4" and "fits_bed" in report["print"]
    printers = run(mcp_mod.list_printers())
    assert "bambu_a1" in {b["id"] for b in printers["beds"]}

    out = run(mcp_mod.export_model(job["job_id"], "stl", str(tmp_path / "exports") + "/", size_mm=50))
    saved = Path(out["path"])
    assert saved.parent == tmp_path / "exports" and saved.suffix == ".stl"
    assert saved.stat().st_size == out["bytes"] > 0


def test_mcp_image_concepts_and_animations(mcp_mod, tmp_path):
    picture = tmp_path / "ref.png"
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (40, 120, 200)).save(buf, format="PNG")
    picture.write_bytes(buf.getvalue())
    job = run(mcp_mod.image_to_model(str(picture), quality="draft"))
    assert job["type"] == "image"
    assert run(mcp_mod.wait_for_model(job["job_id"], timeout_s=30))["status"] == "succeeded"

    concepts = run(mcp_mod.create_concepts("un robot simpático", style="cartoon"))["concepts"]
    assert len(concepts) == 4 and concepts[0]["image_url"].startswith("http://testserver/api/concepts/")
    from_concept = run(mcp_mod.generate_model("robot", quality="draft", concept_id=concepts[0]["concept_id"]))
    assert from_concept["type"] == "image"

    found = run(mcp_mod.search_animations("baile", body="humanoid"))
    assert found["animations"] and found["default"]
    assert run(mcp_mod.search_animations("", body="quadruped"))["animations"]


def test_mcp_rig_and_animate(mcp_mod):
    job = run(mcp_mod.generate_model("un personaje guerrero", quality="draft", a_pose=True))
    run(mcp_mod.wait_for_model(job["job_id"], timeout_s=30))
    result = run(mcp_mod.rig_and_animate(job["job_id"], ["walk"], body="humanoid"))
    rig = run(mcp_mod.wait_for_model(result["character_job_id"], timeout_s=60))
    assert rig["type"] == "rig" and rig["status"] == "succeeded", rig["error"]


def test_mcp_friendly_errors(mcp_mod, monkeypatch):
    with pytest.raises(mcp_mod.ToolError, match="404"):
        run(mcp_mod.get_model("0" * 32))
    with pytest.raises(mcp_mod.ToolError, match="No existe la imagen"):
        run(mcp_mod.image_to_model("/no/existe.png"))
    # app not running
    monkeypatch.setattr(mcp_mod, "CLIENT", None)
    monkeypatch.setenv("FORJA3D_URL", "http://127.0.0.1:9")
    with pytest.raises(mcp_mod.ToolError, match="¿Está abierta la app?"):
        run(mcp_mod.list_models())
