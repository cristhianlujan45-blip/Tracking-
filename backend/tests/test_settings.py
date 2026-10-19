import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_settings_roundtrip_and_hot_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("FORJA_LOCK_SETTINGS", raising=False)
    with TestClient(create_app(Settings(provider="mock", data_dir=tmp_path))) as c:
        s = c.get("/api/settings").json()
        assert s["first_run"] is True and s["active_provider"] == "mock" and s["fal_key_set"] is False
        assert {e["id"] for e in s["fal_engines"]} >= {"hunyuan3d_v3", "trellis2", "tripo_h31", "meshy6"}

        bad = c.put("/api/settings", json={"provider": "fal", "fal_key": ""})
        assert bad.status_code == 400 and "FAL_KEY" in bad.json()["detail"]

        ok = c.put("/api/settings", json={"provider": "fal", "fal_key": "abcd-1234-secret-9876", "fal_engine": "trellis2"})
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert body["active_provider"] == "fal" and body["fal_key_hint"] == "…9876" and body["first_run"] is False
        assert "abcd" not in json.dumps(body)  # the key itself is never sent back
        cfg = c.get("/api/config").json()
        assert cfg["provider"] == "fal" and cfg["default_engine"] == "trellis2"

        # an empty key field keeps the stored one
        assert c.put("/api/settings", json={"fal_key": "", "fal_engine": "meshy6"}).json()["fal_key_set"] is True
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["provider"] == "fal" and saved["fal_engine"] == "meshy6"

    # the next start uses the saved settings
    with TestClient(create_app(Settings(provider="mock", data_dir=tmp_path))) as c:
        assert c.get("/api/config").json()["provider"] == "fal"
        assert c.put("/api/settings", json={"provider": "mock"}).json()["active_provider"] == "mock"


def test_bad_saved_provider_falls_back_to_demo(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"provider": "fal", "fal_key": ""}))
    with TestClient(create_app(Settings(provider="mock", data_dir=tmp_path))) as c:
        cfg = c.get("/api/config").json()
        assert cfg["provider"] == "mock" and "FAL_KEY" in cfg["provider_error"]


def test_settings_can_be_locked(tmp_path, monkeypatch):
    monkeypatch.setenv("FORJA_LOCK_SETTINGS", "1")
    with TestClient(create_app(Settings(provider="mock", data_dir=tmp_path))) as c:
        assert c.put("/api/settings", json={"provider": "mock"}).status_code == 403
