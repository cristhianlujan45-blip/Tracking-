"""In-app updates: version check against the latest release and the installer download."""

import asyncio
import sys

import httpx

from app import updates


def fake_github(version="1.0.9", size=11):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json={
                "tag_name": f"desktop-v{version}", "html_url": "https://github.com/x/y/releases/tag/t",
                "assets": [{"name": f"Forja3D-Setup-{version}.exe", "size": size,
                            "browser_download_url": "https://example.com/setup.exe"},
                           {"name": f"Forja3D-{version}-mac-arm64.dmg", "size": 5,
                            "browser_download_url": "https://example.com/app.dmg"}]})
        return httpx.Response(200, content=b"x" * size)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_newer_release_is_offered(tmp_path):
    up = updates.Updater("1.0.6", tmp_path, client=fake_github("1.0.9"))
    s = asyncio.run(up.check())
    assert s["available"] and s["latest"] == "1.0.9" and s["current"] == "1.0.6"
    assert s["can_install"] == (sys.platform == "win32")
    same = updates.Updater("1.0.9", tmp_path, client=fake_github("1.0.9"))
    assert not asyncio.run(same.check())["available"]
    assert not asyncio.run(updates.Updater("dev", tmp_path, client=fake_github()).check())["available"]


def test_download_then_run(tmp_path, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "win32")
    up = updates.Updater("1.0.6", tmp_path, client=fake_github("1.0.9", size=11))
    ran = []

    async def scenario():
        await up.check()
        up.start(ran.append)
        await up.task

    asyncio.run(scenario())
    assert up.state == "installing" and up.progress == 100
    assert ran == [tmp_path / "Forja3D-Setup-1.0.9.exe"] and ran[0].read_bytes() == b"x" * 11


def test_incomplete_download_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(updates.sys, "platform", "win32")

    def handler(request):
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json={"tag_name": "desktop-v2.0.0", "assets": [
                {"name": "Forja3D-Setup-2.0.0.exe", "size": 100, "browser_download_url": "https://e.com/s.exe"}]})
        return httpx.Response(200, content=b"short")

    up = updates.Updater("1.0.0", tmp_path, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def scenario():
        await up.check()
        up.start(lambda p: None)
        await up.task

    asyncio.run(scenario())
    assert up.state == "error" and "incompleta" in up.error
