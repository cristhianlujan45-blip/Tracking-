"""In-app updates from the project's GitHub releases.

The app downloads the new installer itself and runs it silently, so updates need no browser.
Files saved by the app carry no "downloaded from the internet" mark, so Windows SmartScreen does
not stop them: only the very first install from the browser shows its warning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

from ._version import UPDATE_REPO

REPO = os.getenv("FORJA_UPDATE_REPO", UPDATE_REPO)
CHECK_EVERY_S = 3600


class UpdateError(Exception):
    """User-facing error."""


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in match.groups()) if match else None


def _asset_for_platform(assets: list[dict]) -> dict | None:
    if sys.platform == "win32":
        pattern = re.compile(r"Forja3D-Setup-.*\.exe$")
    elif sys.platform == "darwin":
        pattern = re.compile(r"\.dmg$")
    else:
        return None
    return next((a for a in assets if pattern.search(a.get("name", ""))), None)


class Updater:
    def __init__(self, current: str, folder: Path, client: httpx.AsyncClient | None = None):
        self.current = current
        self.folder = folder
        self.client = client
        self.latest: dict | None = None
        self.checked_at = 0.0
        self.task: asyncio.Task | None = None
        self.progress = 0
        self.state = "idle"  # idle | downloading | installing | error
        self.error = ""

    async def check(self, force: bool = False) -> dict:
        """What the update banner shows. Development builds ("dev") never update."""
        if parse_version(self.current) is None:
            return {"current": self.current, "available": False}
        if force or self.latest is None or time.time() - self.checked_at > CHECK_EVERY_S:
            try:
                async with self._client() as client:
                    resp = await client.get(f"https://api.github.com/repos/{REPO}/releases/latest",
                                            headers={"Accept": "application/vnd.github+json"})
                    resp.raise_for_status()
                    release = resp.json()
                self.latest = {"version": ".".join(map(str, parse_version(release.get("tag_name", "")) or ())),
                               "url": release.get("html_url"), "asset": _asset_for_platform(release.get("assets", []))}
                self.checked_at = time.time()
            except (httpx.HTTPError, ValueError) as exc:
                log.info("No se pudo comprobar si hay actualizaciones: %s", exc)
                return {"current": self.current, "available": False}
        latest = self.latest or {}
        newer = (parse_version(latest.get("version", "")) or (0,)) > parse_version(self.current)
        return {"current": self.current, "latest": latest.get("version"), "available": newer,
                "can_install": newer and sys.platform == "win32" and bool(latest.get("asset")),
                "url": latest.get("url"), "state": self.state, "progress": self.progress, "error": self.error}

    @asynccontextmanager
    async def _client(self):
        if self.client is not None:  # injected (tests): not ours to close
            yield self.client
        else:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                yield client

    def start(self, on_ready) -> None:
        """Download the installer in the background; `on_ready(path)` runs it."""
        if self.task and not self.task.done():
            return
        asset = (self.latest or {}).get("asset")
        if not asset:
            raise UpdateError("No hay un instalador para este sistema en la última versión")
        self.state, self.progress, self.error = "downloading", 0, ""
        self.task = asyncio.create_task(self._download(asset, on_ready))

    async def _download(self, asset: dict, on_ready) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / asset["name"]
        partial = target.with_suffix(".part")
        try:
            async with self._client() as client:
                async with client.stream("GET", asset["browser_download_url"]) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or asset.get("size") or 0)
                    done = 0
                    with partial.open("wb") as fh:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            fh.write(chunk)
                            done += len(chunk)
                            if total:
                                self.progress = int(100 * done / total)
            if asset.get("size") and partial.stat().st_size != asset["size"]:
                raise UpdateError("La descarga de la actualización llegó incompleta")
            partial.replace(target)
            self.state, self.progress = "installing", 100
            on_ready(target)
        except Exception as exc:  # noqa: BLE001 - shown in the banner
            log.exception("Actualización fallida")
            self.state, self.error = "error", str(exc) or exc.__class__.__name__


def run_windows_installer(installer: Path) -> None:
    """Run the Inno Setup installer silently once the app has quit, then reopen the app.

    The paths go through environment variables: cmd.exe reads batch files in the OEM code page, so
    a user name with accents written into the script would break them."""
    script = installer.with_name("actualizar.cmd")
    script.write_text("\r\n".join([
        "@echo off",
        "ping -n 4 127.0.0.1 >nul",  # ~3 s for the app to close (timeout needs a console)
        '"%F3D_SETUP%" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS',
        'start "" "%F3D_EXE%"',
        "",
    ]), encoding="ascii")
    env = {**os.environ, "F3D_SETUP": str(installer), "F3D_EXE": sys.executable}  # the packaged Forja3D.exe
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([os.environ.get("COMSPEC", "cmd.exe"), "/c", str(script)], creationflags=flags,
                     close_fds=True, env=env)
