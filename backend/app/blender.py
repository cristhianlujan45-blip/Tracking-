"""Optional format conversion through Blender (FBX and .blend), run as a separate process.

`BLENDER_PATH` may point to the Blender executable (blender / blender.exe) or to a Python
interpreter that has the `bpy` module installed. When unset, common install locations are tried.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("blender_convert.py")
FORMATS = {"fbx": ("fbx", "application/octet-stream"), "blend": ("blend", "application/x-blender")}

_WINDOWS_GLOBS = [
    r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
    r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
]
_OTHER = ["/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]


class BlenderError(Exception):
    """User-facing conversion error."""


# Blender 4.2 LTS for Windows, downloaded once by the app when a format needs it (FBX, .blend)
PORTABLE_VERSION = "4.2.23"
PORTABLE_URL = f"https://download.blender.org/release/Blender4.2/blender-{PORTABLE_VERSION}-windows-x64.zip"
PORTABLE_SHA256 = "82e791475779a7342424a480bdde9a20b43710da9264c60346125aa16cd910cb"
PORTABLE_MB = 366


class PortableBlender:
    """Downloads and unpacks Blender into the app's data folder, with progress, when FBX or .blend
    are asked for and no Blender is installed: nothing to install by hand."""

    def __init__(self, folder: Path, client=None):
        self.folder = Path(folder)
        self.client = client  # tests inject a transport
        self.task = None
        self.progress = 0
        self.error = ""

    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def path(self) -> str | None:
        exe = self.folder / f"blender-{PORTABLE_VERSION}-windows-x64" / "blender.exe"
        return str(exe) if exe.is_file() else None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def status(self) -> dict:
        state = "ready" if self.path() else "downloading" if self.busy else "error" if self.error else "missing"
        return {"state": state, "progress": self.progress, "error": self.error, "size_mb": PORTABLE_MB}

    def start(self) -> None:
        import asyncio

        if not self.busy and not self.path():
            self.error, self.progress = "", 0
            self.task = asyncio.ensure_future(self._install())

    async def _install(self) -> None:
        import asyncio
        import hashlib
        import zipfile

        import httpx

        self.folder.mkdir(parents=True, exist_ok=True)
        part = self.folder / "blender.zip.part"
        try:
            if shutil.disk_usage(self.folder).free < 2.5 * 1024**3:
                raise BlenderError("Para FBX/.blend la app descarga Blender (≈1 GB instalado): libera espacio en el disco")
            digest = hashlib.sha256()
            async with (self.client or httpx.AsyncClient(follow_redirects=True, timeout=120)) as client:
                async with client.stream("GET", PORTABLE_URL) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or PORTABLE_MB * 1024**2)
                    done = 0
                    with open(part, "wb") as fh:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            fh.write(chunk)
                            digest.update(chunk)
                            done += len(chunk)
                            self.progress = min(95, int(95 * done / total))
            if digest.hexdigest() != PORTABLE_SHA256:
                raise BlenderError("La descarga de Blender llegó dañada: vuelve a intentarlo")
            await asyncio.to_thread(lambda: zipfile.ZipFile(part).extractall(self.folder))
            self.progress = 100
        except Exception as exc:  # noqa: BLE001 - shown next to the download button
            self.error = str(exc) or exc.__class__.__name__
        finally:
            part.unlink(missing_ok=True)


def find_blender(configured: str = "") -> str | None:
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    found = shutil.which("blender")
    if found:
        return found
    candidates = [p for pattern in _WINDOWS_GLOBS for p in glob.glob(pattern)] + [p for p in _OTHER if Path(p).exists()]
    return sorted(candidates)[-1] if candidates else None  # newest version sorts last


def _command(blender: str, src: str, dst: str) -> list[str]:
    name = Path(blender).name.lower()
    if name.startswith("python"):  # a Python with the bpy module
        return [blender, str(SCRIPT), src, dst]
    return [blender, "-b", "--factory-startup", "--python", str(SCRIPT), "--", src, dst]


def convert(glb: bytes, fmt: str, blender: str, timeout: float = 300, source_ext: str = "glb",
            extra: dict[str, bytes] | None = None) -> bytes:
    """`glb` is the model data (GLB, or an OBJ when `source_ext` is "obj" to keep quads); `extra`
    are files it refers to (an OBJ's material and texture), written next to it."""
    if fmt not in FORMATS:
        raise BlenderError(f"Formato no soportado por el conversor: {fmt}")
    with tempfile.TemporaryDirectory(prefix="forja3d-") as tmp:
        src, dst = os.path.join(tmp, f"in.{source_ext}"), os.path.join(tmp, f"out.{FORMATS[fmt][0]}")
        Path(src).write_bytes(glb)
        for name, data in (extra or {}).items():
            Path(tmp, Path(name).name).write_bytes(data)
        try:
            proc = subprocess.run(_command(blender, src, dst), capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired as exc:
            raise BlenderError("Blender tardó demasiado en convertir el modelo") from exc
        except OSError as exc:
            raise BlenderError(f"No se pudo ejecutar Blender ({blender}): {exc}") from exc
        if proc.returncode != 0 or not Path(dst).is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise BlenderError("Blender no pudo convertir el modelo: " + " | ".join(tail)[:400])
        return Path(dst).read_bytes()


if __name__ == "__main__":  # manual check: python -m app.blender model.glb out.fbx
    exe = find_blender(os.getenv("BLENDER_PATH", ""))
    print("Blender:", exe)
    if exe and len(sys.argv) == 3:
        Path(sys.argv[2]).write_bytes(convert(Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]).suffix[1:], exe))
