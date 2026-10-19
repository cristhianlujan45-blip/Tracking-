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


def convert(glb: bytes, fmt: str, blender: str, timeout: float = 300) -> bytes:
    if fmt not in FORMATS:
        raise BlenderError(f"Formato no soportado por el conversor: {fmt}")
    with tempfile.TemporaryDirectory(prefix="forja3d-") as tmp:
        src, dst = os.path.join(tmp, "in.glb"), os.path.join(tmp, f"out.{FORMATS[fmt][0]}")
        Path(src).write_bytes(glb)
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
