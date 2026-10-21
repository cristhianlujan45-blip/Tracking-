"""Free AI on the user's own NVIDIA GPU, installed and started by the app itself.

One click installs a private Python with PyTorch (CUDA 12.8, RTX 50 ready), Hunyuan3D-2 and the
Forja3D GPU worker into the app's data folder, and downloads the models once. After that the worker
starts by itself whenever a generation needs it: no scripts, no keys, no tokens.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import deque
from pathlib import Path

import httpx

from ._version import UPDATE_REPO

log = logging.getLogger(__name__)

VERSION = 1
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"  # CUDA 12.8: first build with RTX 50 (Blackwell) kernels
# pinned: the prebuilt texture rasterizer (.github/workflows/ai-wheels.yml) is compiled against it
TORCH_PKGS = ("torch==2.11.0", "torchvision==0.26.0")
WHEELS_TAG = "ai-wheels-v1"
HUNYUAN_ZIP = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
HUNYUAN_PYPI = "hy3dgen==2.0.2"  # fallback when GitHub is unreachable (older, without FlashVDM)
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/{asset}"
# python.org's own full Python for Windows (with venv and pip), used when uv cannot set up its Python
PY_NUGET = "https://api.nuget.org/v3-flatcontainer/python/3.11.9/python.3.11.9.nupkg"
PY_NUGET_SHA256 = "9283876d58c017e0e846f95b490da3bca0fc0a6ee1134b2870677cfb7eec3c67"
DEFAULT_PORT = 8001

# best quality that still fits 8 GB of VRAM (the worker keeps one model on the GPU at a time)
PREMIUM_ARGS = [
    "--shape-model", "tencent/Hunyuan3D-2", "--shape-subfolder", "hunyuan3d-dit-v2-0",
    "--image-model", "stabilityai/sdxl-turbo",
    "--depth-model", "depth-anything/Depth-Anything-V2-Base-hf",
    "--multiview", "--novel-views",
    "--chat-model", "Qwen/Qwen2.5-3B-Instruct",  # the assistant's free local text model (~6 GB on disk)
]

# (label, share of the whole install)
STEPS = [
    ("Comprobando tu tarjeta gráfica", 2),
    ("Preparando el instalador", 3),
    ("Instalando Python", 5),
    ("Instalando PyTorch con CUDA (≈3 GB)", 30),
    ("Instalando Hunyuan3D-2 y el servidor de IA", 15),
    ("Comprobando que PyTorch ve tu GPU", 3),
    ("Instalando el pintor de texturas con IA", 2),
    ("Descargando los modelos de IA (≈30 GB, solo una vez)", 38),
    ("Terminando", 2),
]


STEP_RETRIES = 2  # a failed install step is tried again by itself
RETRY_WAIT_S = 10.0

MIN_DRIVER = (570, 0)  # CUDA 12.8
MIN_FREE_GB = 45


def _driver_version(line: str) -> tuple[int, int] | None:
    """'NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, 581.57' -> (581, 57)."""
    try:
        major, minor = line.rsplit(",", 1)[1].strip().split(".")[:2]
        return int(major), int(minor)
    except (IndexError, ValueError):
        return None


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _folder_gb(path: Path) -> float:
    """Size of a folder. Links are not followed: Windows refuses to even look through some (os error 448)."""
    total = 0
    for folder, dirs, files in os.walk(path, onerror=lambda exc: None):
        dirs[:] = [d for d in dirs if not _is_link(Path(folder, d))]
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                pass
    return total / 1024**3


def ram_gb() -> tuple[float, float]:
    """(total, free) system memory in GB, for the diagnostics."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ("total", "avail", "page_total", "page_avail",
                                                           "virtual_total", "virtual_avail", "extended")]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 1024**3, status.avail / 1024**3
        page = os.sysconf("SC_PAGE_SIZE")
        return page * os.sysconf("SC_PHYS_PAGES") / 1024**3, page * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0, 0.0


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or _is_junction(path)
    except OSError:
        return True


def _drop_links(folder: Path) -> None:
    """Remove the folder links (junctions) uv leaves in its Python folder: once Windows refuses one,
    every look at it fails. Removing a link never touches what it points to."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        try:
            link = entry.is_symlink() or getattr(entry, "is_junction", lambda: False)()
        except OSError:
            link = True
        if not link:
            continue
        for remove in (os.rmdir, os.unlink):
            try:
                remove(entry.path)
                break
            except OSError:
                continue


def _is_junction(path: Path) -> bool:
    try:
        return path.is_junction()  # Python 3.12+
    except AttributeError:
        return False


class LocalAIError(Exception):
    """User-facing error."""


def _uv_asset() -> str:
    if sys.platform == "win32":
        return "uv-x86_64-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return "uv-aarch64-apple-darwin.tar.gz"
    return "uv-x86_64-unknown-linux-gnu.tar.gz"


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if not found and sys.platform == "win32":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        found = str(candidate) if candidate.is_file() else None
    return found


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class LocalAI:
    def __init__(self, home: Path, resources: Path, backend: str = "hunyuan", worker_args: list[str] | None = None,
                 python: str | None = None):
        """`home`: the app's data folder. `resources`: where the bundled gpu_worker/ lives.
        `python` and `backend="dummy"` let tests run a worker without installing anything."""
        self.root = home / "ai"
        self.resources = resources
        self.backend = backend
        self.worker_args = PREMIUM_ARGS if worker_args is None else worker_args
        self._python = python
        self.proc: subprocess.Popen | None = None
        self.port = DEFAULT_PORT
        self.task: asyncio.Task | None = None
        self.step = ""
        self.progress = 0
        self.error = ""
        self.log = deque(maxlen=40)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ paths & state

    @property
    def python(self) -> Path:
        if self._python:
            return Path(self._python)
        venv = self.root / "venv"
        return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    @property
    def marker(self) -> Path:
        return self.root / "installed.json"

    @property
    def installed(self) -> bool:
        if self._python:
            return True
        try:
            return json.loads(self.marker.read_text())["version"] >= VERSION and self.python.is_file()
        except (OSError, ValueError, KeyError):
            return False

    @property
    def installing(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def env(self) -> dict:
        models = self.root / "models"
        env = {**os.environ, "HF_HOME": str(models / "huggingface"), "TORCH_HOME": str(models / "torch"),
               "U2NET_HOME": str(models / "u2net"), "HY3DGEN_MODELS": str(models / "hy3dgen"),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1",
               "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",  # Windows pipes default to cp1252
               "PYTHONPATH": str(self.root), "UV_PYTHON_INSTALL_DIR": str(self.root / "python"),
               "UV_CACHE_DIR": str(self.root / "cache"), "UV_LINK_MODE": "copy"}
        if sys.platform == "win32":
            env["UV_PYTHON_DOWNLOADS"] = "never"  # its Python folder links can be refused: see _step2
            # model files are copied instead of linked: some PCs refuse links (WinError 1314)
            env["HF_HUB_DISABLE_SYMLINKS"] = env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env.pop("VIRTUAL_ENV", None)
        return env

    def status(self) -> dict:
        return {
            "supported": sys.platform in ("win32", "linux"),
            "gpu": bool(find_nvidia_smi()),
            "installed": self.installed,
            "installing": self.installing,
            "step": self.step,
            "progress": self.progress,
            "error": self.error,
            "running": self.running,
            "url": self.url,
            "light_mode": self.light_mode,
            "ai_textures_available": (self.root / "texturas_ia.txt").is_file(),
            "ai_textures": self.ai_textures,
            "log": list(self.log)[-12:],
        }

    def _say(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self.log.append(line[:300])

    # ------------------------------------------------------------ install

    def start_install(self) -> None:
        if self.installing:
            return
        if not find_nvidia_smi():
            raise LocalAIError("No se encontró una tarjeta NVIDIA con su driver instalado: actualiza el driver "
                               "(nvidia.com/drivers) y vuelve a intentarlo")
        self.error, self.progress = "", 0
        self.root.mkdir(parents=True, exist_ok=True)  # every step runs inside it
        self.task = asyncio.create_task(self._install())

    async def _install(self) -> None:
        done = 0
        try:
            for i, (label, share) in enumerate(STEPS):
                for attempt in range(STEP_RETRIES + 1):
                    self.step = label
                    self.progress = done
                    try:
                        await getattr(self, f"_step{i}")(done, share)
                        break
                    except LocalAIError as exc:
                        # the checks (step 0) say something to fix by hand; anything later is usually a
                        # dropped download: everything finished is reused, so trying again is cheap
                        if i == 0 or attempt == STEP_RETRIES:
                            raise
                        self._say(f"Falló ({exc}); reintentando en {RETRY_WAIT_S:.0f} s…")
                        self.step = f"{label} · reintentando ({attempt + 2}/{STEP_RETRIES + 1})"
                        await asyncio.sleep(RETRY_WAIT_S)
                done += share
            self.marker.write_text(json.dumps({"version": VERSION, "date": time.time()}))
            self.step, self.progress = "Listo", 100
            await self.ensure_running()
        except Exception as exc:  # noqa: BLE001 - shown in the app; the log keeps the details
            log.exception("Instalación de la IA local fallida")
            self.error = str(exc) or exc.__class__.__name__
            self.step = "Error"

    async def _run(self, *argv: str, start: int = 0, share: int = 0, cwd: Path | None = None) -> None:
        """Run a command, keep its output in the log; `@@ pct text` lines move the progress bar.
        A thread instead of asyncio subprocesses: works with any event loop (Windows included)."""
        self._say("$ " + " ".join(Path(a).name if i == 0 else a for i, a in enumerate(argv)))
        self.root.mkdir(parents=True, exist_ok=True)
        # only pass a working folder when one is asked for and really is a folder: Windows rejects the
        # whole command ("WinError 267") otherwise, and the install commands work from anywhere
        workdir = str(cwd) if cwd is not None and Path(cwd).is_dir() else None

        def run() -> int:
            proc = subprocess.Popen(argv, cwd=workdir, env=self.env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **_no_window())
            assert proc.stdout is not None
            buffer = b""
            while chunk := proc.stdout.read1(4096):
                buffer += chunk
                *lines, buffer = buffer.replace(b"\r", b"\n").split(b"\n")
                for raw in lines:
                    line = raw.decode("utf-8", "replace")
                    if line.startswith("@@ "):
                        pct, _, text = line[3:].partition(" ")
                        self.progress = start + int(share * float(pct) / 100)
                        self.step = text or self.step
                    self._say(line)
            return proc.wait()

        if await asyncio.to_thread(run) != 0:
            tail = " · ".join(list(self.log)[-3:])
            raise LocalAIError(f"Falló «{self.step}»: {tail}"[:600])

    async def _step0(self, start, share):  # GPU, driver, disk space and internet before downloading anything
        smi = find_nvidia_smi()
        await self._run(smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader", start=start)
        driver = _driver_version(list(self.log)[-1])
        if driver and driver < MIN_DRIVER:
            raise LocalAIError(f"Actualiza el driver de NVIDIA (nvidia.com/drivers): tienes el {driver[0]}.{driver[1]:02d} "
                               f"y la IA necesita el {MIN_DRIVER[0]} o más nuevo")
        free_gb = shutil.disk_usage(self.root).free / 1024**3
        needed = max(2.0, MIN_FREE_GB - _folder_gb(self.root))  # a retry reuses what is already downloaded
        if free_gb < needed:
            raise LocalAIError(f"Necesitas al menos {needed:.0f} GB libres en el disco para la IA "
                               f"(tienes {free_gb:.0f} GB): libera espacio y vuelve a intentarlo")
        if not await self._online():
            raise LocalAIError("Sin conexión a internet para descargar la IA: revisa tu conexión")

    async def _online(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                return (await client.head("https://huggingface.co")).status_code < 500
        except httpx.HTTPError:
            return False

    async def _step1(self, start, share):  # uv
        self.root.mkdir(parents=True, exist_ok=True)
        if self._uv():
            return
        target = self.root / "uv"
        target.mkdir(exist_ok=True)
        asset = _uv_asset()
        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            resp = await client.get(UV_URL.format(asset=asset))
            resp.raise_for_status()
        data = io.BytesIO(resp.content)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(data) as zf:
                for name in zf.namelist():
                    if Path(name).name in ("uv.exe", "uv"):
                        (target / Path(name).name).write_bytes(zf.read(name))
        else:
            with tarfile.open(fileobj=data) as tf:
                for member in tf.getmembers():
                    if Path(member.name).name == "uv" and member.isfile():
                        (target / "uv").write_bytes(tf.extractfile(member).read())
                        (target / "uv").chmod(0o755)
        if not self._uv():
            raise LocalAIError("No se pudo descargar el instalador uv")

    def _uv(self) -> str | None:
        local = self.root / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
        return str(local) if local.is_file() else None

    async def _step2(self, start, share):  # Python 3.11
        if self.python.is_file():
            return
        venv = self.root / "venv"
        if sys.platform != "win32":
            await self._run(self._uv(), "venv", str(venv), "--python", "3.11", start=start)
            return
        # On Windows uv's own Python needs a folder link that some PCs refuse ("untrusted mount
        # point", os error 448): the environment is built from a Python in a plain folder of ours
        _drop_links(self.root / "python")
        base = self._own_python() or await self._python_org()
        shutil.rmtree(venv, ignore_errors=True)
        await self._run(self._uv(), "venv", str(venv), "--python", str(base), start=start)

    def _own_python(self) -> Path | None:
        """Python 3.11 in ai/py311: python.org's, or a copy uv downloaded earlier, moved out of uv's folder."""
        target = self.root / "py311"
        try:
            if (target / "python.exe").is_file():
                return target / "python.exe"
            for folder in sorted((self.root / "python").glob("cpython-3.11.*"), reverse=True):
                if not _is_link(folder) and folder.is_dir() and (folder / "python.exe").is_file():
                    shutil.rmtree(target, ignore_errors=True)
                    shutil.move(str(folder), str(target))
                    return target / "python.exe"
        except OSError as exc:
            log.warning("No se pudo reutilizar el Python descargado: %s", exc)
        return None

    async def _python_org(self) -> Path:
        self._say("Descargando Python 3.11 (python.org)…")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
                resp = await client.get(PY_NUGET)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LocalAIError(f"No se pudo descargar Python 3.11: {exc}") from exc
        if hashlib.sha256(resp.content).hexdigest() != PY_NUGET_SHA256:
            raise LocalAIError("La descarga de Python 3.11 llegó dañada: vuelve a intentarlo")
        target = self.root / "py311"
        shutil.rmtree(target, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.startswith("tools/") and not name.endswith("/"):
                    dest = target / name[len("tools/"):]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
        return target / "python.exe"

    async def _step3(self, start, share):  # PyTorch
        watch = asyncio.ensure_future(self._watch_download(self.root / "cache", 3.2, start, share,
                                                           "Descargando PyTorch con CUDA"))
        try:
            await self._run(self._uv(), "pip", "install", "--python", str(self.python), *TORCH_PKGS,
                            "--index-url", TORCH_INDEX, start=start)
        finally:
            watch.cancel()

    async def _watch_download(self, folder: Path, expected_gb: float, start: int, share: int, label: str,
                              every_s: float = 3.0, stall_s: float = 300.0) -> None:
        """uv prints nothing while it downloads a big file: show the GB that reached its cache folder."""
        base = await asyncio.to_thread(_folder_gb, folder)
        last_gb, last_change = 0.0, time.monotonic()
        while True:
            await asyncio.sleep(every_s)
            got = max(0.0, await asyncio.to_thread(_folder_gb, folder) - base)
            if got > last_gb + 0.01:
                last_gb, last_change = got, time.monotonic()
            stalled = time.monotonic() - last_change > stall_s
            self.progress = max(self.progress, start + int(share * min(0.95, got / expected_gb)))
            self.step = f"{label}: {got:.1f} de ≈{expected_gb:g} GB" + (
                " (sin avance: revisa tu conexión a internet)" if stalled else "")

    async def _step4(self, start, share):  # Hunyuan3D-2 + worker
        vendor = self.root / "vendor"
        source = vendor / "Hunyuan3D-2-main"
        if not (source / "setup.py").is_file():
            vendor.mkdir(exist_ok=True)
            self._say("Descargando Hunyuan3D-2 (Tencent)…")
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
                    resp = await client.get(HUNYUAN_ZIP)
                    resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(vendor)
            except (httpx.HTTPError, zipfile.BadZipFile) as exc:
                self._say(f"GitHub no disponible ({exc.__class__.__name__}): se usa {HUNYUAN_PYPI} de PyPI")
        self._copy_worker()
        await self._install_packages(start)

    async def _install_packages(self, start: int = 0) -> None:
        source = self.root / "vendor" / "Hunyuan3D-2-main"
        hunyuan = str(source) if (source / "setup.py").is_file() else HUNYUAN_PYPI
        requirements = self.resources / "gpu_worker" / "requirements.txt"
        await self._run(self._uv(), "pip", "install", "--python", str(self.python), hunyuan,
                        "-r", str(requirements), *TORCH_PKGS, start=start)  # keep the CUDA torch
        (self.root / "requirements.sha").write_text(_digest(requirements), encoding="utf-8")

    def _copy_worker(self) -> None:
        shutil.copytree(self.resources / "gpu_worker" / "worker", self.root / "worker", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))

    def _drop_duplicate_models(self) -> None:
        """Older versions kept full copies of the Hunyuan repos (several GB each) in the Hugging Face cache;
        the worker now reads them from models/hy3dgen. Drop a cached repo once its own folder is there."""
        hub = self.root / "models" / "huggingface" / "hub"
        local = self.root / "models" / "hy3dgen" / "tencent"
        for repo, needed in (("Hunyuan3D-2", "hunyuan3d-dit-v2-0"), ("Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv"),
                             ("Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini")):
            cached = hub / f"models--tencent--{repo}"
            if cached.is_dir() and (local / repo / needed / "model.fp16.safetensors").is_file():
                size = _folder_gb(cached)
                shutil.rmtree(cached, ignore_errors=True)
                log.info("Liberados %.1f GB: copia repetida de %s", size, repo)

    async def _refresh(self) -> None:
        """After an app update: the worker code always matches the app, and new dependencies get installed."""
        if self._python or not self.installed or not (self.resources / "gpu_worker" / "worker").is_dir():
            return
        await asyncio.to_thread(self._drop_duplicate_models)
        self._copy_worker()
        requirements = self.resources / "gpu_worker" / "requirements.txt"
        stamp = self.root / "requirements.sha"
        current = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else ""
        if current != _digest(requirements) and self._uv():
            self.step = "Actualizando la IA local"
            try:
                await self._install_packages()
            except LocalAIError:
                log.exception("No se pudieron actualizar los paquetes de la IA local")
            finally:
                self.step = ""

    async def _step5(self, start, share):  # torch must still be the CUDA build
        check = ("import torch,sys; ok=torch.cuda.is_available() and torch.version.cuda is not None; "
                 "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if ok else 'sin GPU'); "
                 "sys.exit(0 if ok else 3)")
        try:
            await self._run(str(self.python), "-c", check, start=start)
        except LocalAIError:
            self._say("Reinstalando PyTorch con CUDA…")
            await self._run(self._uv(), "pip", "install", "--python", str(self.python), "--reinstall-package", "torch",
                            "--reinstall-package", "torchvision", *TORCH_PKGS, "--index-url", TORCH_INDEX,
                            start=start)
            await self._run(str(self.python), "-c", check, start=start)

    async def _step6(self, start, share):  # optional: prebuilt rasterizer for Hunyuan3D-Paint (AI textures)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                resp = await client.get(f"https://api.github.com/repos/{UPDATE_REPO}/releases/tags/{WHEELS_TAG}")
                resp.raise_for_status()
                asset = next(a for a in resp.json()["assets"] if a["name"].startswith("custom_rasterizer")
                             and a["name"].endswith("win_amd64.whl"))
                wheel = self.root / asset["name"]
                wheel.write_bytes((await client.get(asset["browser_download_url"])).content)
            await self._run(self._uv(), "pip", "install", "--python", str(self.python), str(wheel), start=start)
            (self.root / "texturas_ia.txt").write_text(asset["name"], encoding="utf-8")
        except (httpx.HTTPError, StopIteration, KeyError, ValueError, LocalAIError) as exc:
            self._say(f"Pintor de texturas con IA no disponible ({exc.__class__.__name__}); "
                      "se usan las texturas proyectadas desde tus imágenes")

    async def _step7(self, start, share):  # models
        await self._run(str(self.python), "-m", "worker", "--prefetch", *self.args(), start=start, share=share)

    async def _step8(self, start, share):  # free the download cache (several GB)
        if self._uv():
            try:
                await self._run(self._uv(), "cache", "clean", start=start)
            except LocalAIError:
                pass

    # ------------------------------------------------------------ run

    async def healthy(self, timeout: float = 2) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(self.url + "/health")
                return resp.json() if resp.is_success else None
        except (httpx.HTTPError, ValueError):
            return None

    def start(self) -> None:
        if self.running:
            return
        if not self.installed:
            raise LocalAIError("La IA local aún no está instalada: abre Ajustes → «Mi GPU» → Instalar (gratis)")
        if not _port_free(self.port):
            for port in range(DEFAULT_PORT + 1, DEFAULT_PORT + 50):
                if _port_free(port):
                    self.port = port
                    break
        self.root.mkdir(parents=True, exist_ok=True)
        logfile = open(self.root / "worker.log", "ab")  # noqa: SIM115 - owned by the child process
        argv = [str(self.python), "-m", "worker", "--backend", self.backend, "--port", str(self.port), *self.args()]
        log.info("Iniciando la IA local: %s", " ".join(argv))
        cwd = self.root if (self.root / "worker").is_dir() else self.resources / "gpu_worker"
        self.proc = subprocess.Popen(argv, cwd=cwd, env=self.env(), stdout=logfile, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, **_no_window())
        logfile.close()

    async def ensure_running(self, timeout: float = 600) -> str:
        """Start the worker when needed and wait until it answers. Returns its URL."""
        async with self._lock:
            health = await self.healthy()
            if health and (self._python or health.get("version") == self.worker_version()):
                return self.url  # ours, or one started by hand (iniciar.bat -SoloWorker)
            if health:  # left running by an older version of the app (e.g. across an update)
                await self._shutdown_stale()
            await self._refresh()
            self.start()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if await self.healthy():
                    return self.url
                if not self.running:
                    raise LocalAIError("La IA local se cerró al arrancar: " + self.worker_log_tail())
                await asyncio.sleep(1)
            raise LocalAIError("La IA local tarda demasiado en arrancar")

    @property
    def light_mode(self) -> bool:
        """The worker ran out of GPU memory with the full model once: start it with the light one."""
        return (self.root / "modo_ligero.txt").is_file()

    def use_light_mode(self, reason: str) -> None:
        """The worker died while generating (out of memory, most likely): from now on start it with the
        light model, like after a GPU out-of-memory error. Deleting modo_ligero.txt undoes it."""
        log.warning("La IA local se cerró generando; se pasa al modo ligero: %s", reason)
        (self.root / "modo_ligero.txt").write_text(f"La IA local se cerró generando: {reason}\n", encoding="utf-8")
        self.stop()

    @property
    def ai_textures(self) -> bool:
        """Hunyuan3D-Paint is usable (its rasterizer installed) and the user turned it on."""
        return (self.root / "texturas_ia.txt").is_file() and (self.root / "texturas_ia_activadas.txt").is_file()

    def set_ai_textures(self, enabled: bool) -> None:
        flag = self.root / "texturas_ia_activadas.txt"
        if not enabled:
            flag.unlink(missing_ok=True)
            self.stop()  # the next generation starts the worker without it
            return
        flag.write_text("1", encoding="utf-8")
        if self.installing:
            return
        self.error, self.progress = "", 0
        self.task = asyncio.create_task(self._texture_setup())

    async def _texture_setup(self) -> None:
        """Download the texture painter's models (~18 GB) with progress, before a generation needs them."""
        try:
            self.step = "Descargando el pintor de texturas con IA"
            await asyncio.to_thread(self.stop)
            await self._run(str(self.python), "-m", "worker", "--prefetch", *self.args(), start=0, share=100)
            self.step, self.progress = "Listo", 100
        except Exception as exc:  # noqa: BLE001 - shown in Settings; textures stay off
            log.exception("Descarga del pintor de texturas fallida")
            (self.root / "texturas_ia_activadas.txt").unlink(missing_ok=True)
            self.error, self.step = str(exc) or exc.__class__.__name__, "Error"

    def args(self) -> list[str]:
        extra = ["--texture"] if self.ai_textures and "--texture" not in self.worker_args else []
        if not self.light_mode:
            return list(self.worker_args) + extra
        args, skip = [], 0
        for arg in self.worker_args:
            if skip:
                skip -= 1
            elif arg in ("--shape-model", "--shape-subfolder"):
                skip = 1
            elif arg not in ("--multiview", "--novel-views"):  # both need the big multi-view model
                args.append(arg)
        return args + ["--shape-model", "tencent/Hunyuan3D-2mini", "--shape-subfolder", "hunyuan3d-dit-v2-mini"]

    def worker_version(self) -> str:
        """Version of the worker code this app ships (gpu_worker/worker/__init__.py)."""
        try:
            text = (self.resources / "gpu_worker" / "worker" / "__init__.py").read_text(encoding="utf-8")
        except OSError:
            return ""
        match = re.search(r'VERSION = "([^"]+)"', text)
        return match.group(1) if match else ""

    async def _shutdown_stale(self) -> None:
        log.info("Reiniciando una IA local de una versión anterior")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(self.url + "/shutdown")
        except httpx.HTTPError:
            pass
        for _ in range(30):  # wait for it to let go of the port and the GPU
            await asyncio.sleep(0.5)
            if not await self.healthy(1):
                break
        await asyncio.to_thread(self.stop)

    def worker_log_tail(self, lines: int = 3) -> str:
        try:
            text = (self.root / "worker.log").read_text("utf-8", errors="replace").strip().splitlines()
        except OSError:
            return ""
        return " · ".join(text[-lines:])[:500]

    def stop(self) -> None:
        if self.running:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def _no_window() -> dict:
    """No console window flashing on Windows for the background processes."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
