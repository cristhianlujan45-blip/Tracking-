"""Free AI on the user's own NVIDIA GPU, installed and started by the app itself.

One click installs a private Python with PyTorch (CUDA 12.8, RTX 50 ready), Hunyuan3D-2 and the
Forja3D GPU worker into the app's data folder, and downloads the models once. After that the worker
starts by itself whenever a generation needs it: no scripts, no keys, no tokens.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
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

log = logging.getLogger(__name__)

VERSION = 1
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"  # CUDA 12.8: first build with RTX 50 (Blackwell) kernels
HUNYUAN_ZIP = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
HUNYUAN_PYPI = "hy3dgen==2.0.2"  # fallback when GitHub is unreachable (older, without FlashVDM)
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/{asset}"
DEFAULT_PORT = 8001

# best quality that still fits 8 GB of VRAM (the worker keeps one model on the GPU at a time)
PREMIUM_ARGS = [
    "--shape-model", "tencent/Hunyuan3D-2", "--shape-subfolder", "hunyuan3d-dit-v2-0",
    "--image-model", "stabilityai/sdxl-turbo",
    "--depth-model", "depth-anything/Depth-Anything-V2-Base-hf",
    "--multiview",
]

# (label, share of the whole install)
STEPS = [
    ("Comprobando tu tarjeta gráfica", 2),
    ("Preparando el instalador", 3),
    ("Instalando Python", 5),
    ("Instalando PyTorch con CUDA (≈3 GB)", 30),
    ("Instalando Hunyuan3D-2 y el servidor de IA", 15),
    ("Comprobando que PyTorch ve tu GPU", 3),
    ("Descargando los modelos de IA (≈15 GB, solo una vez)", 40),
    ("Terminando", 2),
]


MIN_DRIVER = (570, 0)  # CUDA 12.8
MIN_FREE_GB = 25


def _driver_version(line: str) -> tuple[int, int] | None:
    """'NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, 581.57' -> (581, 57)."""
    try:
        major, minor = line.rsplit(",", 1)[1].strip().split(".")[:2]
        return int(major), int(minor)
    except (IndexError, ValueError):
        return None


def _folder_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3 if path.is_dir() else 0.0


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
               "PYTHONPATH": str(self.root), "UV_PYTHON_INSTALL_DIR": str(self.root / "python"),
               "UV_CACHE_DIR": str(self.root / "cache"), "UV_LINK_MODE": "copy"}
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
                self.step = label
                self.progress = done
                await getattr(self, f"_step{i}")(done, share)
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
        workdir = cwd or self.root
        workdir.mkdir(parents=True, exist_ok=True)  # Windows refuses to start a process in a missing folder

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
        if not self.python.is_file():
            await self._run(self._uv(), "venv", str(self.root / "venv"), "--python", "3.11", start=start)

    async def _step3(self, start, share):  # PyTorch
        await self._run(self._uv(), "pip", "install", "--python", str(self.python), "torch", "torchvision",
                        "--index-url", TORCH_INDEX, start=start)

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
        hunyuan = str(source) if (source / "setup.py").is_file() else HUNYUAN_PYPI
        worker_src = self.resources / "gpu_worker"
        shutil.copytree(worker_src / "worker", self.root / "worker", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        await self._run(self._uv(), "pip", "install", "--python", str(self.python), hunyuan,
                        "-r", str(worker_src / "requirements.txt"), start=start)

    async def _step5(self, start, share):  # torch must still be the CUDA build
        check = ("import torch,sys; ok=torch.cuda.is_available() and torch.version.cuda is not None; "
                 "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if ok else 'sin GPU'); "
                 "sys.exit(0 if ok else 3)")
        try:
            await self._run(str(self.python), "-c", check, start=start)
        except LocalAIError:
            self._say("Reinstalando PyTorch con CUDA…")
            await self._run(self._uv(), "pip", "install", "--python", str(self.python), "--reinstall-package", "torch",
                            "--reinstall-package", "torchvision", "torch", "torchvision", "--index-url", TORCH_INDEX,
                            start=start)
            await self._run(str(self.python), "-c", check, start=start)

    async def _step6(self, start, share):  # models
        await self._run(str(self.python), "-m", "worker", "--prefetch", *self.worker_args, start=start, share=share)

    async def _step7(self, start, share):  # free the download cache (several GB)
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
            if await self.healthy():  # ours, or one started by hand (iniciar.bat -SoloWorker)
                return self.url
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

    def args(self) -> list[str]:
        if not self.light_mode:
            return list(self.worker_args)
        args, skip = [], 0
        for arg in self.worker_args:
            if skip:
                skip -= 1
            elif arg in ("--shape-model", "--shape-subfolder"):
                skip = 1
            elif arg != "--multiview":
                args.append(arg)
        return args + ["--shape-model", "tencent/Hunyuan3D-2mini", "--shape-subfolder", "hunyuan3d-dit-v2-mini"]

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
