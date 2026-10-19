"""Forja3D desktop: starts the app on a private local port and shows it in a native window
(WebView2 on Windows, WebKit on macOS). Your models and settings live in your user folder.

    python desktop/forja3d_desktop.py            # window
    python desktop/forja3d_desktop.py --server   # no window: print the address (for tests)
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "Forja3D"


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, timeout: float = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--server", action="store_true", help="run without a window")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    home = data_dir()
    os.environ.setdefault("DATA_DIR", str(home))
    if sys.platform != "darwin":
        # free AI on the NVIDIA GPU by default: the app installs it on first run and starts it by itself
        os.environ.setdefault("GEN_PROVIDER", "local")
        os.environ.setdefault("FORJA_AUTO_INSTALL", "1")
    if sys.stdout is None or sys.stderr is None:  # windowed build: no console to write to
        log = open(home / "forja3d.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    if not getattr(sys, "frozen", False):  # running from the repository
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

    import uvicorn

    from app.main import create_app

    port = args.port or free_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="info", log_config=None)
    server = uvicorn.Server(config)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_ready(url):
        raise SystemExit(f"{APP_NAME} no pudo arrancar (ver {home / 'forja3d.log'})")

    if args.server:
        print(f"{APP_NAME} listo en {url}", flush=True)
        try:
            thread.join()
        except KeyboardInterrupt:
            server.should_exit = True
        return

    try:
        import webview
    except ImportError:
        webview = None
    if webview is None:  # no native window available: use the default browser
        webbrowser.open(url)
        thread.join()
        return
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    webview.create_window(APP_NAME, url, width=1480, height=920, min_size=(1100, 700), background_color="#080A17")
    try:
        webview.start(private_mode=False, storage_path=str(home / "webview"))
    except Exception:  # noqa: BLE001 - e.g. WebView2 missing: the browser still works
        logging.exception("No se pudo abrir la ventana; se usa el navegador")
        webbrowser.open(url)
        thread.join()
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
