# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.10", "httpx>=0.27"]
# ///
"""Forja3D para Claude Code, Cursor y Claude Desktop: servidor MCP (stdio) que maneja tu app Forja3D.

Crea conceptos y modelos 3D desde texto o imagen, los riggea y anima, revisa si se imprimen y exporta
los archivos a tu proyecto. Necesita la app abierta (escritorio o servidor) y `uv`:

    claude mcp add forja3d -- uv run /ruta/a/forja3d_mcp.py

La dirección de la app sale de FORJA3D_URL, del archivo server.json que escribe la app de escritorio
o, si no, de DEFAULT_URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Literal

import httpx

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp.exceptions import ToolError
except ImportError:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_URL = "http://127.0.0.1:8000"  # la app lo sustituye al descargar el script
APP_NAME = "Forja3D"
CLIENT: httpx.Client | None = None  # tests inject a client bound to the app
POLL_S = 2.0
logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per request is noise in the client

Quality = Literal["draft", "standard", "high"]
Style = Literal["auto", "realistic", "cartoon", "handpainted", "lowpoly", "voxel", "sculpture"]
Body = Literal["auto", "humanoid", "quadruped"]
ExportFormat = Literal["glb", "fbx", "blend", "obj", "stl", "3mf", "3mf_color", "stl_colors", "ply"]

mcp = MCPServer(
    APP_NAME,
    instructions=(
        "Forja3D is a local AI 3D studio. Typical flow: generate_model or image_to_model -> wait_for_model -> "
        "optionally rig_and_animate (characters) or check_printability (3D printing) -> export_model to save "
        "the file in the user's project. Jobs run in the app; ids are returned by every tool."
    ),
)


# ---------------------------------------------------------------- connection


def data_dir() -> Path:
    """Same folder the desktop app uses for its data (see desktop/forja3d_desktop.py)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME


def server_url() -> str:
    """FORJA3D_URL, else the address the running desktop app wrote, else DEFAULT_URL."""
    if os.environ.get("FORJA3D_URL"):
        return os.environ["FORJA3D_URL"].rstrip("/")
    try:
        url = json.loads((data_dir() / "server.json").read_text(encoding="utf-8")).get("url")
        if url:
            return str(url).rstrip("/")
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_URL.rstrip("/")


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    base = str(CLIENT.base_url).rstrip("/") if CLIENT else server_url()
    try:
        if CLIENT:
            resp = CLIENT.request(method, path, **kwargs)
        else:
            with httpx.Client(base_url=base, timeout=httpx.Timeout(300, connect=5)) as client:
                resp = client.request(method, path, **kwargs)
    except httpx.TransportError as exc:
        raise ToolError(
            f"No se puede conectar con Forja3D en {base}. ¿Está abierta la app? "
            "Ábrela (o arranca el servidor) o indica su dirección con la variable FORJA3D_URL."
        ) from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.reason_phrase)
        except ValueError:
            detail = resp.reason_phrase
        if isinstance(detail, list):  # validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
        raise ToolError(f"Forja3D respondió {resp.status_code}: {detail}")
    return resp


async def _call(method: str, path: str, **kwargs) -> httpx.Response:
    return await asyncio.to_thread(_request, method, path, **kwargs)


async def _json(method: str, path: str, **kwargs) -> Any:
    return (await _call(method, path, **kwargs)).json()


def _absolute(url: str | None) -> str | None:
    if not url or url.startswith(("http://", "https://")):
        return url
    base = str(CLIENT.base_url).rstrip("/") if CLIENT else server_url()
    return base + url


def _summary(job: dict) -> dict[str, Any]:
    return {
        "job_id": job["id"], "title": job["title"], "type": job["mode"], "status": job["status"],
        "progress": job["progress"], "stage": job.get("stage"), "error": job.get("error"),
        "quality": job.get("quality"), "parent_id": job.get("parent_id"),
        "model_url": _absolute(job.get("model_url")), "thumbnail_url": _absolute(job.get("thumbnail_url")),
    }


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


# ---------------------------------------------------------------- tools: create


@mcp.tool()
async def create_concepts(prompt: str, style: Style = "auto") -> dict[str, Any]:
    """Draw 4 concept images for an idea so the user can pick one before making it 3D.

    prompt: description of the object or character (English works best).
    Returns concept ids and image URLs; pass a concept_id to generate_model to turn it into 3D.
    """
    result = await _json("POST", "/api/concepts", json={"prompt": prompt, "style": style, "count": 4})
    return {"seed": result["seed"],
            "concepts": [{"concept_id": c["id"], "image_url": _absolute(c["url"])} for c in result["concepts"]]}


@mcp.tool()
async def generate_model(prompt: str, quality: Quality = "high", style: Style = "auto", a_pose: bool = False,
                         concept_id: str | None = None) -> dict[str, Any]:
    """Start generating a textured 3D model from a text description (takes seconds to a few minutes).

    prompt: description of the object or character (English works best).
    quality: draft (fast preview), standard or high.
    a_pose: true for characters that will be rigged and animated later.
    concept_id: optional id from create_concepts to build the model from that image instead.
    Returns the job; call wait_for_model with its job_id.
    """
    if concept_id:
        jobs = await _json("POST", "/api/jobs/image", data={
            "concept_id": concept_id, "prompt": prompt, "quality": quality, "style": style})
    else:
        jobs = await _json("POST", "/api/jobs/text", json={
            "prompt": prompt, "quality": quality, "style": style, "pose": "a-pose" if a_pose else "none"})
    return _summary(jobs[0])


@mcp.tool()
async def image_to_model(image_path: str, quality: Quality = "high", prompt: str | None = None,
                         style: Style = "auto") -> dict[str, Any]:
    """Start generating a 3D model from a local image file (PNG, JPG or WEBP; clear subject, clean background).

    image_path: absolute path (or relative to the current folder) of the picture on this computer.
    prompt: optional short description that names and guides the model.
    Returns the job; call wait_for_model with its job_id.
    """
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ToolError(f"No existe la imagen: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ToolError("Formato no soportado: usa PNG, JPG o WEBP")
    form = {"quality": quality, "style": style}
    if prompt:
        form["prompt"] = prompt
    jobs = await _json("POST", "/api/jobs/image", data=form,
                       files=[("images", (path.name, path.read_bytes(), mime))])
    return _summary(jobs[0])


# ---------------------------------------------------------------- tools: jobs


@mcp.tool()
async def get_model(job_id: str) -> dict[str, Any]:
    """Status (queued, running, succeeded, failed), progress (0-100), stage and download URL of a model or job."""
    return _summary(await _json("GET", f"/api/jobs/{job_id}"))


@mcp.tool()
async def wait_for_model(job_id: str, timeout_s: int = 600) -> dict[str, Any]:
    """Wait until a model or job finishes (succeeded or failed) and return it.

    timeout_s: give up after this many seconds (max 1800) and return the current status with timed_out=true.
    """
    deadline = time.monotonic() + max(1, min(int(timeout_s), 1800))
    while True:
        job = _summary(await _json("GET", f"/api/jobs/{job_id}"))
        if job["status"] in ("succeeded", "failed"):
            return {**job, "timed_out": False}
        if time.monotonic() >= deadline:
            return {**job, "timed_out": True}
        await asyncio.sleep(POLL_S)


@mcp.tool()
async def list_models(limit: int = 20) -> dict[str, Any]:
    """The user's most recent models and jobs in Forja3D, newest first (limit: 1 to 200)."""
    jobs = await _json("GET", "/api/jobs", params={"limit": max(1, min(int(limit), 200))})
    return {"models": [_summary(j) for j in jobs]}


# ---------------------------------------------------------------- tools: characters


@mcp.tool()
async def search_animations(query: str = "", body: Literal["humanoid", "quadruped"] = "humanoid") -> dict[str, Any]:
    """Search the animation library: procedural clips plus 639 motion-capture clips (Spanish or English words).

    query: e.g. "dance", "walk sad", "caminar triste"; empty lists everything.
    body: humanoid (people, robots...) or quadruped (dogs, horses...).
    Returns clip ids to pass to rig_and_animate.
    """
    presets = await _json("GET", "/api/rig/presets")
    if body == "quadruped":
        pool = [(p["id"], p["label"], p["label"], " ".join(p["tags"]), p["duration"])
                for p in presets["quadruped"]["presets"]]
        default = presets["quadruped"]["default"]
    else:
        pool = [(p["id"], p["label"], p["label"], " ".join(p["tags"]), p["duration"]) for p in presets["presets"]]
        pool += [(c["id"], c["label"], c["label_en"], c["category"], c["seconds"]) for c in presets["library"]]
        default = presets["default"]
    words = [w for w in _fold(query).split() if len(w) > 1]
    hits = []
    for clip_id, label, label_en, extra, seconds in pool:
        text = _fold(f"{clip_id} {label} {label_en} {extra}")
        score = sum(w in text for w in words)
        if score or not words:
            hits.append((-score, clip_id, label, label_en, seconds))
    hits.sort()
    return {"default": default, "total": len(hits),
            "animations": [{"id": h[1], "label": h[2], "label_en": h[3], "seconds": h[4]} for h in hits[:25]]}


@mcp.tool()
async def rig_and_animate(job_id: str, animations: list[str] | None = None, body: Body = "auto",
                          hd_textures: Literal["none", "2k", "4k", "8k"] = "4k") -> dict[str, Any]:
    """Turn a finished model into an animated character: HD/PBR textures, skeleton, skin weights and clips.

    animations: clip ids from search_animations (empty = a default set: idle, walk, run...).
    body: auto-detect, humanoid or quadruped.
    hd_textures: upscale the textures first (none to skip).
    Returns the jobs started; wait_for_model on character_job_id, then export_model (glb, fbx or blend).
    """
    payload: dict[str, Any] = {"template": body, "target": hd_textures}
    if animations:
        payload["animations"] = animations
    jobs = await _json("POST", f"/api/jobs/{job_id}/character", json=payload)
    return {"character_job_id": jobs[-1]["id"], "jobs": [_summary(j) for j in jobs]}


# ---------------------------------------------------------------- tools: printing & export


@mcp.tool()
async def check_printability(job_id: str, size_mm: float = 100, printer: str = "bambu_a1") -> dict[str, Any]:
    """3D-printing report of a finished model printed size_mm tall: watertight, holes, non-manifold edges,
    volume, grams of PLA and whether it fits the printer's bed.

    printer: printer id from list_printers (e.g. bambu_a1, prusa_mk4, ender3).
    """
    return await _json("GET", f"/api/jobs/{job_id}/analysis", params={"size_mm": size_mm, "bed": printer})


@mcp.tool()
async def list_printers() -> dict[str, Any]:
    """Printers with their bed size (ids for check_printability) and the slicers installed on this computer."""
    return await _json("GET", "/api/printers")


@mcp.tool()
async def export_model(job_id: str, format: ExportFormat = "glb", path: str | None = None,
                       size_mm: float | None = None) -> dict[str, Any]:
    """Download a finished model to a file on this computer.

    format: glb (web, Unity, Unreal, Godot; keeps rig and animations), fbx or blend (need Blender installed),
    obj (zip), stl / 3mf (3D printing, Z up, millimetres), 3mf_color / stl_colors (multicolour printing), ply.
    path: file or folder to save to (default: current folder, named after the model).
    size_mm: final height in millimetres (printing formats default to 100 mm).
    """
    params: dict[str, Any] = {"format": format}
    if size_mm:
        params["size_mm"] = size_mm
    resp = await _call("GET", f"/api/jobs/{job_id}/export", params=params)
    match = re.search(r'filename="([^"]+)"', resp.headers.get("content-disposition", ""))
    filename = match.group(1) if match else f"{job_id}.{format}"
    target = Path(path).expanduser() if path else Path.cwd()
    if target.is_dir() or (path and path.endswith(("/", "\\"))):
        target = target / filename
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    return {"path": str(target), "bytes": len(resp.content), "format": format,
            "content_type": resp.headers.get("content-type")}


if __name__ == "__main__":
    mcp.run()
