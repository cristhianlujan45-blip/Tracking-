#!/usr/bin/env python3
"""Forja3D command line: one command turns a prompt or an image into a finished 3D model.

Only the Python standard library is needed. It talks to a running Forja3D (the desktop app or the web server):

    forja3d.py text "una guerrera samurái con armadura roja" --pose a-pose --faces 30000 --pbr --wait
    forja3d.py image foto.png --quality high --topology quad --wait --export glb
    forja3d.py rig JOB_ID --animations idle,walk --wait --export fbx
    forja3d.py animations dance
    forja3d.py list
    forja3d.py export JOB_ID --format 3mf --size-mm 80 -o figura.3mf

The server is http://127.0.0.1:8000 unless --server or the FORJA_URL environment variable says otherwise.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000"
QUALITIES = ("draft", "standard", "high")
FORMATS = ("glb", "fbx", "blend", "obj", "stl", "3mf", "3mf_color", "stl_colors", "ply")


class CliError(Exception):
    pass


def server_url(args: argparse.Namespace) -> str:
    return (args.server or os.getenv("FORJA_URL") or DEFAULT_URL).rstrip("/")


def request(args: argparse.Namespace, method: str, path: str, *, json_body=None, form=None, files=None, params=None):
    """Returns (body bytes, headers). Errors from the server become CliError with its own message."""
    url = server_url(args) + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data, headers = None, {}
    if json_body is not None:
        data, headers = json.dumps(json_body).encode(), {"Content-Type": "application/json"}
    elif form is not None or files:
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in (form or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, (name, content, mime) in (files or []):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\n'
                         f"Content-Type: {mime}\r\n\r\n".encode() + content + b"\r\n")
        data = b"".join(parts) + f"--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise CliError(f"{exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise CliError(f"No se pudo conectar con Forja3D en {server_url(args)} ({exc.reason}). "
                       "Abre la app o indica --server.") from exc


def api(args, method, path, **kwargs):
    body, _ = request(args, method, path, **kwargs)
    return json.loads(body) if body else None


def say(args, value):
    if isinstance(value, str) and not args.json:
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def summary(job: dict) -> str:
    pct = f" {job['progress']}%" if job["status"] == "running" else ""
    return f"{job['id']}  {job['status']}{pct}  {job['mode']}  {job['title']}"


def wait_for(args, job_id: str, timeout: int) -> dict:
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        job = api(args, "GET", f"/api/jobs/{job_id}")
        line = f"{job['status']} {job['progress']}% {job.get('stage') or ''}".strip()
        if line != last and not args.json:
            print(f"  {line}", file=sys.stderr)
            last = line
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            raise CliError(job.get("error") or "La generación falló")
        time.sleep(1.5)
    raise CliError(f"Tiempo agotado esperando el modelo {job_id}")


def production(args) -> dict:
    out = {"topology": args.topology, "pbr": args.pbr}
    if args.faces:
        out["target_faces"] = args.faces
    return out


def finish(args, job: dict):
    """--wait / --export handling shared by the commands that start a job."""
    if args.wait or args.export:
        job = wait_for(args, job["id"], args.timeout)
    if args.export:
        say(args, save_export(args, job["id"], args.export, args.output, None))
    else:
        say(args, job if args.json else summary(job))


def save_export(args, job_id: str, fmt: str, output: str | None, size_mm: float | None) -> dict | str:
    body, headers = request(args, "GET", f"/api/jobs/{job_id}/export", params={"format": fmt, "size_mm": size_mm})
    match = re.search(r'filename="([^"]+)"', headers.get("Content-Disposition", ""))
    name = match.group(1) if match else f"{job_id}.{fmt}"
    target = Path(output).expanduser() if output else Path.cwd()
    if target.is_dir() or (output and output.endswith(("/", "\\"))):
        target = target / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return {"path": str(target.resolve()), "bytes": len(body), "format": fmt} if args.json else str(target.resolve())


# ---------------------------------------------------------------- commands


def cmd_text(args):
    body = {"prompt": args.prompt, "quality": args.quality, "style": args.style, "pose": args.pose,
            "symmetry": args.symmetry, "variants": 1, **production(args)}
    if args.seed is not None:
        body["seed"] = args.seed
    finish(args, api(args, "POST", "/api/jobs/text", json_body=body)[0])


def cmd_image(args):
    path = Path(args.image).expanduser()
    if not path.is_file():
        raise CliError(f"No existe la imagen: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise CliError("Formato no soportado: usa PNG, JPG o WEBP")
    form = {"quality": args.quality, "style": args.style, "variants": 1, "topology": args.topology,
            "pbr": str(args.pbr).lower()}
    if args.faces:
        form["target_faces"] = args.faces
    if args.prompt:
        form["prompt"] = args.prompt
    jobs = api(args, "POST", "/api/jobs/image", form=form, files=[("images", (path.name, path.read_bytes(), mime))])
    finish(args, jobs[0])


def cmd_list(args):
    jobs = api(args, "GET", "/api/jobs")[: args.limit]
    say(args, jobs if args.json else "\n".join(summary(j) for j in jobs) or "Aún no hay modelos.")


def cmd_get(args):
    job = api(args, "GET", f"/api/jobs/{args.job_id}")
    say(args, job if args.json else summary(job))


def cmd_wait(args):
    job = wait_for(args, args.job_id, args.timeout)
    say(args, job if args.json else summary(job))


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def cmd_animations(args):
    presets = api(args, "GET", "/api/rig/presets")
    pool = [(p["id"], p["label"], " ".join(p["tags"]), p["duration"]) for p in presets["presets"]]
    pool += [(c["id"], c["label"], f"{c['label_en']} {c['category']}", c["seconds"]) for c in presets["library"]]
    words = [w for w in fold(args.query).split() if len(w) > 1]
    hits = [(clip, label, secs) for clip, label, extra, secs in pool
            if not words or any(w in fold(f"{clip} {label} {extra}") for w in words)]
    hits = hits[: args.limit]
    say(args, [{"id": c, "label": label, "seconds": s} for c, label, s in hits] if args.json
        else "\n".join(f"{c}\t{label}\t{s:.1f}s" for c, label, s in hits) or "Sin resultados.")


def cmd_rig(args):
    body = {"template": args.body, "target": args.hd_textures}
    if args.animations:
        body["animations"] = [a.strip() for a in args.animations.split(",") if a.strip()]
    jobs = api(args, "POST", f"/api/jobs/{args.job_id}/character", json_body=body)
    finish(args, jobs[-1])


def cmd_export(args):
    say(args, save_export(args, args.job_id, args.format, args.output, args.size_mm))


def cmd_slicer(args):
    """Bambu Studio, Creality Print, OrcaSlicer…: only on the computer that runs Forja3D (the server opens it)."""
    body = {"slicer": args.slicer, "size_mm": args.size_mm, "format": "3mf_color" if args.colors else "3mf",
            "bed": args.bed}
    if args.colors:
        body["colors"] = args.colors
    say(args, api(args, "POST", f"/api/jobs/{args.job_id}/slicer", json_body={k: v for k, v in body.items() if v}))


def cmd_printers(args):
    info = api(args, "GET", "/api/printers")
    say(args, info if args.json else json.dumps(info, ensure_ascii=False, indent=2))


def cmd_health(args):
    say(args, api(args, "GET", "/api/health"))


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forja3d", description="Forja3D en la terminal: texto o imagen → modelo 3D.")
    parser.add_argument("--server", help=f"dirección de Forja3D (por defecto {DEFAULT_URL} o FORJA_URL)")
    parser.add_argument("--json", action="store_true", help="salida en JSON (para scripts)")
    sub = parser.add_subparsers(dest="command", required=True)

    def generative(p):
        p.add_argument("--quality", choices=QUALITIES, default="high")
        p.add_argument("--style", default="auto", help="auto, realistic, cartoon, handpainted, lowpoly, voxel, sculpture")
        p.add_argument("--topology", choices=("triangle", "quad"), default="triangle")
        p.add_argument("--faces", type=int, metavar="N", help="caras objetivo (500 a 300000)")
        p.add_argument("--pbr", action="store_true", help="crear mapas PBR")
        p.add_argument("--wait", action="store_true", help="esperar a que termine")
        p.add_argument("--timeout", type=int, default=900, help="segundos máximos de espera (por defecto 900)")
        p.add_argument("--export", choices=FORMATS, help="al terminar, guardar el archivo (implica --wait)")
        p.add_argument("-o", "--output", help="archivo o carpeta de destino para --export")

    p = sub.add_parser("text", help="texto → 3D")
    p.add_argument("prompt")
    p.add_argument("--pose", choices=("none", "a-pose", "t-pose"), default="none")
    p.add_argument("--symmetry", action="store_true")
    p.add_argument("--seed", type=int)
    generative(p)
    p.set_defaults(fn=cmd_text)

    p = sub.add_parser("image", help="imagen → 3D")
    p.add_argument("image")
    p.add_argument("--prompt", help="descripción opcional")
    generative(p)
    p.set_defaults(fn=cmd_image)

    p = sub.add_parser("list", help="modelos recientes")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("get", help="estado de un modelo")
    p.add_argument("job_id")
    p.set_defaults(fn=cmd_get)

    p = sub.add_parser("wait", help="esperar a que termine un modelo")
    p.add_argument("job_id")
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("animations", help="buscar animaciones (español o inglés)")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(fn=cmd_animations)

    p = sub.add_parser("rig", help="personaje completo: texturas HD + esqueleto + animaciones")
    p.add_argument("job_id")
    p.add_argument("--animations", help="ids separados por comas (vacío = conjunto básico)")
    p.add_argument("--body", choices=("auto", "humanoid", "quadruped"), default="auto")
    p.add_argument("--hd-textures", choices=("none", "2k", "4k", "8k"), default="4k")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--export", choices=FORMATS)
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_rig)

    p = sub.add_parser("export", help="descargar un modelo terminado")
    p.add_argument("job_id")
    p.add_argument("--format", choices=FORMATS, default="glb")
    p.add_argument("--size-mm", type=float, help="altura final en milímetros (impresión)")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("slicer", help="abrir el modelo en tu laminador (Bambu Studio, Creality Print, Orca…)")
    p.add_argument("job_id")
    p.add_argument("--slicer", help="bambu, creality, orca, prusa, cura, elegoo… (vacío = el predeterminado de .3mf)")
    p.add_argument("--size-mm", type=float, help="altura final en milímetros")
    p.add_argument("--colors", type=int, help="filamentos: 3MF multicolor (AMS / MMU)")
    p.add_argument("--bed", default="bambu_a1", help="impresora (ver `printers`)")
    p.set_defaults(fn=cmd_slicer)

    sub.add_parser("printers", help="impresoras y laminadores").set_defaults(fn=cmd_printers)
    sub.add_parser("health", help="comprobar la conexión").set_defaults(fn=cmd_health)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
