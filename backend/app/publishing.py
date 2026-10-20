"""Ready-to-upload kit for MakerWorld, Printables and Thingiverse.

Those sites have no public upload API, so the app prepares everything their upload forms ask for
in one ZIP: the print files at the chosen size, preview pictures and a description with tags
(Spanish and English) to paste; then it opens the site's upload page.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from . import geometry, printing

UPLOAD_PAGES = {
    "makerworld": "https://makerworld.com/es/my/models/upload",
    "printables": "https://www.printables.com/model/create",
    "thingiverse": "https://www.thingiverse.com/upload",
}

VIEWS = {"frente": 0, "tres_cuartos": 35, "lado": 90, "espalda": 180}


def render_views(glb: bytes, size: int = 900, only: tuple[str, ...] | None = None) -> dict[str, bytes]:
    """Soft studio previews from several angles (dense surface splatting with simple lighting)."""
    mesh = geometry.to_color_mesh(geometry.load_scene(glb))
    count = int(min(1_500_000, max(300_000, len(mesh.faces) * 12)) * min(1.0, (size / 900) ** 2 * 2))
    pts, face = trimesh.sample.sample_surface(mesh, count, seed=7)
    nrm = mesh.face_normals[face]
    try:
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face], pts)
        col = (mesh.visual.vertex_colors[:, :3].astype(float)[mesh.faces[face]] * bary[:, :, None]).sum(1)
    except (AttributeError, IndexError, ValueError):
        col = np.full((len(pts), 3), 200.0)
    light_dir = np.array([0.35, 0.55, 0.75]) / np.linalg.norm([0.35, 0.55, 0.75])
    out = {}
    for name, yaw in VIEWS.items():
        if only and name not in only:
            continue
        a = np.radians(yaw)
        rot = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        tilt = np.radians(-12)  # look slightly down at the model, like a product photo
        rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]) @ rot
        p, n = pts @ rot.T, nrm @ rot.T
        lo, hi = p.min(0), p.max(0)
        scale = size * 0.82 / max(hi[0] - lo[0], hi[1] - lo[1])
        x = ((p[:, 0] - (lo[0] + hi[0]) / 2) * scale + size / 2).astype(int).clip(0, size - 1)
        y = (size / 2 - (p[:, 1] - (lo[1] + hi[1]) / 2) * scale).astype(int).clip(0, size - 1)
        order = np.argsort(p[:, 2])  # far first, near points overwrite
        shade = np.clip(n @ light_dir, 0, 1) * 0.6 + 0.4
        rgb = np.zeros((size, size, 3))
        hit = np.zeros((size, size), bool)
        rgb[y[order], x[order]] = (col * shade[:, None])[order]
        hit[y[order], x[order]] = True
        # background: soft vertical gradient with a contact shadow; max filter closes tiny splat holes
        bg = (np.linspace(246, 222, size)[:, None, None] * np.ones((1, size, 3))).astype(np.uint8)
        img = Image.fromarray(bg)
        shadow = Image.new("L", (size, size), 0)
        ImageDraw.Draw(shadow).ellipse([size * 0.25, size * 0.87, size * 0.75, size * 0.95], fill=90)
        img.paste((150, 150, 158), (0, 0), shadow.filter(ImageFilter.GaussianBlur(16)))
        mask = Image.fromarray(hit.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
        filled = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
        img.paste(filled, (0, 0), mask)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out[name] = buf.getvalue()
    return out


def description(title: str, report: dict, size_mm: float, printer_label: str) -> dict:
    x, y, z = report["size_mm"]
    tags = ["forja3d", "ai", "3dprinting", "figure", "miniature"]
    solid = report["watertight"] and not report["non_manifold_edges"]
    weight = f"≈{report['weight_g']} g PLA" if report.get("weight_g") else "—"
    es = (f"{title}\n\nModelo creado con Forja3D y preparado para imprimir.\n\n"
          f"• Tamaño: {x:.0f} × {y:.0f} × {z:.0f} mm\n• Peso aproximado: {weight} (20 % de relleno, 3 paredes)\n"
          f"• Malla {'cerrada y lista para laminar' if solid else 'reparada automáticamente; revisa en tu laminador'}\n"
          f"• Probado para: {printer_label}\n\nAjustes recomendados: capa 0,16–0,20 mm, relleno 15–20 %, "
          "soportes en voladizos de más de 50°, borde (brim) si la base es pequeña.")
    en = (f"{title}\n\nModel made with Forja3D and prepared for printing.\n\n"
          f"• Size: {x:.0f} × {y:.0f} × {z:.0f} mm\n• Approx. weight: {weight} (20 % infill, 3 walls)\n"
          f"• Mesh {'watertight and ready to slice' if solid else 'auto-repaired; check it in your slicer'}\n"
          f"• Tested for: {printer_label}\n\nSuggested settings: 0.16–0.20 mm layers, 15–20 % infill, "
          "supports for overhangs above 50°, brim if the base is small.")
    return {"title": title, "es": es, "en": en, "tags": tags, "license": "CC BY 4.0"}


def kit(glb: bytes, title: str, size_mm: float, bed: str = "bambu_a1") -> bytes:
    """ZIP with 3MF + STL at the final size, PNG previews and the description texts."""
    report = printing.printability(glb, size_mm, "height")
    printer_label = printing.BEDS.get(bed, printing.BEDS["bambu_a1"])[0]
    slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")[:40] or "modelo"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        data, _, _ = geometry.export(glb, "3mf", size_mm, "height")
        zf.writestr(f"{slug}.3mf", data)
        data, _, _ = geometry.export(glb, "stl", size_mm, "height")
        zf.writestr(f"{slug}.stl", data)
        for name, png in render_views(glb).items():
            zf.writestr(f"fotos/{slug}_{name}.png", png)
        info = description(title, report, size_mm, printer_label)
        zf.writestr("descripcion_es.txt", info["es"])
        zf.writestr("description_en.txt", info["en"])
        zf.writestr("etiquetas.txt", ", ".join(info["tags"]))
        zf.writestr("info.json", json.dumps({**info, "print": report, "upload_pages": UPLOAD_PAGES},
                                            ensure_ascii=False, indent=1))
    return buf.getvalue()
