"""3D printing helpers: printability report at the final size, splitting to fit a printer's bed
(each piece closed and laid on the plate) and opening the result in an installed slicer."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .geometry import GeometryError, load_scene, mesh_to_glb, repair, to_color_mesh, transfer_colors

DEFAULT_SIZE_MM = 100.0
FIT_AXES = {"width": 0, "height": 1, "depth": 2}

# build volume in mm (x, y, z)
BEDS = {
    "bambu_a1mini": ("Bambu Lab A1 mini", (180, 180, 180)),
    "bambu_a1": ("Bambu Lab A1 / P1S / X1C", (256, 256, 256)),
    "bambu_h2d": ("Bambu Lab H2D", (325, 320, 325)),
    "prusa_mk4": ("Prusa MK4 / Core One", (250, 210, 220)),
    "prusa_mini": ("Prusa MINI", (180, 180, 180)),
    "ender3": ("Creality Ender-3 V3", (220, 220, 250)),
    "creality_k1": ("Creality K1 / K1C", (220, 220, 250)),
    "creality_k2": ("Creality K2 Plus", (350, 350, 350)),
    "elegoo_n4": ("Elegoo Neptune 4", (225, 225, 265)),
    "anycubic_kobra3": ("Anycubic Kobra 3", (250, 250, 260)),
    "saturn4": ("Elegoo Saturn 4 (resina)", (218, 123, 220)),
    "photon_m5": ("Anycubic Photon Mono M5s (resina)", (218, 123, 200)),
}


def scale_for(extents: np.ndarray, size_mm: float | None, fit: str) -> float:
    current = float(extents.max() if fit == "max" else extents[FIT_AXES.get(fit, 1)]) or float(extents.max()) or 1.0
    return (size_mm or DEFAULT_SIZE_MM) / current


def to_print_space(mesh: trimesh.Trimesh, size_mm: float | None, fit: str) -> trimesh.Trimesh:
    """Viewer space (Y up, any unit) → printer space (Z up, millimetres, resting on z = 0)."""
    out = mesh.copy()
    out.apply_scale(scale_for(out.extents, size_mm, fit))
    out.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    out.apply_translation([-out.bounds[:, 0].mean(), -out.bounds[:, 1].mean(), -out.bounds[0][2]])
    return out


# ---------------------------------------------------------------- printability


def printability(glb: bytes, size_mm: float | None = None, fit: str = "height") -> dict:
    """Numbers a slicer cares about, measured at the final print size."""
    mesh = to_color_mesh(load_scene(glb))
    mesh.merge_vertices()
    edges = mesh.edges_sorted
    counts = trimesh.grouping.group_rows(edges, require_count=1)
    boundary = edges[counts]
    holes = len(trimesh.graph.connected_components(boundary, min_len=3)) if len(boundary) else 0
    _, edge_use = np.unique(edges, axis=0, return_counts=True)
    non_manifold = int((edge_use > 2).sum())
    scale = scale_for(mesh.extents, size_mm, fit)
    size = mesh.extents * scale
    volume_cm3 = float(abs(mesh.volume)) * scale ** 3 / 1000 if mesh.is_watertight else None
    return {
        "watertight": bool(mesh.is_watertight),
        "holes": int(holes),
        "non_manifold_edges": non_manifold,
        "size_mm": [round(float(size[0]), 1), round(float(size[2]), 1), round(float(size[1]), 1)],  # X, Y, Z(up)
        "volume_cm3": round(volume_cm3, 2) if volume_cm3 is not None else None,
        # PLA 1.24 g/cm³ at ~20 % infill with 3 walls ≈ 45 % of the solid weight
        "weight_g": round(volume_cm3 * 1.24 * 0.45, 1) if volume_cm3 is not None else None,
    }


def fits(size_mm: list[float], bed: tuple[int, int, int]) -> bool:
    footprint = sorted(size_mm[:2])
    return footprint[0] <= min(bed[:2]) and footprint[1] <= max(bed[:2]) and size_mm[2] <= bed[2]


# ---------------------------------------------------------------- split to fit the bed


@dataclass
class BedSplit:
    glb: bytes
    pieces: int


def split_for_bed(glb: bytes, bed: tuple[float, float, float], size_mm: float | None = None,
                  fit: str = "height", margin: float = 0.92) -> BedSplit:
    """Cut the model (at its final size) into closed pieces that each fit the build volume.

    Cuts are planar and capped; the model is repaired first when it is not watertight. The result
    is in viewer space (Y up, original units) with the pieces spread out so the cuts are visible."""
    source = to_color_mesh(load_scene(glb))
    source.merge_vertices()
    if not source.is_watertight:
        source = to_color_mesh(load_scene(repair(glb)))
        source.merge_vertices()
    scale = scale_for(source.extents, size_mm, fit)
    limit = np.array([bed[0], bed[2], bed[1]], dtype=float) * margin / scale  # viewer axes: X, Y(up), Z

    pieces, todo = [], [source]
    while todo:
        piece = todo.pop()
        over = piece.extents / limit
        if over.max() <= 1.0 or len(pieces) + len(todo) > 64:
            pieces.append(piece)
            continue
        axis = int(np.argmax(over))
        n = int(np.ceil(over[axis]))
        lo, hi = piece.bounds[:, axis]
        rest = piece
        for k in range(1, n):
            at = lo + (hi - lo) * k / n
            normal = np.eye(3)[axis]
            part = rest.slice_plane(normal * at, -normal, cap=True)
            rest = rest.slice_plane(normal * at, normal, cap=True)
            if part is not None and len(part.faces):
                todo.append(part)
        if rest is not None and len(rest.faces):
            todo.append(rest)
    pieces = [p for p in pieces if p is not None and len(p.faces)]
    if not pieces:
        raise GeometryError("No se pudo dividir el modelo")

    scene = trimesh.Scene()
    gap = float(source.extents.max()) * 0.08
    centre = source.bounds.mean(axis=0)
    for i, piece in enumerate(pieces):
        piece.update_faces(piece.nondegenerate_faces())
        piece.remove_unreferenced_vertices()
        piece.visual = trimesh.visual.ColorVisuals(piece, vertex_colors=transfer_colors(source, piece))
        offset = piece.bounds.mean(axis=0) - centre  # explode outwards so the cuts show
        norm = np.linalg.norm(offset)
        if norm > 1e-9:
            piece.apply_translation(offset / norm * gap)
        scene.add_geometry(piece, node_name=f"pieza_{i + 1}")
    return BedSplit(mesh_to_glb(scene), len(pieces))


def plate_3mf(glb: bytes, size_mm: float | None = None, fit: str = "height",
              bed: tuple[float, float, float] = (256, 256, 256)) -> bytes:
    """3MF with every piece as its own object, lying flat on the plate and arranged side by side."""
    scene = load_scene(glb)
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        raise GeometryError("El modelo no contiene mallas")
    whole = trimesh.util.concatenate(meshes)
    scale = scale_for(whole.extents, size_mm, fit)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    placed, x, y, row_h = trimesh.Scene(), 0.0, 0.0, 0.0
    spacing = 6.0
    for i, mesh in enumerate(meshes):
        m = mesh.copy()
        m.apply_scale(scale)
        m.apply_transform(rot)
        m.apply_translation(-m.bounds[0])  # corner at the origin, on the bed
        w, d = m.extents[0], m.extents[1]
        if x and x + w > bed[0]:
            x, y, row_h = 0.0, y + row_h + spacing, 0.0
        m.apply_translation([x, y, 0])
        x, row_h = x + w + spacing, max(row_h, d)
        placed.add_geometry(m, node_name=f"pieza_{i + 1}")
    lo, hi = placed.bounds
    placed.apply_translation([bed[0] / 2 - (lo[0] + hi[0]) / 2, bed[1] / 2 - (lo[1] + hi[1]) / 2, 0])
    return placed.export(file_type="3mf")


# ---------------------------------------------------------------- slicers

# (id, label, Windows globs, macOS app names, Linux commands)
SLICERS = [
    ("bambu", "Bambu Studio", [r"C:\Program Files\Bambu Studio\bambu-studio.exe"], ["BambuStudio", "Bambu Studio"],
     ["bambu-studio", "BambuStudio"]),
    ("orca", "OrcaSlicer", [r"C:\Program Files\OrcaSlicer\orca-slicer.exe"], ["OrcaSlicer"], ["orca-slicer", "OrcaSlicer"]),
    ("prusa", "PrusaSlicer", [r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe"],
     ["PrusaSlicer", "Original Prusa Drivers/PrusaSlicer"], ["prusa-slicer"]),
    ("cura", "UltiMaker Cura", [r"C:\Program Files\UltiMaker Cura*\UltiMaker-Cura.exe", r"C:\Program Files\Ultimaker Cura*\Cura.exe"],
     ["UltiMaker Cura", "Ultimaker Cura"], ["cura", "UltiMaker-Cura"]),
    ("creality", "Creality Print", [r"C:\Program Files\Creality\Creality Print*\CrealityPrint.exe",
                                    r"C:\Program Files\Creality Print*\CrealityPrint.exe"], ["Creality Print", "CrealityPrint"],
     ["CrealityPrint"]),
    ("elegoo", "ElegooSlicer", [r"C:\Program Files\ElegooSlicer\elegoo-slicer.exe"], ["ElegooSlicer"], ["elegoo-slicer"]),
    ("flashforge", "Orca-Flashforge", [r"C:\Program Files\Orca-Flashforge\orca-flashforge.exe"], ["Orca-Flashforge"], []),
    ("snapmaker", "Snapmaker Orca", [r"C:\Program Files\Snapmaker_Orca\snapmaker-orca.exe"], ["Snapmaker Orca"], []),
    ("lychee", "Lychee Slicer", [r"C:\Program Files\Mango3D\LycheeSlicer\LycheeSlicer.exe",
                                 r"C:\Program Files\LycheeSlicer\LycheeSlicer.exe"], ["Lychee Slicer", "LycheeSlicer"], []),
]


def find_slicers() -> list[dict]:
    found = []
    for sid, label, win, mac, nix in SLICERS:
        target = None
        if sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA", "")
            patterns = win + [p.replace(r"C:\Program Files", local + r"\Programs") for p in win if local]
            hits = sorted(h for p in patterns for h in glob.glob(p))
            target = hits[-1] if hits else None
        elif sys.platform == "darwin":
            for name in mac:
                for base in ("/Applications", str(Path.home() / "Applications")):
                    if Path(base, f"{name}.app").exists():
                        target = str(Path(base, f"{name}.app"))
                        break
                if target:
                    break
        else:
            target = next((shutil.which(c) for c in nix if shutil.which(c)), None)
        if target:
            found.append({"id": sid, "label": label, "path": target})
    return found


def open_in_slicer(path: Path, slicer_id: str | None) -> str:
    """Launch the chosen slicer (or the system's default app for .3mf) with `path`. Returns its name."""
    slicer = next((s for s in find_slicers() if s["id"] == slicer_id), None) if slicer_id else None
    if slicer:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-a", slicer["path"], str(path)])
        else:
            subprocess.Popen([slicer["path"], str(path)])
        return slicer["label"]
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - opens with the program associated to .3mf
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return "default"
