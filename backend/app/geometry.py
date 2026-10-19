"""Local mesh processing: remesh, low-poly, voxels, print checks/repair, splitting and format export.

Everything here runs on CPU with trimesh, independent of the AI provider, so it works on models
from any source (generated or uploaded). Textures are baked into vertex colors where an operation
changes the topology.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import zipfile
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

GREY = np.array([180, 180, 186, 255], dtype=np.uint8)
CLAY = np.array([0.78, 0.74, 0.70])

UPLOAD_TYPES = {"glb", "gltf", "obj", "stl", "ply", "off", "3mf"}
EXPORT_FORMATS = {
    # format: (file extension, media type, is a 3D-printing format -> Z-up, millimetres)
    "glb": ("glb", "model/gltf-binary", False),
    "obj": ("zip", "application/zip", False),
    "stl": ("stl", "model/stl", True),
    "3mf": ("3mf", "model/3mf", True),
    "ply": ("ply", "application/octet-stream", False),
    "3mf_color": ("3mf", "model/3mf", True),  # multicolour (Bambu AMS, Prusa MMU), one filament per colour
    "stl_colors": ("zip", "application/zip", True),  # one STL per colour, for any slicer
}
DEFAULT_PRINT_HEIGHT_MM = 100.0


class GeometryError(Exception):
    """User-facing processing error."""


# ---------------------------------------------------------------- loading


# Vertex colours live in sRGB inside the app (like pictures, 3MF and PLY), but glTF's COLOR_0 is
# linear: convert when reading and writing GLB so viewers, Blender and engines show the real colours.
_TO_LINEAR = np.round(255 * np.where((x := np.arange(256) / 255) <= 0.04045, x / 12.92,
                                     ((x + 0.055) / 1.055) ** 2.4)).astype(np.uint8)
_TO_SRGB = np.round(255 * np.where((x := np.arange(256) / 255) <= 0.0031308, x * 12.92,
                                   1.055 * x ** (1 / 2.4) - 0.055)).clip(0, 255).astype(np.uint8)


def srgb_to_linear(rgba: np.ndarray) -> np.ndarray:
    out = np.array(rgba, dtype=np.uint8, copy=True)
    out[..., :3] = _TO_LINEAR[out[..., :3]]
    return out


def _vertex_colored(scene: trimesh.Scene):
    for geom in scene.geometry.values():
        visual = getattr(geom, "visual", None)
        if isinstance(visual, trimesh.visual.ColorVisuals) and visual.kind in ("vertex", "face"):
            yield geom


def load_scene(data: bytes, file_type: str = "glb") -> trimesh.Scene:
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="scene")
    except Exception as exc:  # noqa: BLE001 - trimesh raises many different types
        raise GeometryError(f"No se pudo leer el archivo {file_type.upper()}: {exc}") from exc
    if not loaded.geometry:
        raise GeometryError("El archivo no contiene ninguna malla")
    if file_type == "glb":
        for geom in _vertex_colored(loaded):
            colors = np.array(geom.visual.vertex_colors, dtype=np.uint8)
            colors[:, :3] = _TO_SRGB[colors[:, :3]]
            geom.visual.vertex_colors = colors
    return loaded


def glb_bytes(scene: trimesh.Scene) -> bytes:
    """GLB with vertex colours written as linear COLOR_0 (the scene itself is left untouched)."""
    saved = []
    for geom in _vertex_colored(scene):
        colors = np.array(geom.visual.vertex_colors, dtype=np.uint8)
        saved.append((geom, colors))
        geom.visual.vertex_colors = srgb_to_linear(colors)
    try:
        return scene.export(file_type="glb", include_normals=True)
    finally:
        for geom, colors in saved:
            geom.visual.vertex_colors = colors


def to_color_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Flatten a scene into a single mesh with per-vertex colors (textures are sampled)."""
    parts = []
    for mesh in scene.dump():
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        mesh = mesh.copy()
        try:
            if mesh.visual.kind == "texture":
                colors = mesh.visual.to_color().vertex_colors
            elif mesh.visual.kind in ("vertex", "face"):
                colors = mesh.visual.vertex_colors  # ColorVisuals derives per-vertex from face colors
            else:
                colors = None
        except Exception:  # noqa: BLE001 - broken/missing texture: fall back to grey
            colors = None
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=colors if colors is not None and len(colors) == len(mesh.vertices) else
            np.tile(GREY, (len(mesh.vertices), 1))
        )
        parts.append(mesh)
    if not parts:
        raise GeometryError("El modelo no contiene triángulos")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def mesh_to_glb(mesh_or_scene) -> bytes:
    scene = mesh_or_scene if isinstance(mesh_or_scene, trimesh.Scene) else trimesh.Scene(mesh_or_scene)
    return glb_bytes(scene)


def transfer_colors(source: trimesh.Trimesh, target: trimesh.Trimesh, points: np.ndarray | None = None) -> np.ndarray:
    """Colors of the nearest source vertex for each target vertex (or for the given points)."""
    _, idx = cKDTree(source.vertices).query(target.vertices if points is None else points)
    return source.visual.vertex_colors[idx]


# ---------------------------------------------------------------- remesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    target_faces = int(max(100, target_faces))
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    out = mesh.simplify_quadric_decimation(face_count=target_faces)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=transfer_colors(mesh, out))
    return out


def facet(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Flat shading: one color and one normal per face."""
    face_colors = mesh.visual.vertex_colors[mesh.faces].astype(float).mean(axis=1).astype(np.uint8)
    out = mesh.copy()
    out.unmerge_vertices()
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.repeat(face_colors, 3, axis=0))
    return out


def lowpoly(mesh: trimesh.Trimesh, target_faces: int = 1500) -> trimesh.Trimesh:
    return facet(decimate(mesh, target_faces))


def voxelize(mesh: trimesh.Trimesh, resolution: int = 32) -> trimesh.Trimesh:
    resolution = int(np.clip(resolution, 8, 96))
    pitch = float(mesh.extents.max()) / resolution
    grid = mesh.voxelized(pitch)
    try:
        grid = grid.fill()
    except Exception:  # noqa: BLE001 - open meshes cannot be filled; keep the shell
        pass
    centers = grid.points
    if len(centers) == 0:
        raise GeometryError("La resolución es demasiado baja para este modelo")
    filled = grid.matrix
    idx = np.argwhere(filled)
    colors = transfer_colors(mesh, mesh, points=grid.indices_to_points(idx))
    return _voxel_surface(filled, idx, colors, grid.transform)


def _voxel_surface(filled: np.ndarray, idx: np.ndarray, colors: np.ndarray, transform: np.ndarray) -> trimesh.Trimesh:
    """Cube faces that touch empty space only (much lighter and faster than one box per voxel)."""
    padded = np.pad(filled, 1)
    verts, faces, vcols = [], [], []
    count = 0
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        # e_u x e_v points along +axis for axes 0 and 2, along -axis for axis 1
        parity = -1 if axis == 1 else 1
        for sign in (-1, 1):
            offset = np.zeros(3, dtype=int)
            offset[axis] = sign
            exposed = ~padded[tuple((idx + 1 + offset).T)]
            sel, col = idx[exposed], colors[exposed]
            if not len(sel):
                continue
            corners = np.zeros((4, 3))
            corners[:, axis] = 0.5 * sign
            corners[:, u] = [-0.5, 0.5, 0.5, -0.5]
            corners[:, v] = [-0.5, -0.5, 0.5, 0.5]
            quad_order = [[0, 1, 2], [0, 2, 3]] if sign * parity > 0 else [[0, 2, 1], [0, 3, 2]]
            base = count + np.arange(len(sel))[:, None, None] * 4
            verts.append((sel[:, None, :] + corners[None]).reshape(-1, 3))
            faces.append((base + np.array(quad_order)[None]).reshape(-1, 3))
            vcols.append(np.repeat(col, 4, axis=0))
            count += len(sel) * 4
    out = trimesh.Trimesh(
        trimesh.transform_points(np.vstack(verts), transform), np.vstack(faces), process=False
    )
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.vstack(vcols))
    return out


def recolor(mesh: trimesh.Trimesh, rgb: np.ndarray, rng: np.random.Generator | None = None) -> trimesh.Trimesh:
    """Single-material repaint with a soft vertical gradient (used for clay / quick retexture)."""
    rng = rng or np.random.default_rng(0)
    out = mesh.copy()
    y = out.vertices[:, 1]
    t = (y - y.min()) / max(float(np.ptp(y)), 1e-6)
    shade = 0.7 + 0.3 * t[:, None] + rng.normal(0, 0.02, (len(y), 1))
    rgba = np.hstack([np.clip(rgb[None, :] * shade, 0, 1), np.ones((len(y), 1))])
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=(rgba * 255).astype(np.uint8))
    return out


def remesh(glb: bytes, mode: str, target_faces: int = 5000, resolution: int = 32) -> bytes:
    mesh = to_color_mesh(load_scene(glb))
    if mode == "decimate":
        out = decimate(mesh, target_faces)
    elif mode == "lowpoly":
        out = lowpoly(mesh, target_faces)
    elif mode == "voxel":
        out = voxelize(mesh, resolution)
    elif mode == "smooth":
        out = mesh.copy()
        trimesh.smoothing.filter_taubin(out, iterations=10)
    else:
        raise GeometryError(f"Modo de remallado desconocido: {mode}")
    return mesh_to_glb(out)


@contextlib.contextmanager
def _quiet_stdout():
    """Silence C++ libraries that print straight to the process's stdout."""
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, ValueError, io.UnsupportedOperation):
        yield
        return
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)


def quad_remesh(glb: bytes, target_faces: int = 5000) -> tuple[bytes, bytes]:
    """Instant Meshes field-aligned remesh into pure quads: (triangulated GLB, quad OBJ)."""
    import pynanoinstantmeshes

    mesh = to_color_mesh(load_scene(glb))
    mesh.merge_vertices()
    target = int(np.clip(target_faces, 200, 400_000))
    with _quiet_stdout():
        verts, quads = pynanoinstantmeshes.remesh(
            np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32),
            max(target // 4, 50), rosy=4, posy=4, deterministic=True)
    verts, quads = np.asarray(verts, dtype=float), np.asarray(quads, dtype=np.int64)
    if not len(quads):
        raise GeometryError("No se pudo generar la malla de quads (¿modelo demasiado pequeño o roto?)")
    colors = transfer_colors(mesh, mesh, points=verts)
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    out = trimesh.Trimesh(verts, tris, process=False)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=colors)
    lines = ["# Forja3D quad remesh (Instant Meshes)"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f} {r / 255:.4f} {g / 255:.4f} {b / 255:.4f}"
              for (x, y, z), (r, g, b, _a) in zip(verts, colors)]
    lines += [f"f {a + 1} {b + 1} {c + 1} {d + 1}" for a, b, c, d in quads]
    return mesh_to_glb(out), ("\n".join(lines) + "\n").encode()


def apply_style(glb: bytes, style: str) -> bytes:
    """Styles that are guaranteed by geometry, not only by the prompt."""
    if style not in ("lowpoly", "voxel", "sculpture"):
        return glb
    mesh = to_color_mesh(load_scene(glb))
    if style == "lowpoly":
        return mesh_to_glb(lowpoly(mesh, 1200))
    if style == "voxel":
        return mesh_to_glb(voxelize(mesh, 28))
    return mesh_to_glb(recolor(mesh, CLAY))


# ---------------------------------------------------------------- 3D printing


@dataclass
class Analysis:
    faces: int
    vertices: int
    bodies: int
    watertight: bool
    winding_consistent: bool
    volume: float | None
    area: float
    extents: list[float]
    printable: bool
    issues: list[str]

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def analyze(glb: bytes) -> Analysis:
    mesh = to_color_mesh(load_scene(glb))
    mesh.merge_vertices()  # flat-shaded / voxel meshes duplicate vertices per face
    issues = []
    if not mesh.is_watertight:
        issues.append("La malla tiene agujeros (no es estanca)")
    if not mesh.is_winding_consistent:
        issues.append("Normales con orientación inconsistente")
    degenerate = int((mesh.area_faces < 1e-12).sum())
    if degenerate:
        issues.append(f"{degenerate} triángulos degenerados")
    bodies = len(mesh.split(only_watertight=False))
    if bodies > 1:
        issues.append(f"{bodies} piezas sueltas")
    return Analysis(
        faces=len(mesh.faces),
        vertices=len(mesh.vertices),
        bodies=bodies,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        volume=float(abs(mesh.volume)) if mesh.is_watertight else None,
        area=float(mesh.area),
        extents=[float(x) for x in mesh.extents],
        printable=bool(mesh.is_watertight and mesh.is_winding_consistent and not degenerate),
        issues=issues,
    )


def repair(glb: bytes) -> bytes:
    """Best-effort fix for printing: merge, drop degenerate faces, fill holes, fix normals."""
    mesh = to_color_mesh(load_scene(glb))
    colors_src = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(mesh)
    if not mesh.is_watertight:
        mesh = _cap_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        # Fall back to a watertight shell rebuilt from voxels when holes are too large to patch.
        rebuilt = voxelize(colors_src, 80)
        rebuilt.merge_vertices()
        trimesh.smoothing.filter_taubin(rebuilt, iterations=8)
        mesh = rebuilt
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=transfer_colors(colors_src, mesh))
    return mesh_to_glb(mesh)


def _cap_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close every boundary loop with a triangle fan around its centroid.

    Each boundary edge (a, b) belongs to exactly one face; the cap triangle (b, a, center) uses the
    opposite direction, so the winding stays consistent with the neighbouring surface.
    """
    directed = mesh.edges
    boundary = directed[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    if not len(boundary):
        return mesh
    vertices, faces = [mesh.vertices], [mesh.faces]
    next_index = len(mesh.vertices)
    for loop in trimesh.graph.connected_components(boundary, min_len=3):
        loop_edges = boundary[np.isin(boundary[:, 0], loop)]
        vertices.append(mesh.vertices[loop].mean(axis=0, keepdims=True))
        faces.append(np.column_stack([loop_edges[:, 1], loop_edges[:, 0], np.full(len(loop_edges), next_index)]))
        next_index += 1
    capped = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    capped.merge_vertices()
    return capped


def split_parts(glb: bytes, mode: str = "height", parts: int = 2) -> bytes:
    """Split for printing: by connected components or into horizontal slabs (with caps)."""
    mesh = to_color_mesh(load_scene(glb))
    mesh.merge_vertices()
    if mode == "components":
        pieces = mesh.split(only_watertight=False)
    elif mode == "height":
        parts = int(np.clip(parts, 2, 8))
        if not mesh.is_watertight:
            raise GeometryError("Para cortar por altura la malla debe ser estanca: usa primero «Reparar»")
        lo, hi = mesh.bounds[:, 1]
        cuts = np.linspace(lo, hi, parts + 1)[1:-1]
        pieces, rest = [], mesh
        for y in cuts:
            below = rest.slice_plane([0, y, 0], [0, -1, 0], cap=True)
            rest = rest.slice_plane([0, y, 0], [0, 1, 0], cap=True)
            pieces.append(below)
        pieces.append(rest)
    else:
        raise GeometryError(f"Modo de división desconocido: {mode}")

    pieces = [p for p in pieces if p is not None and len(p.faces)]
    for piece in pieces:  # cap triangulation can leave slivers
        piece.update_faces(piece.nondegenerate_faces())
        piece.remove_unreferenced_vertices()
    if len(pieces) < 2:
        raise GeometryError("No se encontraron piezas separables con ese modo")
    scene = trimesh.Scene()
    gap = float(mesh.extents.max()) * 0.15
    for i, piece in enumerate(pieces):
        piece.visual = trimesh.visual.ColorVisuals(piece, vertex_colors=transfer_colors(mesh, piece))
        if mode == "height":  # explode vertically so the cuts are visible
            piece.apply_translation([0, gap * i, 0])
        scene.add_geometry(piece, node_name=f"pieza_{i + 1}")
    return mesh_to_glb(scene)


# ---------------------------------------------------------------- export


FIT_AXES = {"width": 0, "height": 1, "depth": 2}


def color_groups(mesh: trimesh.Trimesh, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster face colours into `n_colors` filaments: (label per face, palette uint8 (k, 3))."""
    from scipy.cluster.vq import kmeans2

    face_rgb = mesh.visual.vertex_colors[mesh.faces][:, :, :3].astype(float).mean(axis=1)
    k = int(np.clip(n_colors, 1, 16))
    unique = np.unique(np.round(face_rgb), axis=0)
    if len(unique) <= k:  # already few colours: keep them exactly
        palette = unique
        labels = np.argmin(((face_rgb[:, None, :] - palette[None]) ** 2).sum(axis=2), axis=1)
    else:
        sample = face_rgb[np.random.default_rng(0).choice(len(face_rgb), min(len(face_rgb), 20000), replace=False)]
        palette, _ = kmeans2(sample, k, minit="++", seed=0)
        labels = np.argmin(((face_rgb[:, None, :] - palette[None]) ** 2).sum(axis=2), axis=1)
    used = np.unique(labels)  # drop empty clusters, renumber 0..n-1
    remap = np.full(len(palette), -1)
    remap[used] = np.arange(len(used))
    return remap[labels], np.clip(np.round(palette[used]), 0, 255).astype(np.uint8)


def _paint_code(index: int) -> str:
    """PrusaSlicer/Bambu/Orca per-triangle extruder encoding (TriangleSelector serialisation)."""
    state = index + 1  # extruder number, 1-based
    if state < 3:
        return "%X" % (state << 2)
    rest = state - 3
    return ("%X" % rest) + "C" if rest < 16 else "%X" % 0xC


def multicolor_3mf(mesh: trimesh.Trimesh, n_colors: int) -> bytes:
    labels, palette = color_groups(mesh, n_colors)
    v, f = mesh.vertices, mesh.faces
    hexes = ["#%02X%02X%02X" % tuple(c) for c in palette]
    vertices = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    triangles = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{k}" paint_color="{_paint_code(k)}" '
        f'slic3rpe:mmu_segmentation="{_paint_code(k)}"/>'
        for (a, b, c), k in zip(f, labels))
    bases = "".join(f'<base name="Filamento {i + 1}" displaycolor="{h}FF"/>' for i, h in enumerate(hexes))
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="es-ES" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
        '<metadata name="Application">Forja3D</metadata>'
        f'<resources><basematerials id="1">{bases}</basematerials>'
        f'<object id="2" type="model" pid="1" pindex="0"><mesh><vertices>{vertices}</vertices>'
        f'<triangles>{triangles}</triangles></mesh></object></resources>'
        '<build><item objectid="2"/></build></model>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        zf.writestr("_rels/.rels",
                    '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
                    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        zf.writestr("3D/3dmodel.model", model)
        zf.writestr("Metadata/filamentos.txt", "\n".join(f"Filamento {i + 1}: {h}" for i, h in enumerate(hexes)) + "\n")
    return buf.getvalue()


def stl_per_color(mesh: trimesh.Trimesh, n_colors: int) -> bytes:
    labels, palette = color_groups(mesh, n_colors)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, color in enumerate(palette):
            part = mesh.submesh([np.where(labels == i)[0]], append=True)
            name = "color_%d_%02X%02X%02X.stl" % (i + 1, *color)
            zf.writestr(name, part.export(file_type="stl"))
    return buf.getvalue()


def export(glb: bytes, fmt: str, size_mm: float | None = None, fit: str = "height",
           n_colors: int = 4) -> tuple[bytes, str, str]:
    """Return (data, file extension, media type). Print formats are Z-up, in mm, resting on the bed.

    `size_mm` is the final size of the dimension named by `fit` (width / height / depth, or `max`
    for the longest side), measured on the model as shown in the viewer (Y up).
    """
    if fmt not in EXPORT_FORMATS:
        raise GeometryError(f"Formato no soportado: {fmt}")
    ext, media_type, is_print = EXPORT_FORMATS[fmt]
    scene = load_scene(glb)

    extents = scene.extents
    current = float(extents.max() if fit == "max" else extents[FIT_AXES.get(fit, 1)]) or float(extents.max()) or 1.0
    if is_print:
        scene.apply_scale((size_mm or DEFAULT_PRINT_HEIGHT_MM) / current)
        scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        scene.apply_translation([0, 0, -scene.bounds[0][2]])
    elif size_mm:
        scene.apply_scale(size_mm / 1000 / current)  # glTF units are metres

    if fmt == "glb":
        return glb_bytes(scene), ext, media_type
    if fmt == "obj":
        mesh = scene.to_geometry()
        text, files = trimesh.exchange.obj.export_obj(mesh, include_texture=True, return_texture=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.obj", text)
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue(), ext, media_type

    mesh = to_color_mesh(scene)
    if fmt == "3mf_color":
        return multicolor_3mf(mesh, n_colors), ext, media_type
    if fmt == "stl_colors":
        return stl_per_color(mesh, n_colors), ext, media_type
    if fmt == "3mf":
        return trimesh.Scene(mesh).export(file_type="3mf"), ext, media_type
    return mesh.export(file_type=fmt), ext, media_type


def convert_upload(data: bytes, file_type: str) -> bytes:
    """Any supported upload -> normalized GLB (centered, resting on the ground)."""
    if file_type not in UPLOAD_TYPES:
        raise GeometryError("Formato no soportado: usa GLB, OBJ, STL, PLY, OFF o 3MF")
    scene = load_scene(data, file_type)
    if file_type in ("stl", "3mf"):  # print formats are Z-up
        scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    center = scene.bounds.mean(axis=0)
    scene.apply_translation([-center[0], -scene.bounds[0][1], -center[2]])
    return glb_bytes(scene)
