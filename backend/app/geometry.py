"""Local mesh processing: remesh, low-poly, voxels, print checks/repair, splitting and format export.

Everything here runs on CPU with trimesh, independent of the AI provider, so it works on models
from any source (generated or uploaded). Textures are baked into vertex colors where an operation
changes the topology.
"""

from __future__ import annotations

import io
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
}
DEFAULT_PRINT_HEIGHT_MM = 100.0


class GeometryError(Exception):
    """User-facing processing error."""


# ---------------------------------------------------------------- loading


def load_scene(data: bytes, file_type: str = "glb") -> trimesh.Scene:
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="scene")
    except Exception as exc:  # noqa: BLE001 - trimesh raises many different types
        raise GeometryError(f"No se pudo leer el archivo {file_type.upper()}: {exc}") from exc
    if not loaded.geometry:
        raise GeometryError("El archivo no contiene ninguna malla")
    return loaded


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
    return scene.export(file_type="glb", include_normals=True)


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


def export(glb: bytes, fmt: str, height_mm: float | None = None) -> tuple[bytes, str, str]:
    """Return (data, file extension, media type). Print formats are Z-up, in mm, resting on the bed."""
    if fmt not in EXPORT_FORMATS:
        raise GeometryError(f"Formato no soportado: {fmt}")
    ext, media_type, is_print = EXPORT_FORMATS[fmt]
    scene = load_scene(glb)

    height = float(scene.extents[1]) or 1.0
    if is_print:
        scene.apply_scale((height_mm or DEFAULT_PRINT_HEIGHT_MM) / height)
        scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        scene.apply_translation([0, 0, -scene.bounds[0][2]])
    elif height_mm:
        scene.apply_scale(height_mm / 1000 / height)  # glTF units are metres

    if fmt == "glb":
        return scene.export(file_type="glb", include_normals=True), ext, media_type
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
    return scene.export(file_type="glb", include_normals=True)
