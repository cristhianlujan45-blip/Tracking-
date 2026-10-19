"""Models built directly from images or text, without a generative 3D model.

* **scene**:      photo + depth map → textured 2.5D mesh (backdrops, environments, Blender)
* **relief**:     photo + depth map → watertight plaque with the scene in relief (3D printing)
* **lithophane**: photo brightness → thin plaque that shows the photo when backlit (3D printing)
* **text3d**:     extruded text, optionally on a plate or as a keychain (3D printing, signage)

All meshes follow glTF conventions: Y up, 1 unit ≈ 1 metre; print exports rescale them.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter

from .geometry import GeometryError, mesh_to_glb

MAX_TEXTURE = 2048
WALL_COLOR = np.array([200, 200, 205, 255], dtype=np.uint8)


# ---------------------------------------------------------------- images & depth


def open_rgb(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise GeometryError("No se pudo leer la imagen") from exc


def pseudo_depth(image_bytes: bytes) -> bytes:
    """Depth guess without AI: lower in the frame and brighter means closer.

    Good enough for demos and simple photos; real depth comes from the provider (Depth Anything V2).
    """
    img = open_rgb(image_bytes)
    img.thumbnail((512, 512))
    lum = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(6)), dtype=float) / 255
    rows = np.linspace(0, 1, lum.shape[0])[:, None]  # 0 at top (far), 1 at bottom (near)
    depth = 0.65 * rows + 0.35 * lum
    depth = (depth - depth.min()) / max(np.ptp(depth), 1e-6)
    out = io.BytesIO()
    Image.fromarray((depth * 255).astype(np.uint8), "L").save(out, format="PNG")
    return out.getvalue()


def _grid(image: Image.Image, resolution: int) -> tuple[int, int]:
    w, h = image.size
    scale = resolution / max(w, h)
    return max(2, round(w * scale)), max(2, round(h * scale))


def _depth_array(depth_bytes: bytes, size: tuple[int, int], smooth: float = 1.0) -> np.ndarray:
    """Depth map resized to `size`, normalised to 0 (far) .. 1 (near)."""
    try:
        depth = Image.open(io.BytesIO(depth_bytes))
    except Exception as exc:  # noqa: BLE001
        raise GeometryError("Mapa de profundidad no válido") from exc
    if depth.mode.startswith("I"):  # 16/32-bit integer depth maps
        depth = depth.convert("I")
    img = depth.convert("F").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=float)
    if smooth:
        arr = gaussian_filter(arr, smooth)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)  # ignore outliers
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def _grid_faces(w: int, h: int) -> np.ndarray:
    """Two triangles per cell of a w×h vertex grid (row-major), facing +Y when laid on XZ."""
    r, c = np.mgrid[0 : h - 1, 0 : w - 1]
    a = (r * w + c).ravel()
    b, cc, d = a + w, a + 1, a + w + 1
    return np.concatenate([np.column_stack([a, b, cc]), np.column_stack([cc, b, d])])


# ---------------------------------------------------------------- scene 2.5D


def depth_scene(image_bytes: bytes, depth_bytes: bytes, resolution: int = 256, strength: float = 0.4,
                cut_edges: bool = True) -> bytes:
    """Textured, camera-facing depth mesh (X right, Y up, +Z towards the viewer)."""
    image = open_rgb(image_bytes)
    w, h = _grid(image, int(np.clip(resolution, 32, 512)))
    depth = _depth_array(depth_bytes, (w, h))

    aspect_w, aspect_h = (1.0, h / w) if w >= h else (w / h, 1.0)
    xs = np.linspace(-aspect_w / 2, aspect_w / 2, w)
    ys = np.linspace(aspect_h, 0, h)  # row 0 = top of the photo
    gx, gy = np.meshgrid(xs, ys)
    gz = depth * float(np.clip(strength, 0.02, 2.0)) * max(aspect_w, aspect_h)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # rows run downwards (-Y) and columns right (+X), so the grid triangles face the viewer (+Z)
    faces = _grid_faces(w, h)
    if cut_edges:
        # drop the "rubber sheet" triangles that stretch across depth discontinuities
        flat = depth.ravel()
        spread = np.ptp(flat[faces], axis=1)
        faces = faces[spread < 0.12]
    if not len(faces):
        raise GeometryError("La profundidad es demasiado irregular; desactiva «cortar bordes»")

    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    texture = image.copy()
    texture.thumbnail((MAX_TEXTURE, MAX_TEXTURE))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True
    )
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([u.ravel(), v.ravel()]), material=material)
    mesh.remove_unreferenced_vertices()
    return mesh_to_glb(mesh)


# ---------------------------------------------------------------- printable plaques


def heightmap_solid(height: np.ndarray, colors: np.ndarray | None, base: float, relief: float,
                    width: float = 1.0) -> trimesh.Trimesh:
    """Watertight plaque lying on XZ: flat bottom at y=0, top at base + height·relief.

    `height` is h×w in 0..1; row 0 (top of the image) ends up at -Z so the plaque reads correctly
    from above.
    """
    h, w = height.shape
    depth_len = width * (h - 1) / (w - 1)
    xs = np.linspace(-width / 2, width / 2, w)
    zs = np.linspace(-depth_len / 2, depth_len / 2, h)
    gx, gz = np.meshgrid(xs, zs)
    top = np.column_stack([gx.ravel(), (base + height * relief).ravel(), gz.ravel()])
    bottom = top.copy()
    bottom[:, 1] = 0.0
    n = len(top)

    grid = _grid_faces(w, h)
    faces = [grid, grid[:, ::-1] + n]
    # side walls along the perimeter, walked in order
    ring = np.concatenate([
        np.arange(0, w),                       # top row, left → right
        np.arange(1, h) * w + (w - 1),         # right column, downwards
        (h - 1) * w + np.arange(w - 2, -1, -1),  # bottom row, right → left
        np.arange(h - 2, 0, -1) * w,           # left column, upwards
    ])
    # the ring runs clockwise seen from above, so (p, q, q') winds outwards
    p, q = ring, np.roll(ring, -1)
    faces.append(np.column_stack([p, q, q + n]))
    faces.append(np.column_stack([p, q + n, p + n]))

    mesh = trimesh.Trimesh(np.vstack([top, bottom]), np.vstack(faces), process=False)
    if colors is not None:
        rgba = np.hstack([colors.reshape(-1, 3), np.full((n, 1), 255)]).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.vstack([rgba, np.tile(WALL_COLOR, (n, 1))]))
    return mesh


def relief(image_bytes: bytes, depth_bytes: bytes, resolution: int = 200, strength: float = 0.4,
           base_ratio: float = 0.03) -> bytes:
    image = open_rgb(image_bytes)
    w, h = _grid(image, int(np.clip(resolution, 32, 400)))
    height = _depth_array(depth_bytes, (w, h), smooth=1.2)
    colors = np.asarray(image.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    relief_height = 0.12 * float(np.clip(strength, 0.05, 2.0))
    return mesh_to_glb(heightmap_solid(height, colors, base=base_ratio, relief=relief_height))


def lithophane(image_bytes: bytes, resolution: int = 250, thickness_ratio: float = 0.03,
               min_ratio: float = 0.008) -> bytes:
    """Dark areas thick, light areas thin: the photo appears when light shines through."""
    image = open_rgb(image_bytes)
    w, h = _grid(image, int(np.clip(resolution, 32, 400)))
    lum = np.asarray(ImageOps.autocontrast(image.convert("L")).resize((w, h), Image.BILINEAR), dtype=float) / 255
    height = 1.0 - lum
    white = np.full((h, w, 3), 245, dtype=np.uint8)
    return mesh_to_glb(heightmap_solid(height, white, base=min_ratio, relief=thickness_ratio - min_ratio))


# ---------------------------------------------------------------- 3D text

FONTS = {
    "sans": "DejaVu Sans",
    "serif": "DejaVu Serif",
    "mono": "DejaVu Sans Mono",
}


def _text_shape(text: str, font: str, bold: bool):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    shape = None
    for line_no, line in enumerate(text.splitlines() or [text]):
        if not line.strip():
            continue
        prop = FontProperties(family=FONTS.get(font, FONTS["sans"]), weight="bold" if bold else "normal")
        path = TextPath((0, -1.25 * line_no), line, size=1, prop=prop)
        line_shape = None
        # glyph outlines and their holes: even-odd composition handles "o", "A", "B"…
        for ring in path.to_polygons():
            if len(ring) < 3:
                continue
            poly = Polygon(ring).buffer(0)
            if poly.is_empty:
                continue
            line_shape = poly if line_shape is None else line_shape.symmetric_difference(poly)
        if line_shape is not None:
            shape = line_shape if shape is None else unary_union([shape, line_shape])
    if shape is None or shape.is_empty:
        raise GeometryError("El texto no contiene caracteres visibles")
    return shape


def _extrude(shape, height: float) -> list[trimesh.Trimesh]:
    polys = getattr(shape, "geoms", [shape])
    return [trimesh.creation.extrude_polygon(p, height) for p in polys if p.area > 1e-6]


def _paint(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile([*rgb, 255], (len(mesh.vertices), 1)))
    return mesh


def text3d(text: str, font: str = "sans", bold: bool = True, style: str = "letters", depth: float = 0.25,
           color: tuple[int, int, int] = (124, 92, 255), plate_color: tuple[int, int, int] = (240, 240, 240)) -> bytes:
    """Extruded text. `letters` stands upright; `plate` and `keychain` lie flat, ready to print."""
    from shapely.geometry import Point, box

    text = text.strip()
    if not text:
        raise GeometryError("Escribe algún texto")
    shape = _text_shape(text, font, bold)
    letters = [_paint(m, color) for m in _extrude(shape, float(np.clip(depth, 0.02, 2.0)))]

    if style == "letters":
        parts = letters
    elif style in ("plate", "keychain"):
        minx, miny, maxx, maxy = shape.bounds
        margin = 0.25 * (maxy - miny) / max(1, text.count("\n") + 1) + 0.15
        plate = box(minx - margin, miny - margin, maxx + margin, maxy + margin).buffer(0.15, join_style=1)
        if style == "keychain":
            radius = 0.18
            tab_center = (minx - margin - radius * 1.3, (miny + maxy) / 2)
            plate = plate.union(Point(tab_center).buffer(radius * 2.1)).difference(Point(tab_center).buffer(radius))
        thickness = 0.15
        base = [_paint(m, plate_color) for m in _extrude(plate, thickness)]
        for m in letters:  # letters sit on top of the plate
            m.apply_translation([0, 0, thickness])
        parts = base + letters
    else:
        raise GeometryError(f"Estilo de texto desconocido: {style}")

    mesh = trimesh.util.concatenate(parts)
    if style != "letters":
        # lie flat: extrusion (+Z) becomes up (+Y), the top of the text points to -Z
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -bounds[0][1], -center[2]])
    mesh.apply_scale(1.0 / max(mesh.extents[0], 1e-6))  # 1 unit wide; exports rescale
    return mesh_to_glb(mesh)
