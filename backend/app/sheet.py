"""Character sheets: find the full-body views on a model sheet (front, side, back…) and cut each one
out with a transparent background, ready for multi-view 3D generation and view projection.

Works on sheets with a plain background (the usual turnaround layout). Background colour comes from
the image border; GrabCut then separates the figure from it, which keeps dark shading on the
character while leaving real gaps (between the legs) transparent.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


class SheetError(Exception):
    pass


def _biggest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if not n:
        return mask
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    return lab == ids[np.argmax(counts)]


def _background(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    return np.median(border.astype(float), axis=0)


def _cutout(rgb: np.ndarray, dist: np.ndarray, box: tuple[int, int, int, int], pad: int = 25) -> Image.Image:
    h, w = dist.shape
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    sub = dist[Y0:Y1, X0:X1]
    core = _biggest(ndi.binary_closing(sub > 22, iterations=2))
    gc = np.full(sub.shape, cv2.GC_PR_BGD, np.uint8)
    gc[ndi.binary_dilation(core, iterations=6)] = cv2.GC_PR_FGD
    gc[ndi.binary_erosion(core, iterations=4) & (sub > 30)] = cv2.GC_FGD
    gc[~ndi.binary_dilation(core, iterations=15)] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).any() and (gc == cv2.GC_BGD).any():
        bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(np.ascontiguousarray(rgb[Y0:Y1, X0:X1, ::-1]), gc, None, bgm, fgm, 5, cv2.GC_INIT_WITH_MASK)
        mask = _biggest((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    else:
        mask = core
    holes = ndi.binary_fill_holes(mask) & ~mask
    lab, n = ndi.label(holes)
    for j in range(1, n + 1):  # fill holes in the body, keep real gaps that show the background
        hole = lab == j
        if hole.sum() < 300 or sub[hole].mean() > 14:
            mask |= hole
    alpha = (ndi.gaussian_filter(mask.astype(float), 0.7) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb[Y0:Y1, X0:X1], alpha]), "RGBA")


def _square(img: Image.Image, size: int = 768) -> Image.Image:
    side = int(max(img.size) * 1.08)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(img, ((side - img.size[0]) // 2, (side - img.size[1]) // 2))
    return out.resize((size, size), Image.LANCZOS)


def split_sheet(data: bytes, max_views: int = 6) -> list[dict]:
    """Full-body figures on a sheet, left to right: [{"box", "role", "image" (RGBA PNG bytes)}].

    Roles are a guess for the common turnaround order: the first wide figure is the front, the
    narrowest one the side, the next wide one the back; anything else is an extra view (e.g. 3/4)."""
    try:
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        raise SheetError("No se pudo leer la imagen") from exc
    scale = 1.0
    if max(rgb.shape[:2]) > 2048:  # keep it fast on huge sheets
        scale = 2048 / max(rgb.shape[:2])
        rgb = np.asarray(Image.fromarray(rgb).resize((int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)), Image.LANCZOS))
    h, w = rgb.shape[:2]
    dist = np.linalg.norm(rgb.astype(float) - _background(rgb), axis=2)
    mask = ndi.binary_fill_holes(ndi.binary_closing(dist > 22, iterations=3))
    lab, n = ndi.label(mask)
    if not n:
        raise SheetError("No se encontraron figuras: usa una hoja con fondo liso")
    boxes = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if (lab[sl] == i).sum() > 0.004 * h * w and bh > 0.12 * h:
            boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    if not boxes:
        raise SheetError("No se encontraron figuras completas en la imagen")
    tallest = max(b[3] - b[1] for b in boxes)
    # the turnaround: the tallest figures, on the same row as the tallest one
    row = [b for b in boxes if (b[3] - b[1]) >= 0.75 * tallest]
    ref = max(row, key=lambda b: b[3] - b[1])
    row = sorted((b for b in row if b[1] < ref[3] and b[3] > ref[1]), key=lambda b: b[0])[:max_views]

    ratios = [(b[2] - b[0]) / (b[3] - b[1]) for b in row]
    roles = ["extra"] * len(row)
    if len(row) >= 2:
        side = int(np.argmin(ratios))
        if ratios[side] < 0.85 * np.median(ratios) or len(row) == 2:
            roles[side] = "left"
        wide = [i for i in range(len(row)) if roles[i] == "extra"]
        if wide:
            roles[wide[0]] = "front"
        if len(wide) > 1 and roles.count("left") and wide[1] > roles.index("left"):
            roles[wide[1]] = "back"
    else:
        roles[0] = "front"
    out = []
    for box, role in zip(row, roles):
        buf = io.BytesIO()
        _square(_cutout(rgb, dist, box)).save(buf, format="PNG")
        out.append({"box": [int(round(c / scale)) for c in box], "role": role, "image": buf.getvalue()})
    return out


def remove_background(image: Image.Image) -> Image.Image:
    """RGBA cut-out of a single picture with a plain background (keeps existing transparency)."""
    if image.mode in ("RGBA", "LA") and np.asarray(image.getchannel("A")).min() < 250:
        return image.convert("RGBA")
    rgb = np.asarray(image.convert("RGB"))
    dist = np.linalg.norm(rgb.astype(float) - _background(rgb), axis=2)
    ys, xs = np.nonzero(dist > 22)
    if not len(xs):
        return image.convert("RGBA")
    return _cutout(rgb, dist, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
