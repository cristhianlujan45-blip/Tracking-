import numpy as np
import trimesh
from PIL import Image, ImageDraw

from app.projection import ProjectionError, project_views


def _view(color, size=128, stripe=None):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, size - 8, size - 8], fill=color + (255,))
    if stripe:  # a mark on the image's left half
        d.rectangle([8, size // 2 - 6, size // 2 - 10, size // 2 + 6], fill=stripe + (255,))
    return img


def test_views_land_on_the_right_side_and_unseen_parts_are_filled():
    mesh = trimesh.creation.icosphere(subdivisions=5)
    red, blue, green, white = (220, 30, 30), (30, 30, 220), (30, 200, 30), (250, 250, 250)
    out = project_views(mesh, {"front": _view(red, stripe=white), "back": _view(blue), "left": _view(green)})
    colors = out.visual.vertex_colors[:, :3].astype(int)
    v = out.vertices

    def mean(sel):
        return colors[sel].mean(0)

    assert mean(v[:, 2] > 0.9)[0] > 150  # front → red
    assert mean(v[:, 2] < -0.9)[2] > 150  # back → blue
    assert mean(v[:, 0] > 0.9)[1] > 150  # model's left side (+X) → green
    assert mean(v[:, 0] < -0.9)[1] > 150  # right side mirrored from the left
    # the stripe drawn on the image's left half of the front view lands on the model's right (-X)
    front_band = (v[:, 2] > 0.6) & (np.abs(v[:, 1]) < 0.06)
    assert colors[front_band & (v[:, 0] < -0.3)].mean() > colors[front_band & (v[:, 0] > 0.3)].mean() + 40
    assert (colors.sum(1) > 0).all()  # top and bottom poles filled by diffusion


def test_requires_views():
    try:
        project_views(trimesh.creation.box(), {})
    except ProjectionError:
        return
    raise AssertionError("expected ProjectionError")
