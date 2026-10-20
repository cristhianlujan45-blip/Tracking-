import numpy as np
import trimesh
from PIL import Image, ImageDraw

from app.projection import AZIMUTHS, VIEWS, ProjectionError, bake_views, project_views, view_axes


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


def test_view_axes_match_the_four_views():
    for name, az in AZIMUTHS.items():
        for got, want in zip(view_axes(az), VIEWS[name]):
            assert np.allclose(got, want)
    cam, right, up = view_axes(90, 20)  # looking down at the left side: image up leans away from the camera
    assert cam[1] > 0.3 and up[1] > 0.9 and up[0] < -0.3 and abs(right @ up) < 1e-9


def test_posed_views_paint_where_they_were_seen_from():
    """Pictures taken between the axes (AI views) paint around their own azimuth, not the nearest axis."""
    mesh = trimesh.creation.icosphere(subdivisions=5)
    red, green = (220, 30, 30), (30, 200, 30)
    out = project_views(mesh, {"front": _view(red), "left": _view(green)}, mirror_side=False,
                        poses={"front": (-45, 0), "left": (45, 0)}, trust={"left": 0.6})
    colors = out.visual.vertex_colors[:, :3].astype(int)
    v = out.vertices
    toward = lambda az: v @ np.array([np.sin(np.radians(az)), 0, np.cos(np.radians(az))]) > 0.9
    assert colors[toward(45)][:, 1].mean() > 150  # green around azimuth 45 (front-left)
    assert colors[toward(-45)][:, 0].mean() > 150  # red around azimuth -45 (front-right)


def test_requires_views():
    try:
        project_views(trimesh.creation.box(), {})
    except ProjectionError:
        return
    raise AssertionError("expected ProjectionError")


def test_single_picture_paints_the_back_without_copying_details():
    """One front picture: the back takes its colour regions but not its details (no second face)."""
    mesh = trimesh.creation.icosphere(subdivisions=5)
    img = _view((40, 120, 220))
    ImageDraw.Draw(img).ellipse([59, 40, 69, 50], fill=(250, 250, 250, 255))  # an "eye" in the middle
    out = project_views(mesh, {"front": img})
    colors = out.visual.vertex_colors[:, :3].astype(int)
    v = out.vertices
    eye = lambda z: (np.abs(v[:, 0]) < 0.05) & (v[:, 1] > 0.33) & (v[:, 1] < 0.45) & (z * v[:, 2] > 0.7)
    assert colors[eye(1)].mean() > 180  # the eye on the front
    back_eye = colors[eye(-1)]
    assert back_eye[:, 2].mean() > 150 and back_eye.mean() < 170  # blue on the back, no white eye


def test_baked_texture_keeps_detail_finer_than_the_vertices():
    """A thin stripe narrower than the triangles survives in the baked texture."""
    mesh = trimesh.creation.icosphere(subdivisions=3)  # ~640 faces: coarse
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 248, 248], fill=(30, 160, 60, 255))
    d.rectangle([8, 126, 248, 130], fill=(255, 255, 255, 255))  # 5 px stripe across the equator
    out = bake_views(mesh, {"front": img, "back": img}, size=512)
    assert out.visual.kind == "texture" and out.visual.material.baseColorTexture.size == (512, 512)
    tex = np.asarray(out.visual.material.baseColorTexture.convert("RGB"), dtype=int)
    # sample the texture where the front equator and a point above it land
    def colour_at(point):  # texture colour at the point on the surface closest to `point`
        closest, _, face = trimesh.proximity.closest_point(out, [point])
        bary = trimesh.triangles.points_to_barycentric(out.triangles[face], closest)[0]
        uv = bary @ out.visual.uv[out.faces[face[0]]]
        return tex[int((1 - uv[1]) * 511), int(uv[0] * 511)]
    assert colour_at(np.array([0, 0.0, 1.0])).mean() > 180  # white stripe kept
    assert colour_at(np.array([0, 0.5, 0.86]))[1] > 120  # green above it
