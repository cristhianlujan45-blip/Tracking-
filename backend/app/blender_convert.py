"""Runs inside Blender (or a Python with `bpy`): GLB in, FBX or .blend out.

    blender -b --factory-startup --python blender_convert.py -- in.glb out.fbx
"""

import sys

import bpy  # noqa: I001 - addon_utils needs bpy imported first
import addon_utils

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
src, dst = args[-2], args[-1]

addon_utils.enable("io_scene_gltf2", default_set=True)
addon_utils.enable("io_scene_fbx", default_set=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
if src.lower().endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=src)  # keeps quads and vertex colours
else:
    bpy.ops.import_scene.gltf(filepath=src)

# the glTF importer adds a display shape (an icosphere) for the bones; it is not part of the model
shapes = set()
for obj in bpy.data.objects:
    if obj.type == "ARMATURE":
        for bone in obj.pose.bones:
            if bone.custom_shape:
                shapes.add(bone.custom_shape)
                bone.custom_shape = None
for shape in shapes:
    bpy.data.objects.remove(shape, do_unlink=True)


def arrange_clips(objects, gap: int = 10) -> int:
    """Put every animation clip on its own numbered NLA track ("01 idle", "02 walk"…), one after
    another on the timeline, so all of them play and stay editable. Returns the number of clips."""
    count, last = 0, 1
    for obj in objects:
        ad = obj.animation_data if obj.type == "ARMATURE" else None
        if ad is None or not ad.nla_tracks:
            continue
        # the glTF importer adds one muted track per clip, newest on top: reverse for file order
        actions = [s.action for t in reversed(list(ad.nla_tracks)) for s in t.strips if s.action]
        for track in list(ad.nla_tracks):
            ad.nla_tracks.remove(track)
        start = 1
        for i, action in enumerate(actions, 1):
            track = ad.nla_tracks.new()
            track.name = f"{i:02d} {action.name}"
            strip = track.strips.new(track.name, start, action)
            start = int(strip.frame_end) + gap
        ad.action = None
        count, last = max(count, len(actions)), max(last, start - gap)
    if count:
        bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, last
    return count



if dst.lower().endswith(".fbx"):
    bpy.ops.export_scene.fbx(
        filepath=dst,
        path_mode="COPY",
        embed_textures=True,
        add_leaf_bones=False,  # our skeleton already has *_End joints
        bake_anim=True,
        bake_anim_use_all_actions=True,
        bake_anim_use_nla_strips=False,
        bake_anim_simplify_factor=0.0,
        axis_forward="-Z",
        axis_up="Y",
    )
else:
    arrange_clips(bpy.data.objects)  # .blend: clips laid out as NLA strips, ready to edit
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
print("forja3d-convert: ok")
