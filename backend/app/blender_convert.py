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
    bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
print("forja3d-convert: ok")
