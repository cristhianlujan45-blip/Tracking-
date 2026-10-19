"""Prueba del complemento con Blender real contra una app Forja3D en marcha.

    blender -b --python integrations/blender/smoke_test.py -- http://localhost:8000
    # o con Blender como módulo de Python (pip install bpy, Python 3.11):
    python integrations/blender/smoke_test.py http://localhost:8000

Necesita al menos un modelo terminado en la app. Instala el complemento, importa cada modelo
y envía un objeto de vuelta como modelo subido.
"""

import json
import os
import sys
import urllib.request

import bpy  # noqa: I001 - addon_utils only exists once bpy is imported
import addon_utils

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
server = (args[0] if args else "http://localhost:8000").rstrip("/")

addon_utils.enable("io_scene_gltf2", default_set=True)
bpy.ops.preferences.addon_install(filepath=os.path.join(os.path.dirname(__file__), "forja3d_blender.py"), overwrite=True)
assert addon_utils.enable("forja3d_blender", default_set=True), "no se pudo activar el complemento"
bpy.context.preferences.addons["forja3d_blender"].preferences.server_url = server

assert bpy.ops.forja.refresh() == {"FINISHED"}
items = bpy.context.scene.forja_items
assert len(items), "la app no tiene modelos terminados"
meshes_before = {o for o in bpy.data.objects if o.type == "MESH"}
for i in range(len(items)):
    bpy.context.scene.forja_index = i
    assert bpy.ops.forja.import_model() == {"FINISHED"}, items[i].title
imported = [o for o in bpy.data.objects if o.type == "MESH" and o not in meshes_before]
print(f"Importados {len(items)} modelos ({len(imported)} mallas)")

before = len(json.loads(urllib.request.urlopen(f"{server}/api/jobs").read()))
bpy.ops.object.select_all(action="DESELECT")
imported[0].select_set(True)
bpy.context.view_layer.objects.active = imported[0]
assert bpy.ops.forja.send_selection() == {"FINISHED"}
after = len(json.loads(urllib.request.urlopen(f"{server}/api/jobs").read()))
assert after == before + 1, "el envío no creó un trabajo nuevo"
print("OK: complemento de Blender verificado")
