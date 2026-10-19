"""Forja3D para Blender: importa tus modelos generados con un clic y envía objetos de vuelta.

Instalación: Edit › Preferences › Add-ons › Install from Disk… → elige este archivo.
Uso: Vista 3D › barra lateral (tecla N) › pestaña «Forja3D».
"""

bl_info = {
    "name": "Forja3D",
    "author": "Forja3D",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "Vista 3D › Barra lateral (N) › Forja3D",
    "description": "Importa modelos de tu servidor Forja3D y envía objetos para remallar, reparar o imprimir",
    "category": "Import-Export",
}

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import webbrowser

import bpy
from bpy.props import CollectionProperty, FloatProperty, IntProperty, StringProperty

DEFAULT_SERVER = "http://localhost:8000"  # la app lo sustituye al descargar el complemento
MODE_ICONS = {
    "text": "FONT_DATA", "image": "IMAGE_DATA", "multi_image": "RENDERLAYERS", "upload": "IMPORT",
    "remesh": "MOD_REMESH", "retexture": "BRUSH_DATA", "repair": "MODIFIER", "split": "MOD_EXPLODE",
    "scene": "WORLD", "relief": "MOD_DISPLACE", "lithophane": "LIGHT", "text3d": "OUTLINER_OB_FONT",
}


# ---------------------------------------------------------------- HTTP helpers


def addon_prefs(context):
    addon = context.preferences.addons.get(__name__)  # None when run as a plain script
    return addon.preferences if addon else None


def server_url(context) -> str:
    prefs = addon_prefs(context)
    return ((prefs.server_url if prefs else "") or DEFAULT_SERVER).rstrip("/")


def http(url: str, data: bytes | None = None, headers: dict | None = None, method: str | None = None,
         timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except Exception:  # noqa: BLE001
            detail = exc.reason
        raise RuntimeError(f"Forja3D respondió {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se puede conectar con {url.split('/api')[0]} ¿Está iniciada la app?") from exc


def multipart(field: str, filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------- data


class ForjaItem(bpy.types.PropertyGroup):
    job_id: StringProperty()
    title: StringProperty()
    mode: StringProperty()
    created: FloatProperty()


class ForjaPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    server_url: StringProperty(name="Servidor", default=DEFAULT_SERVER,
                               description="Dirección de tu app Forja3D")

    def draw(self, context):
        self.layout.prop(self, "server_url")


# ---------------------------------------------------------------- operators


class FORJA_OT_refresh(bpy.types.Operator):
    bl_idname = "forja.refresh"
    bl_label = "Actualizar"
    bl_description = "Descarga la lista de modelos terminados"

    def execute(self, context):
        try:
            jobs = json.loads(http(f"{server_url(context)}/api/jobs?limit=500"))
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        items = context.scene.forja_items
        items.clear()
        for job in jobs:
            if job.get("status") != "succeeded":
                continue
            item = items.add()
            item.job_id, item.title, item.mode = job["id"], job["title"][:120], job["mode"]
            item.created = job.get("created_at", 0.0)
        context.scene.forja_index = min(context.scene.forja_index, max(len(items) - 1, 0))
        self.report({"INFO"}, f"{len(items)} modelos")
        return {"FINISHED"}


class FORJA_OT_import(bpy.types.Operator):
    bl_idname = "forja.import_model"
    bl_label = "Importar"
    bl_description = "Importa el modelo seleccionado en la escena (en el cursor 3D)"
    bl_options = {"REGISTER", "UNDO"}

    job_id: StringProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        scene = context.scene
        job_id = self.job_id
        if not job_id:
            if not scene.forja_items:
                self.report({"WARNING"}, "Pulsa «Actualizar» primero")
                return {"CANCELLED"}
            job_id = scene.forja_items[scene.forja_index].job_id
        title = next((i.title for i in scene.forja_items if i.job_id == job_id), "Forja3D")
        try:
            data = http(f"{server_url(context)}/api/jobs/{job_id}/model.glb", timeout=300)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        path = os.path.join(tempfile.gettempdir(), f"forja3d-{job_id}.glb")
        with open(path, "wb") as fh:
            fh.write(data)
        before = set(bpy.data.objects)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.import_scene.gltf(filepath=path)
        new_objects = [o for o in bpy.data.objects if o not in before]
        roots = [o for o in new_objects if o.parent not in new_objects]

        # place on the 3D cursor and optionally scale to a target height
        target = scene.forja_height
        height = 0.0
        if target > 0:
            zs = [(o.matrix_world @ v.co).z for o in new_objects if o.type == "MESH" for v in o.data.vertices]
            height = (max(zs) - min(zs)) if zs else 0.0
        for obj in roots:
            obj.location = scene.cursor.location.copy()
            if target > 0 and height > 0:
                obj.scale = [s * target / height for s in obj.scale]
        for obj in new_objects:
            obj.select_set(True)
            if obj.type == "MESH" and len(new_objects) == 1:
                obj.name = title[:60]
        if roots:
            context.view_layer.objects.active = roots[0]
        self.report({"INFO"}, f"Importado: {title}")
        return {"FINISHED"}


class FORJA_OT_send(bpy.types.Operator):
    bl_idname = "forja.send_selection"
    bl_label = "Enviar selección a Forja3D"
    bl_description = "Exporta los objetos seleccionados como GLB y los sube a la app"

    @classmethod
    def poll(cls, context):
        return any(o.type == "MESH" for o in context.selected_objects)

    def execute(self, context):
        name = (context.active_object or context.selected_objects[0]).name
        path = os.path.join(tempfile.gettempdir(), f"forja3d-send-{int(time.time())}.glb")
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)
        with open(path, "rb") as fh:
            body, content_type = multipart("model", f"{name}.glb", fh.read(), "model/gltf-binary")
        try:
            job = json.loads(http(f"{server_url(context)}/api/jobs/upload", data=body,
                                  headers={"Content-Type": content_type}, method="POST", timeout=300))
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            os.remove(path)
        self.report({"INFO"}, f"Enviado a Forja3D: {job.get('title', name)}")
        bpy.ops.forja.refresh()
        return {"FINISHED"}


class FORJA_OT_open_web(bpy.types.Operator):
    bl_idname = "forja.open_web"
    bl_label = "Abrir Forja3D"
    bl_description = "Abre la app en el navegador"

    def execute(self, context):
        webbrowser.open(server_url(context))
        return {"FINISHED"}


# ---------------------------------------------------------------- UI


class FORJA_UL_jobs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.title, icon=MODE_ICONS.get(item.mode, "MESH_DATA"))

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        needle = context.scene.forja_search.lower()
        flags = [self.bitflag_filter_item if needle in i.title.lower() else 0 for i in items]
        return flags, []


class FORJA_PT_panel(bpy.types.Panel):
    bl_label = "Forja3D"
    bl_idname = "FORJA_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Forja3D"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        prefs = addon_prefs(context)
        if prefs:
            layout.prop(prefs, "server_url", text="")
        else:
            layout.label(text=DEFAULT_SERVER, icon="URL")
        row = layout.row(align=True)
        row.operator("forja.refresh", icon="FILE_REFRESH")
        row.operator("forja.open_web", text="", icon="URL")
        layout.prop(scene, "forja_search", text="", icon="VIEWZOOM")
        layout.template_list("FORJA_UL_jobs", "", scene, "forja_items", scene, "forja_index", rows=8)
        layout.prop(scene, "forja_height")
        layout.operator("forja.import_model", icon="IMPORT")
        layout.separator()
        layout.operator("forja.send_selection", icon="EXPORT")


CLASSES = (ForjaItem, ForjaPreferences, FORJA_OT_refresh, FORJA_OT_import, FORJA_OT_send, FORJA_OT_open_web,
           FORJA_UL_jobs, FORJA_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.forja_items = CollectionProperty(type=ForjaItem)
    bpy.types.Scene.forja_index = IntProperty(default=0)
    bpy.types.Scene.forja_search = StringProperty(name="Buscar", default="")
    bpy.types.Scene.forja_height = FloatProperty(
        name="Altura al importar", default=0.0, min=0.0, unit="LENGTH",
        description="Escala el modelo a esta altura (0 = tamaño original)",
    )


def unregister():
    for attr in ("forja_items", "forja_index", "forja_search", "forja_height"):
        delattr(bpy.types.Scene, attr)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
