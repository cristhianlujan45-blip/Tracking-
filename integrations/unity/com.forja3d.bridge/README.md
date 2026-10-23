# Forja3D Bridge para Unity

1. En Unity: **Window → Package Manager → + → Add package from disk…** y elige `package.json` de esta carpeta
   (o descarga el paquete ya configurado desde Forja3D: `/api/integrations/unity`).
2. Abre **Window → Forja3D**, escribe la dirección de Forja3D (por defecto `http://127.0.0.1:8000`).
3. Genera desde texto o pulsa **Importar** en cualquier modelo terminado. Se guarda en `Assets/Forja3D/`.

FBX conserva el esqueleto Humanoid y una toma por animación (Forja3D lo crea con Blender instalado).
GLB funciona sin Blender, pero Unity necesita el paquete **glTFast** para leerlo.
