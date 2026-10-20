# Forja3D — estudio 3D con IA

Programa propio para crear, texturizar, riggear, animar y exportar modelos 3D con IA, al estilo de Meshy o Tripo3D. Funciona como **aplicación de escritorio para Windows y macOS** o como web autoalojada. Interfaz en **español e inglés**.

## Instalar (Windows y Mac)

Descarga la última versión desde la página **Releases** del repositorio (se compila sola con GitHub Actions):

- **Windows 10/11 (64 bits):** `Forja3D-Setup-x.y.z.exe` → ejecútalo. Crea el acceso directo y abre la app en su propia ventana. Si SmartScreen avisa, pulsa «Más información» → «Ejecutar de todas formas» (el instalador no está firmado).
- **macOS (Apple Silicon):** `Forja3D-x.y.z-mac-arm64.dmg` → arrastra Forja3D a Aplicaciones. La primera vez: clic derecho → «Abrir» (no está firmada por Apple).

**IA gratis en tu GPU, sin configurar nada.** En Windows con una NVIDIA RTX (8 GB bastan, incluida la serie 50) la app instala sola su IA la primera vez que la abres (≈30 GB, una única vez, con barra de progreso) y la arranca cada vez que generas: sin claves, sin tokens, sin scripts y sin límite de generaciones. Usa la máxima calidad que cabe en 8 GB: Hunyuan3D-2 (forma) con FlashVDM, Hunyuan3D-2mv (varias vistas), Zero123++ (laterales y espalda de una sola imagen), SDXL-Turbo (conceptos), Depth Anything V2 Base y Real-ESRGAN, y traduce sola tus descripciones en español. También se puede instalar o reparar desde **Ajustes → Mi GPU**.

La nube (fal.ai: Hunyuan3D 3, TRELLIS 2, Tripo, Meshy) es opcional y se paga por uso a fal.ai con tu clave. Tus modelos y ajustes se guardan en `%APPDATA%\Forja3D` (Windows) o `~/Library/Application Support/Forja3D` (Mac). En Mac (sin NVIDIA) usa la nube o el modo demo.

## Funciones

**Crear**
- **Texto → 3D** con 7 estilos, simetría, pose A/T y semilla; **conceptos primero** (4 imágenes, eliges una y pasa a 3D).
- **Imagen → 3D** y **multi-vista → 3D** (frente, lateral, espalda, lateral derecho).
- **Vistas laterales y de espalda con IA**: con una sola imagen (o tras el concepto de Texto → 3D), la IA local genera con Zero123++ los dos lados y la espalda, que alimentan la forma multi-vista y el pintado del modelo en lugar de inventar la parte de atrás.
- **Hoja de personaje**: sube un *model sheet* y Forja3D encuentra y recorta solo las vistas de frente, lado y espalda (con fondo transparente) para generar el modelo multi-vista.
- Motores de primera línea en la nube: **Hunyuan3D 3**, **TRELLIS 2**, **Tripo H3.1** y **Meshy 6** (PBR, hasta 1 M de caras); en tu GPU, **Hunyuan3D-2mini/2mv**.
- **Escena / foto → 3D**: escena 2.5D, relieve y litofanía a partir de la profundidad estimada con IA.
- **Texto 3D** (letras, placas, llaveros) y **subida de modelos** (GLB, OBJ, STL, PLY, OFF, 3MF).

**Texturas**
- **Full HD 2K / 4K / 8K**: ampliación con IA (Real-ESRGAN) y mapas **PBR** (normal, rugosidad, oclusión). Si el motor ya entregó mapas PBR reales, se conservan y se amplían en lugar de sustituirlos.
- **Retexturizado IA** (Meshy 7 hasta 8K) desde una descripción.
- **Pintado desde las vistas**: cuando el motor devuelve solo la forma (Hunyuan3D-2mini en GPUs de 8 GB), el modelo se colorea proyectando tus imágenes: cada vista se alinea con la silueta del modelo, solo pinta las superficies que ve de frente (nada de manchas estiradas en los costados) y sin los bordes del dibujo; lo que ninguna vista ve se rellena por difusión. El resultado se hornea en una **textura UV real de 2048 px** (lista para ampliar a 4K/8K con PBR), y con una sola foto la espalda toma los colores de los bordes de la figura (pelo, ropa, piel) sin copiar la cara ni otros detalles.
- **Inspector de materiales** en el visor: ver solo el color, normal, rugosidad, metal u oclusión, y la malla sobre la textura.

**Personajes: rig y animación**
- **Personaje completo en un clic**: texturas HD + PBR → esqueleto → animaciones elegidas.
- Auto-rig de **humanos** (27 huesos con nombres Mixamo/Unity Humanoid, más **dedos** de 3 falanges colocados según la forma de la mano y **mandíbula**) y de **cuadrúpedos** (perros, gatos, caballos…; 26 huesos), con detección automática.
- **Cara**: *blendshapes* procedurales (abrir la boca, sonrisa y parpadeo cuando se ven los ojos en el color o la textura), con clips Hablar y Parpadear; en Blender/FBX llegan como *shape keys*.
- **Marcadores ajustables** estilo Mixamo (vista frontal para personas, lateral para animales).
- Pesos por difusión de calor (*bone heat*) con visibilidad: una barriga grande no se mueve con los brazos que cuelgan a su lado (criaturas, monstruos), máximo 4 huesos por vértice; visor de huesos y mapa de pesos.
- **650+ animaciones**: 19 procedurales (también puño, señalar y agarrar con los dedos), 6 andares de cuadrúpedo y una **biblioteca mocap de 639 clips** (CMU, en español e inglés) en 15 categorías con buscador; además tus propios **BVH**.
- El personaje riggeado conserva **todos los mapas PBR y tangentes** (validado con el validador oficial de Khronos: 0 errores, 0 avisos).

**Exportar**
- **Destinos con un clic**: Blender, Unity, Unreal, Godot, Roblox, web/AR e impresora 3D (elige el mejor formato disponible).
- GLB con animaciones, **FBX** y **.blend** (con Blender instalado): en el `.blend` cada animación queda en su **pista NLA numerada** (01 idle, 02 walk…), lista para editar.
- OBJ (quads reales tras el remallado en quads), STL, 3MF, PLY, USDZ (iPhone/AR).
- **Impresión multicolor**: 3MF pintado por colores (Bambu AMS, Prusa MMU) o un STL por color.
- **Complemento de Blender**: lista tus modelos y los importa con un clic, con las animaciones ya en pistas NLA.

**Editar e imprimir**
- Refinar HD, variaciones, remallado (reducir, **quads limpios**, low-poly, vóxel, suavizar).
- Imprimibilidad a la medida final (cerrado, agujeros, aristas no-manifold, volumen y gramos de PLA), reparación automática, **dividir para que quepa en la cama** de tu impresora (Bambu, Prusa, Creality, Elegoo, Anycubic, resina) y **abrir en tu laminador** (Bambu Studio, OrcaSlicer, PrusaSlicer, Cura, Creality Print, ElegooSlicer, Lychee…).

**Agente IA**
- Chat (botón «Agente IA», abajo a la derecha) que crea conceptos y modelos, los anima y revisa si se imprimen, con tarjetas para abrir cada resultado. Funciona **gratis sin clave** con un asistente integrado; con una clave de Anthropic (opcional, Ajustes) entiende cualquier petición y responde como un experto en 3D, Blender e impresión.

**Claude Code / Cursor (MCP)**
- Servidor **MCP** para manejar Forja3D desde Claude Code, Cursor o Claude Desktop (como el de Meshy): crear conceptos y modelos desde texto o imagen, esperar el resultado, buscar animaciones, riggear y animar personajes, revisar la imprimibilidad y **exportar el archivo directamente a tu proyecto** (GLB, FBX, .blend, STL, 3MF…).
- Un solo archivo (`integrations/mcp/forja3d_mcp.py`) que se ejecuta con [uv](https://docs.astral.sh/uv/) sin instalar nada más. En la app: **Exportar → ✦ Claude Code / Cursor (MCP) → ⧉ Copiar comando**, y pégalo en tu terminal:
  ```
  claude mcp add forja3d -- uv run <ruta>/forja3d_mcp.py
  ```
  Cursor / Claude Desktop: `{"mcpServers": {"forja3d": {"command": "uv", "args": ["run", "<ruta>/forja3d_mcp.py"]}}}` en `~/.cursor/mcp.json` o `claude_desktop_config.json`.
- Encuentra sola la app de escritorio (que anota su dirección en `server.json` de su carpeta de datos); para otro servidor define `FORJA3D_URL` (p. ej. `http://192.168.1.20:8000`).
- Herramientas: `create_concepts`, `generate_model`, `image_to_model`, `get_model`, `wait_for_model`, `list_models`, `search_animations`, `rig_and_animate`, `check_printability`, `list_printers`, `export_model`.

**Interfaz**
- Diseño neón propio, **español / inglés** (selector arriba), barra de flujo **01 Entrada → 02 Modelo texturizado → 03 Rig → 04 Blender y motores**, galería con filtros, historial de versiones, captura PNG y vídeo 360°, enlace para compartir con AR.

### Comparación con Meshy y Tripo

| | Forja3D | Meshy / Tripo |
|---|---|---|
| Texto, imagen y multi-vista → 3D con motores punteros | ✅ (Hunyuan3D 3, TRELLIS 2, Tripo H3.1, Meshy 6) | ✅ |
| Hoja de personaje → vistas recortadas automáticamente | ✅ | no figura |
| Texturas 8K + PBR | ✅ | ✅ |
| Auto-rig humano y cuadrúpedo | ✅ | ✅ |
| Corregir el rig con marcadores, ver pesos | ✅ | no figura |
| Animaciones | 650+ (639 mocap) + tus BVH | 600+ |
| Agente IA que crea, anima y revisa la impresión hablando | ✅ (gratis; más listo con clave de Anthropic) | ✅ |
| Dividir para la cama, enviar al laminador | ✅ | ✅ |
| Exportar a Blender con pistas NLA, FBX, Unity/Unreal/Godot | ✅ | ✅ |
| Impresión multicolor, quads | ✅ | ✅ |
| IA en tu propia GPU, sin pagar por modelo | ✅ | ❌ |
| Escritorio Windows/Mac, sin suscripción, el código es tuyo | ✅ | ❌ |

### Pendiente

- Firma de código (Windows y Apple) para que no aparezcan avisos al instalar; versión para Mac Intel.
- Cuentas, galería pública y app móvil.

## Proveedores de IA

Se eligen en **Ajustes** dentro de la app (o con `GEN_PROVIDER` en `.env` para servidores):

- **Nube (fal.ai)**: pega tu clave de <https://fal.ai/dashboard/keys>. Motores 3D: Hunyuan3D 3 (por defecto), TRELLIS 2, Tripo H3.1, Meshy 6, TRELLIS 1; FLUX para conceptos; Meshy 7 para retexturizar; Real-ESRGAN y Depth Anything.
- **Mi GPU** (`local`, gratis): la app instala y arranca sola Hunyuan3D-2 + FlashVDM, Hunyuan3D-2mv, Zero123++, SDXL-Turbo, Depth Anything V2, Real-ESRGAN y el traductor Opus-MT en tu NVIDIA (8 GB bastan). Los scripts `scripts/windows/instalar.bat` / `iniciar.bat` siguen disponibles para usarla desde el código fuente.
- **Demo** (`mock`): sin IA, formas procedurales para probar todo el flujo.

Las herramientas de geometría, rig, animación, texturas HD (sin IA, con reescalado de calidad) y exportación funcionan en local con cualquier proveedor.

## Desarrollo

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
cd backend && ../.venv/bin/uvicorn app.main:app --reload     # http://localhost:8000
```

- Ventana de escritorio desde el código: `pip install pywebview` y `python desktop/forja3d_desktop.py`.
- Empaquetar: `pip install pyinstaller` y `pyinstaller desktop/forja3d.spec` (el instalador de Windows usa Inno Setup con `desktop/windows/forja3d.iss`; ver `.github/workflows/desktop.yml`).
- Docker: `docker build -t forja3d . && docker run -p 8000:8000 --env-file .env -v forja3d-data:/data forja3d`
- Tests: `cd backend && ../.venv/bin/python -m pytest` y `cd gpu_worker && ../.venv/bin/python -m pytest tests`

## Arquitectura

```
desktop/               lanzador de escritorio (ventana nativa), spec de PyInstaller, instalador Windows
gpu_worker/            servidor de IA local en tu GPU (Hunyuan3D-2, SD-Turbo, Depth Anything, Real-ESRGAN)
scripts/windows/       instalar.bat / iniciar.bat (worker de GPU)
integrations/blender/  complemento de Blender
integrations/mcp/      servidor MCP para Claude Code, Cursor y Claude Desktop
frontend/              HTML + JS sin build; three.js y fuentes incluidos (funciona sin internet)
  index.html, app.js   estudio: crear, galería, herramientas, flujo 01–04
  i18n.js              español / inglés
  viewer.js            visor: modos, inspector de mapas, luces, animación, editor de marcadores
  rig.js, settings.js  panel de rig y animación · ajustes
backend/app/
  main.py              API REST (FastAPI) + sirve el frontend
  pipelines.py         trabajos: generación IA, operaciones locales, subidas
  jobs.py              ejecutor asíncrono con límite de concurrencia y dependencias
  geometry.py          remallado (quads incl.), análisis, reparación, corte, exportación, multicolor
  textures.py          texturas HD y mapas PBR
  projection.py        pintado del modelo desde sus vistas
  sheet.py             hojas de personaje: detección y recorte de vistas
  creations.py         escena 2.5D, relieve, litofanía, texto 3D
  rigging/             skeleton · quadruped · skinning · animations · mocap (CMU) · gltf_writer
  blender.py           conversión FBX/.blend con Blender (pistas NLA)
  providers/           fal.py (nube) · local.py (tu GPU) · mock.py (demo)
```

### API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/config` | proveedor, motores, capacidades, estilos, formatos |
| `GET` / `PUT` | `/api/settings` | motor de IA, claves opcionales, Blender (las claves nunca se devuelven) |
| `GET` / `POST` | `/api/local-ai` · `/api/local-ai/install` | estado / instalación con un clic de la IA gratis en tu GPU |
| `POST` | `/api/agent/chat` | `{message, conversation_id?, lang}` → `{reply, conversation_id, artifacts}` |
| `GET` | `/api/printers` | camas de impresoras y laminadores instalados |
| `POST` | `/api/jobs/{id}/slicer` | `{slicer?, format, size_mm, colors, bed}` abre el modelo en tu laminador |
| `POST` | `/api/concepts` | `{prompt, style, symmetry, pose, count, seed?}` → imágenes concepto |
| `POST` | `/api/sheet/split` | multipart `image` (hoja de personaje) → vistas recortadas con su papel |
| `POST` | `/api/jobs/text` | `{prompt, quality, style, symmetry, pose, seed?, variants, engine?}` |
| `POST` | `/api/jobs/image` | multipart: `images` (1–4: frente, lateral, espalda, derecho) o `concept_id`, `engine?` |
| `POST` | `/api/jobs/upload` · `/scene` · `/text3d` | subir modelo · foto → 3D · texto 3D |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | listado / detalle |
| `PATCH` / `DELETE` | `/api/jobs/{id}` | renombrar, favorito / eliminar |
| `GET` | `/api/jobs/{id}/export?format=glb\|obj\|stl\|3mf\|3mf_color\|stl_colors\|ply\|fbx\|blend&size_mm=&fit=&colors=` | descarga |
| `POST` | `/api/jobs/{id}/enhance` | `{target: 2k\|4k\|8k, pbr}` texturas HD |
| `POST` | `/api/jobs/{id}/retexture` | `{prompt, style, resolution}` |
| `POST` | `/api/jobs/{id}/remesh` | `{mode: decimate\|quads\|lowpoly\|voxel\|smooth, target_faces, resolution}` |
| `POST` | `/api/jobs/{id}/repair` · `/split` · `/refine` · `/variations` | impresión y variantes |
| `GET` | `/api/rig/presets` | animaciones (humano, cuadrúpedo, biblioteca mocap) y marcadores |
| `POST` | `/api/jobs/{id}/rig/markers` | `{template?, yaw?, auto?}` → marcadores |
| `POST` | `/api/jobs/{id}/rig` | `{template, markers?, yaw?, animations, in_place}` |
| `POST` | `/api/jobs/{id}/character` | `{target, pbr, template, animations}` → texturas HD + rig en un clic |
| `POST` | `/api/jobs/{id}/animate` | multipart: `animations`, `bvh`, `keep_bvh`, `in_place` |
| `GET` | `/api/integrations/blender` | complemento de Blender preconfigurado |
| `GET` | `/api/integrations/mcp` · `/api/integrations/mcp/setup` | servidor MCP preconfigurado · comandos para Claude Code, Cursor y Claude Desktop |
