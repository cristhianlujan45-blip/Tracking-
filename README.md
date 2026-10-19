# Forja3D — generador de modelos 3D con IA

Aplicación web propia para crear, editar y preparar modelos 3D con IA, al estilo de Meshy o Tripo3D.

## Funciones

**Crear**
- **Texto → 3D**, con 7 estilos (realista, cartoon, pintado a mano, low-poly, vóxel, escultura), simetría, pose A/T para personajes y semilla reproducible.
- **Conceptos primero**: genera 4 imágenes, elige la que te guste y conviértela en 3D (no pagas el paso 3D por diseños que no quieres).
- **Imagen → 3D** y **Multi-vista → 3D** (2–4 fotos del mismo objeto para una espalda fiel).
- **Variantes**: hasta 4 modelos por petición, con semillas consecutivas.
- **Escena / foto → 3D**: cualquier foto (paisaje, habitación, retrato) a partir de su profundidad estimada con IA (Depth Anything V2 en tu GPU):
  - **Escena 2.5D**: malla con la foto como textura, para fondos y entornos en Blender, Unity o Unreal.
  - **Relieve**: placa sólida con la escena en bajorrelieve, lista para imprimir.
  - **Litofanía**: placa fina que muestra la foto al iluminarla por detrás.
- **Texto 3D**: letras extruidas, placas con nombre o llaveros con agujero; elige fuente, grosor y colores.
- **Subir tu modelo** (GLB, OBJ, STL, PLY, OFF, 3MF) para verlo, convertirlo o editarlo.

**Editar** (cada edición crea un modelo nuevo y conserva el original)
- **Refinar HD**: la misma entrada y semilla en calidad alta.
- **Variaciones**: 3 alternativas nuevas de un modelo.
- **Remallado**: reducir polígonos (de 200 a 200k), low-poly facetado, vóxeles (8³–96³) o suavizado.
- **Texturizado IA**: repintar un modelo existente desde texto.

**Impresión 3D**
- Análisis: estanqueidad, normales, triángulos degenerados, piezas sueltas, volumen.
- **Reparación automática**: tapa agujeros de cualquier tamaño y corrige normales.
- **División en piezas**: cortes por altura con tapas, o separación por componentes.
- Exportación STL/3MF con el eje Z hacia arriba, en milímetros y apoyada en la cama. Eliges qué medida fijar: alto, ancho, fondo o lado mayor.

**Ver y compartir**
- Visor three.js: modos textura / arcilla / normales / malla, iluminación estudio / exterior / noche, reproducción de animaciones.
- Captura PNG y **vídeo 360° en WebM** con un clic.
- Miniaturas automáticas, galería con búsqueda, filtros, favoritos y renombrado.
- **Enlace para compartir con AR** (Android, y iOS con Quick Look) y código `<iframe>` para insertar en cualquier web.
- Exportación: **GLB, OBJ (zip con textura), STL, 3MF, PLY y USDZ** (iPhone/AR).
- **Historial de versiones**: cada modelo muestra de dónde viene y qué se hizo a partir de él.

**Blender**
- **Complemento de Blender** (descárgalo desde el panel Exportar de la app; ya viene configurado con la dirección de tu servidor). En Blender: *Edit › Preferences › Add-ons › Install from Disk*, y luego la barra lateral (tecla **N**) › pestaña **Forja3D**:
  - lista y busca tus modelos, e **impórtalos con un clic** en el cursor 3D, opcionalmente escalados a una altura;
  - **envía la selección de Blender a Forja3D** para remallar, reparar, dividir o exportar para impresión.
- Probado con Blender 5.0. Sin el complemento: *Archivo › Importar › glTF 2.0* con el GLB descargado.

### Comparación con Meshy

| | Forja3D | Meshy |
|---|---|---|
| Texto/imagen/multi-vista → 3D | ✅ | ✅ |
| Elegir entre conceptos antes del 3D | ✅ | ✅ |
| Estilos, simetría, pose, semilla | ✅ | ✅ |
| Foto/escena → 3D, relieve y litofanía | ✅ | ❌ |
| Texto 3D, placas y llaveros | ✅ | ✅ (generador de texto 3D) |
| Remallado / low-poly / vóxel | ✅ local, sin coste | ✅ |
| Retexturizado | ✅ demo y GPU local (`--texture`) · ❌ fal.ai | ✅ |
| Análisis, reparación y división para impresión | ✅ | ✅ |
| STL/3MF listos para slicer, a escala en mm | ✅ | ✅ |
| Complemento de Blender (importar y enviar) | ✅ | ✅ |
| Vídeo 360°, AR, iframe, USDZ | ✅ | parcial |
| IA en tu propia GPU, sin coste por modelo | ✅ (8 GB bastan) | ❌ |
| Autoalojado, sin suscripción, el código es tuyo | ✅ | ❌ |
| Rigging y animación automáticos | ❌ | ✅ |
| Topología en quads (para animar) | ❌ | ✅ |
| Texturas PBR separadas (normal, rugosidad, metal) | ❌ | ✅ |
| FBX, impresión multicolor (AMS), abrir en el slicer | ❌ | ✅ |
| Cuentas, créditos, comunidad, app móvil | ❌ | ✅ |

### Qué falta para igualar a Meshy (en orden de prioridad)

1. **Rigging y animación automáticos** (humanoides y cuadrúpedos). Es lo que más pesa para juegos y vídeo. Se puede hacer con modelos abiertos de auto-rigging (p. ej. UniRig) en tu GPU y una biblioteca de movimientos en formato BVH/FBX.
2. **Remallado en quads** con [Instant Meshes](https://github.com/wjakob/instant-meshes) (CPU, código abierto): topología limpia para animar y esculpir en Blender.
3. **Exportar FBX** usando Blender sin interfaz (`blender -b`) como conversor. Ya está comprobado que funciona como módulo de Python.
4. **Impresión multicolor**: reducir los colores del modelo a 4–16 filamentos y exportar 3MF por colores (Bambu AMS, Prusa MMU). Añadir botones para abrir directamente en Bambu Studio u OrcaSlicer.
5. **Texturas PBR** (normal, rugosidad, metal). Hunyuan3D-2.1 las genera, pero pide más de 8 GB de VRAM; en tu PC habría que generarlas en la nube.
6. **Editor de imagen previo**: recorte, quitar fondo con vista previa, varias vistas generadas por IA a partir de una sola foto.
7. **Producto**: cuentas, créditos, galería pública, API con claves y app móvil (PWA).

## Proveedores de IA

Se eligen con `GEN_PROVIDER` en `.env`:

- **`local`**: IA real **en tu propia GPU NVIDIA** (desde 8 GB de VRAM, RTX serie 50 incluidas), gratis y privada. Usa Hunyuan3D-2mini para la forma y SD-Turbo para los conceptos; varias vistas y texturas son opcionales. Ver [gpu_worker/README.md](gpu_worker/README.md).
- **`mock`** (por defecto): sin GPU ni clave. Genera figuras procedurales y conceptos dibujados, y respeta colores, estilos y semillas. Sirve para desarrollar y hacer demos.
- **`fal`**: IA real a través de [fal.ai](https://fal.ai):
  - imagen → 3D: **TRELLIS** (`fal-ai/trellis`) o **Hunyuan3D-2** (`fal-ai/hunyuan3d/v2`)
  - multi-vista → 3D: `fal-ai/trellis/multi`
  - texto → imagen (y conceptos): **FLUX** (`fal-ai/flux/schnell`)
  - El retexturizado aún no está disponible con este proveedor; la interfaz lo oculta.

Todas las herramientas de geometría (remallado, estilos low-poly/vóxel/escultura, reparación, división, exportación) se ejecutan **en local** con cualquier proveedor.

## Arranque rápido

**Windows con GPU NVIDIA** (IA local):

```bat
scripts\windows\instalar.bat
scripts\windows\iniciar.bat
```

**Modo demo** (cualquier sistema, sin GPU):

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
cp .env.example .env          # GEN_PROVIDER=mock por defecto
cd backend
../.venv/bin/uvicorn app.main:app --reload
```

Abre <http://localhost:8000>.

**Con IA real:** crea una clave en <https://fal.ai/dashboard/keys> y pon `GEN_PROVIDER=fal` y `FAL_KEY=…` en `.env`. Cada modelo tarda de 30 s a 2 min y fal.ai lo cobra por uso.

**Docker:**
```bash
docker build -t forja3d .
docker run -p 8000:8000 --env-file .env -v forja3d-data:/data forja3d
```

**Tests:** `cd backend && ../.venv/bin/python -m pytest` y `cd gpu_worker && ../.venv/bin/python -m pytest tests`

## Arquitectura

```
gpu_worker/            servidor de IA local en tu GPU (Hunyuan3D-2mini, SD-Turbo)
scripts/windows/       instalar.bat / iniciar.bat
integrations/blender/  complemento de Blender
frontend/              HTML + JS sin build (three.js y model-viewer desde CDN)
  index.html, app.js   estudio: crear, galería, herramientas
  viewer.js            visor: modos, luces, animación, captura, vídeo 360°
  share.html           página pública con AR / iframe
backend/app/
  main.py              API REST (FastAPI) + sirve el frontend
  pipelines.py         trabajos: generación IA, operaciones locales, subidas
  jobs.py              ejecutor asíncrono con límite de concurrencia
  geometry.py          remallado, vóxeles, análisis, reparación, corte, exportación
  creations.py         escena 2.5D, relieve, litofanía, texto 3D
  styles.py            estilos y construcción de prompts
  db.py                SQLite con migraciones automáticas
  providers/           local.py (tu GPU) · fal.py (fal.ai) · mock.py (demo) · base.py
```

### API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/config` | proveedor, capacidades, estilos, formatos |
| `POST` | `/api/concepts` | `{prompt, style, symmetry, pose, count, seed?}` → imágenes concepto |
| `POST` | `/api/jobs/text` | `{prompt, quality, style, symmetry, pose, seed?, variants}` → lista de trabajos |
| `POST` | `/api/jobs/image` | multipart: `images` (1–4) o `concept_id`, `prompt?`, `quality`, `style`, `seed?`, `variants` |
| `POST` | `/api/jobs/upload` | multipart: `model` (glb/obj/stl/ply/off/3mf) |
| `POST` | `/api/jobs/scene` | multipart: `image`, `kind: scene\|relief\|lithophane`, `resolution`, `strength`, `cut_edges` |
| `POST` | `/api/jobs/text3d` | `{text, style: letters\|plate\|keychain, font, bold, depth, color, plate_color}` |
| `GET` | `/api/jobs/{id}/depth` | mapa de profundidad usado (escenas y relieves) |
| `GET` | `/api/integrations/blender` | complemento de Blender preconfigurado |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | listado / detalle |
| `PATCH` | `/api/jobs/{id}` | `{name?, favorite?}` |
| `DELETE` | `/api/jobs/{id}` | elimina trabajo y archivos |
| `GET` | `/api/jobs/{id}/model.glb` · `/image` · `/thumbnail` | archivos |
| `PUT` | `/api/jobs/{id}/thumbnail` | miniatura (cuerpo PNG/JPG/WEBP) |
| `GET` | `/api/jobs/{id}/analysis` | informe de imprimibilidad |
| `GET` | `/api/jobs/{id}/export?format=glb\|obj\|stl\|3mf\|ply&size_mm=&fit=height\|width\|depth\|max` | descarga convertida |
| `POST` | `/api/jobs/{id}/remesh` | `{mode: decimate\|lowpoly\|voxel\|smooth, target_faces, resolution}` |
| `POST` | `/api/jobs/{id}/retexture` | `{prompt, style, seed?}` |
| `POST` | `/api/jobs/{id}/repair` | reparación para impresión |
| `POST` | `/api/jobs/{id}/split` | `{mode: height\|components, parts}` |
| `POST` | `/api/jobs/{id}/refine` | misma entrada y semilla, calidad alta |
| `POST` | `/api/jobs/{id}/variations` | `{count}` nuevas semillas |

## Añadir otro proveedor (p. ej. tu propia GPU)

Implementa `generate`, `generate_images`, `retexture` y `estimate_depth` (ver `providers/base.py`), declara tus `capabilities` y regístralo en `providers/__init__.py`; la interfaz se adapta sola a lo que soporte. Para autoalojar, levanta en una GPU de 16–24 GB un servicio HTTP con [TRELLIS](https://github.com/microsoft/TRELLIS) o [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2). Hunyuan3D-2 incluye texturizado de mallas, lo que habilitaría el retexturizado real.
