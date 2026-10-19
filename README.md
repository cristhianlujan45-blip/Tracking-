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

**Rigging y animación de personajes**
- **Auto-rig con un clic** para humanoides en pose A o T: esqueleto de 27 huesos con nombres estándar (Mixamo, Unity Humanoid y retargeting de Unreal). Los pies quedan apoyados en el suelo.
- **Marcadores ajustables** al estilo Mixamo (barbilla, hombros, codos, muñecas, ingle, rodillas, tobillos), con vista frontal, simetría y giro de 90° si el personaje no mira al frente.
- **Pesos de piel por difusión de calor** (*bone heat*, el método de Pinocchio): codos, rodillas y hombros se doblan con transiciones suaves y como máximo 4 huesos por vértice.
- **Biblioteca de 14 animaciones** que se adaptan a las proporciones y a la pose de reposo de cada personaje: reposo, caminar, correr, saltar, saludar, bailar, puñetazos, victoria, asentir, negar, mirar alrededor, sentado, T-pose y A-pose. Los pies se pegan al suelo automáticamente.
- **Mocap BVH**: sube capturas de movimiento (CMU, Rokoko, Mixamo convertido en Blender…) y se adaptan a tu personaje, aunque el BVH esté en pose T y el personaje en pose A. Opción "en el sitio" o con desplazamiento.
- Visor con **esqueleto** y **mapa de pesos por hueso** para revisar el rig.
- Exportación **GLB con animaciones** y **FBX / .blend** (convertidos con Blender) para Unity, Unreal, Maya o 3ds Max. Validado con el validador oficial de Khronos y reimportado en Blender 5.0.

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

Datos de Meshy según su web (octubre de 2026). "No figura" significa que la web no lo menciona.

| | Forja3D | Meshy |
|---|---|---|
| Texto/imagen/multi-vista → 3D | ✅ | ✅ |
| Elegir entre conceptos antes del 3D | ✅ | ✅ |
| Auto-rig de humanoides | ✅ | ✅ (y cuadrúpedos) |
| Corregir el rig con marcadores (estilo Mixamo) | ✅ | no figura |
| Ver huesos y mapa de pesos antes de exportar | ✅ | no figura |
| Biblioteca de animaciones | 14, adaptadas a cada personaje | 600+ |
| Importar tu propio mocap BVH | ✅ | no figura |
| Exportar GLB/FBX con animaciones | ✅ | ✅ |
| Foto/escena → 3D, relieve y litofanía | ✅ | no figura |
| Texto 3D, placas y llaveros | ✅ | ✅ |
| Remallado / low-poly / vóxel | ✅ local | ✅ |
| Retexturizado | ✅ demo y GPU local (`--texture`) | ✅ |
| Análisis, reparación y división para impresión | ✅ | ✅ |
| STL/3MF a escala en mm | ✅ | ✅ |
| Complemento de Blender | ✅ | ✅ |
| IA en tu propia GPU, sin coste por modelo | ✅ (8 GB bastan) | ❌ |
| Autoalojado, sin suscripción, el código es tuyo | ✅ | ❌ |
| Rig de cuadrúpedos y dedos | ❌ | ✅ cuadrúpedos |
| Texturas PBR separadas, quads | ❌ | ✅ |
| Impresión multicolor, cuentas, comunidad | ❌ | ✅ |

### Qué falta (en orden de prioridad)

1. **Rig de cuadrúpedos y de dedos.** La misma técnica (marcadores, pesos por calor y animaciones) se amplía con otra plantilla de esqueleto.
2. **Más animaciones**: una biblioteca de mocap con licencia libre (p. ej. CMU) precargada; el importador BVH ya está hecho.
3. **Remallado en quads** con Instant Meshes, para una topología limpia al animar o esculpir.
4. **Impresión multicolor**: 3MF por colores (Bambu AMS, Prusa MMU) y abrir directamente en Bambu Studio u OrcaSlicer.
5. **Texturas PBR** (normal, rugosidad, metal). Hunyuan3D-2.1 las genera, pero pide más de 8 GB de VRAM.
6. **Producto**: cuentas, créditos, galería pública, API con claves y app móvil.

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
  rigging/             skeleton.py (marcadores, esqueleto) · skinning.py (bone heat)
                       animations.py (biblioteca + BVH) · gltf_writer.py (GLB con skin)
  blender.py           conversión FBX/.blend con Blender en segundo plano
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
| `GET` | `/api/rig/presets` | animaciones disponibles y marcadores |
| `POST` | `/api/jobs/{id}/rig/markers` | `{yaw?, auto?}` → marcadores estimados (o los guardados) |
| `POST` | `/api/jobs/{id}/rig` | `{markers?, yaw?, symmetric, animations, in_place}` → personaje con rig |
| `POST` | `/api/jobs/{id}/animate` | multipart: `animations`, `bvh` (archivos), `keep_bvh`, `in_place` |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | listado / detalle |
| `PATCH` | `/api/jobs/{id}` | `{name?, favorite?}` |
| `DELETE` | `/api/jobs/{id}` | elimina trabajo y archivos |
| `GET` | `/api/jobs/{id}/model.glb` · `/image` · `/thumbnail` | archivos |
| `PUT` | `/api/jobs/{id}/thumbnail` | miniatura (cuerpo PNG/JPG/WEBP) |
| `GET` | `/api/jobs/{id}/analysis` | informe de imprimibilidad |
| `GET` | `/api/jobs/{id}/export?format=glb\|obj\|stl\|3mf\|ply\|fbx\|blend&size_mm=&fit=height\|width\|depth\|max` | descarga convertida (FBX/.blend necesitan Blender) |
| `POST` | `/api/jobs/{id}/remesh` | `{mode: decimate\|lowpoly\|voxel\|smooth, target_faces, resolution}` |
| `POST` | `/api/jobs/{id}/retexture` | `{prompt, style, seed?}` |
| `POST` | `/api/jobs/{id}/repair` | reparación para impresión |
| `POST` | `/api/jobs/{id}/split` | `{mode: height\|components, parts}` |
| `POST` | `/api/jobs/{id}/refine` | misma entrada y semilla, calidad alta |
| `POST` | `/api/jobs/{id}/variations` | `{count}` nuevas semillas |

## Añadir otro proveedor (p. ej. tu propia GPU)

Implementa `generate`, `generate_images`, `retexture` y `estimate_depth` (ver `providers/base.py`), declara tus `capabilities` y regístralo en `providers/__init__.py`; la interfaz se adapta sola a lo que soporte. Para autoalojar, levanta en una GPU de 16–24 GB un servicio HTTP con [TRELLIS](https://github.com/microsoft/TRELLIS) o [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2). Hunyuan3D-2 incluye texturizado de mallas, lo que habilitaría el retexturizado real.
