# Forja3D — generador de modelos 3D con IA

Aplicación web propia para crear, editar y preparar modelos 3D con IA, al estilo de Meshy o Tripo3D.

## Funciones

**Crear**
- **Texto → 3D**, con 7 estilos (realista, cartoon, pintado a mano, low-poly, vóxel, escultura), simetría, pose A/T para personajes y semilla reproducible.
- **Conceptos primero**: genera 4 imágenes, elige la que te guste y conviértela en 3D (no pagas el paso 3D por diseños que no quieres).
- **Imagen → 3D** y **Multi-vista → 3D** (2–4 fotos del mismo objeto para una espalda fiel).
- **Variantes**: hasta 4 modelos por petición, con semillas consecutivas.
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
- Exportación STL/3MF con el eje Z hacia arriba, en milímetros, a la altura que elijas y apoyada en la cama.

**Ver y compartir**
- Visor three.js: modos textura / arcilla / normales / malla, iluminación estudio / exterior / noche, reproducción de animaciones.
- Captura PNG y **vídeo 360° en WebM** con un clic.
- Miniaturas automáticas, galería con búsqueda, filtros, favoritos y renombrado.
- **Enlace para compartir con AR** (Android, y iOS con Quick Look) y código `<iframe>` para insertar en cualquier web.
- Exportación: **GLB, OBJ (zip con textura), STL, 3MF, PLY**.

### Comparación con Meshy

| | Forja3D | Meshy |
|---|---|---|
| Texto/imagen/multi-vista → 3D | ✅ | ✅ |
| Elegir entre conceptos antes del 3D | ✅ | ✅ |
| Estilos, simetría, pose, semilla | ✅ | ✅ |
| Remallado / low-poly / vóxel | ✅ local, sin coste | parcial |
| Retexturizado | ✅ en modo demo · ⏳ con fal.ai | ✅ |
| Análisis, reparación y división para impresión | ✅ | ✅ |
| STL/3MF listos para slicer, a escala en mm | ✅ | ✅ |
| Vídeo 360°, AR, iframe | ✅ | parcial |
| Subir y convertir tus propios modelos | ✅ | ✅ |
| Autoalojado, sin suscripción, el código es tuyo | ✅ | ❌ |
| Rigging y animación automáticos | ❌ (el visor sí reproduce animaciones) | ✅ |
| Exportar FBX / USDZ, mapas PBR separados | ❌ | ✅ |
| Cuentas, créditos y comunidad | ❌ | ✅ |

## Proveedores de IA

Se eligen con `GEN_PROVIDER` en `.env`:

- **`mock`** (por defecto): sin GPU ni clave. Genera figuras procedurales y conceptos dibujados, y respeta colores, estilos y semillas. Sirve para desarrollar y hacer demos.
- **`fal`**: IA real a través de [fal.ai](https://fal.ai):
  - imagen → 3D: **TRELLIS** (`fal-ai/trellis`) o **Hunyuan3D-2** (`fal-ai/hunyuan3d/v2`)
  - multi-vista → 3D: `fal-ai/trellis/multi`
  - texto → imagen (y conceptos): **FLUX** (`fal-ai/flux/schnell`)
  - El retexturizado aún no está disponible con este proveedor; la interfaz lo oculta.

Todas las herramientas de geometría (remallado, estilos low-poly/vóxel/escultura, reparación, división, exportación) se ejecutan **en local** con cualquier proveedor.

## Arranque rápido

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

**Tests:** `cd backend && ../.venv/bin/python -m pytest`

## Arquitectura

```
frontend/              HTML + JS sin build (three.js y model-viewer desde CDN)
  index.html, app.js   estudio: crear, galería, herramientas
  viewer.js            visor: modos, luces, animación, captura, vídeo 360°
  share.html           página pública con AR / iframe
backend/app/
  main.py              API REST (FastAPI) + sirve el frontend
  pipelines.py         trabajos: generación IA, operaciones locales, subidas
  jobs.py              ejecutor asíncrono con límite de concurrencia
  geometry.py          remallado, vóxeles, análisis, reparación, corte, exportación
  styles.py            estilos y construcción de prompts
  db.py                SQLite con migraciones automáticas
  providers/           mock.py (offline) · fal.py (fal.ai) · base.py (interfaz)
```

### API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/config` | proveedor, capacidades, estilos, formatos |
| `POST` | `/api/concepts` | `{prompt, style, symmetry, pose, count, seed?}` → imágenes concepto |
| `POST` | `/api/jobs/text` | `{prompt, quality, style, symmetry, pose, seed?, variants}` → lista de trabajos |
| `POST` | `/api/jobs/image` | multipart: `images` (1–4) o `concept_id`, `prompt?`, `quality`, `style`, `seed?`, `variants` |
| `POST` | `/api/jobs/upload` | multipart: `model` (glb/obj/stl/ply/off/3mf) |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | listado / detalle |
| `PATCH` | `/api/jobs/{id}` | `{name?, favorite?}` |
| `DELETE` | `/api/jobs/{id}` | elimina trabajo y archivos |
| `GET` | `/api/jobs/{id}/model.glb` · `/image` · `/thumbnail` | archivos |
| `PUT` | `/api/jobs/{id}/thumbnail` | miniatura (cuerpo PNG/JPG/WEBP) |
| `GET` | `/api/jobs/{id}/analysis` | informe de imprimibilidad |
| `GET` | `/api/jobs/{id}/export?format=glb\|obj\|stl\|3mf\|ply&height_mm=` | descarga convertida |
| `POST` | `/api/jobs/{id}/remesh` | `{mode: decimate\|lowpoly\|voxel\|smooth, target_faces, resolution}` |
| `POST` | `/api/jobs/{id}/retexture` | `{prompt, style, seed?}` |
| `POST` | `/api/jobs/{id}/repair` | reparación para impresión |
| `POST` | `/api/jobs/{id}/split` | `{mode: height\|components, parts}` |
| `POST` | `/api/jobs/{id}/refine` | misma entrada y semilla, calidad alta |
| `POST` | `/api/jobs/{id}/variations` | `{count}` nuevas semillas |

## Añadir otro proveedor (p. ej. tu propia GPU)

Implementa `generate`, `generate_images` y `retexture` (ver `providers/base.py`), declara tus `capabilities` y regístralo en `providers/__init__.py`; la interfaz se adapta sola a lo que soporte. Para autoalojar, levanta en una GPU de 16–24 GB un servicio HTTP con [TRELLIS](https://github.com/microsoft/TRELLIS) o [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2). Hunyuan3D-2 incluye texturizado de mallas, lo que habilitaría el retexturizado real.

## Próximos pasos sugeridos

- Rigging y animación automáticos (p. ej. con modelos abiertos de auto-rigging y una librería de movimientos).
- Exportar FBX / USDZ y mapas PBR separados (albedo, normal, rugosidad, metal).
- Cuentas de usuario, créditos y pagos; galería pública de la comunidad.
- Almacenamiento en S3/R2 y cola distribuida (Redis + workers) para escalar.
