# Forja3D — generador de modelos 3D con IA

Aplicación web propia para generar modelos 3D (GLB) a partir de **texto** o de una **imagen**, al estilo de Meshy o Tripo3D.

- **Texto → 3D** y **Imagen → 3D**, con tres niveles de calidad.
- Cola de trabajos en segundo plano con progreso en vivo y límite de concurrencia.
- Visor 3D integrado (three.js): rotación, vista de malla, encuadre, estadísticas y descarga del `.glb`.
- Galería persistente (SQLite + archivos en disco).
- **Proveedores intercambiables**:
  - `mock`: funciona sin GPU ni claves; crea modelos procedurales (árbol, hongo, roca, dona…) para desarrollar la app.
  - `fal`: IA real mediante [fal.ai](https://fal.ai) con modelos abiertos de última generación — **TRELLIS** (Microsoft) o **Hunyuan3D-2** (Tencent) para imagen→3D, y **FLUX** para el paso texto→imagen.

## Arranque rápido

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
cp .env.example .env          # GEN_PROVIDER=mock por defecto
cd backend
../.venv/bin/uvicorn app.main:app --reload
```

Abre <http://localhost:8000>.

### Con IA real

1. Crea una clave en <https://fal.ai/dashboard/keys>.
2. En `.env`: `GEN_PROVIDER=fal` y `FAL_KEY=tu_clave`.
3. Opcional: `FAL_IMAGE_TO_3D_MODEL=fal-ai/hunyuan3d/v2` para usar Hunyuan3D-2 en lugar de TRELLIS.

Cada generación tarda ~30 s – 2 min y se cobra por uso en fal.ai.

### Docker

```bash
docker build -t forja3d .
docker run -p 8000:8000 --env-file .env -v forja3d-data:/data forja3d
```

### Tests

```bash
cd backend && ../.venv/bin/python -m pytest
```

## Arquitectura

```
frontend/            HTML + JS sin build (three.js desde CDN)
  app.js             formulario, galería, sondeo de estado
  viewer.js          visor GLB
backend/app/
  main.py            API REST (FastAPI) + sirve el frontend
  jobs.py            ejecutor de trabajos asíncrono con semáforo
  db.py              persistencia SQLite
  providers/
    base.py          interfaz Provider (generate(request, on_progress) -> GLB)
    mock.py          generador procedural offline
    fal.py           pipeline real texto→imagen→3D / imagen→3D
```

### API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/jobs/text` | JSON `{prompt, quality}` → crea trabajo |
| `POST` | `/api/jobs/image` | multipart `image`, `quality`, `prompt?` → crea trabajo |
| `GET` | `/api/jobs` | lista de trabajos |
| `GET` | `/api/jobs/{id}` | estado, progreso y URLs |
| `GET` | `/api/jobs/{id}/model.glb` | descarga del modelo |
| `GET` | `/api/jobs/{id}/image` | imagen de entrada / referencia generada |
| `DELETE` | `/api/jobs/{id}` | elimina trabajo y archivos |

`quality` ∈ `draft | standard | high`. `status` ∈ `queued | running | succeeded | failed`.

## Añadir otro proveedor (p. ej. GPU propia)

Implementa una clase con `name` y `async generate(req, on_progress) -> GenerationResult` en `backend/app/providers/` y regístrala en `providers/__init__.py`. Para autoalojar, lo habitual es levantar en una GPU (≥16–24 GB VRAM) un servicio con [TRELLIS](https://github.com/microsoft/TRELLIS) o [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) que exponga un endpoint HTTP, y escribir un proveedor que lo llame igual que `fal.py`.

## Próximos pasos sugeridos

- Cuentas de usuario y créditos / pagos.
- Almacenamiento en S3/R2 y cola distribuida (Redis + workers) para escalar.
- Exportar a FBX/OBJ/USDZ, remallado (retopología) y rigging automático.
- Edición de texturas (texto → textura sobre una malla existente).
