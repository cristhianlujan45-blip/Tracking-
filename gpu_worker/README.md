# Worker de IA local (tu GPU)

Servidor que ejecuta los modelos de IA en tu propia tarjeta gráfica. La app Forja3D lo usa con `GEN_PROVIDER=local`: no hay coste por uso ni nada sale de tu PC.

| Función | Modelo | VRAM aprox. | Por defecto |
|---|---|---|---|
| Imagen → forma 3D | Hunyuan3D-2mini (Tencent, 0,6 B) | ~5–6 GB | ✅ |
| Conceptos / texto → imagen | SD-Turbo (o SDXL-Turbo con `--image-model`) | ~3 GB (~7 GB SDXL) | ✅ |
| Varias vistas → 3D | Hunyuan3D-2mv | ~7 GB | `--multiview` |
| Texturas y retexturizado | Hunyuan3D-Paint | ~16 GB (con descarga a RAM en 8 GB, lento) | `--texture` |

Sólo hay **un modelo en la GPU a la vez**: el resto espera en la RAM del sistema. Así todo cabe en **8 GB** (p. ej. una RTX 5060 de portátil). Con 16 GB de RAM o más va bien.

Sin `--texture`, los modelos salen sin textura (grises). Todo lo demás funciona igual: estilos, remallado, vóxeles, reparación, impresión y exportación.

## Instalación en Windows

Antes de empezar instala **Python 3.11** (marca "Add to PATH"), **Git** y el **driver de NVIDIA** más reciente. Después, desde la carpeta del proyecto:

```bat
scripts\windows\instalar.bat
scripts\windows\iniciar.bat
```

El instalador:
- instala PyTorch con **CUDA 12.8**, el mínimo para las RTX serie 50 (Blackwell, `sm_120`);
- descarga Hunyuan3D-2;
- instala la app y el worker;
- comprueba que PyTorch ve tu GPU;
- pone `GEN_PROVIDER=local` en `.env`.

`iniciar.bat` arranca el worker y la app y abre el navegador. Opciones:

```bat
scripts\windows\iniciar.bat -Demo        :: sin IA, para comprobar que todo funciona
scripts\windows\iniciar.bat -Multiview   :: activa varias vistas
scripts\windows\iniciar.bat -ImageModel stabilityai/sdxl-turbo   :: conceptos de más calidad
```

La **primera generación descarga los modelos** desde Hugging Face (varios GB) y tarda bastante. Las siguientes son mucho más rápidas.

### Texturas (experimental en 8 GB)

```bat
scripts\windows\instalar.bat -Texture
scripts\windows\iniciar.bat -Texture
```

Necesita además **Visual Studio 2022 Build Tools** (carga "Desarrollo de escritorio con C++") y **CUDA Toolkit 12.8**, para compilar las extensiones de Hunyuan3D-Paint. Con 8 GB de VRAM el texturizado usa la RAM del sistema y puede tardar varios minutos por modelo. En calidad *Borrador* no se texturiza.

## Linux / instalación manual

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2 gpu_worker/vendor/Hunyuan3D-2
pip install -r gpu_worker/vendor/Hunyuan3D-2/requirements.txt -e gpu_worker/vendor/Hunyuan3D-2
pip install -r backend/requirements.txt -r gpu_worker/requirements.txt

cd gpu_worker && python -m worker --backend hunyuan          # terminal 1
cd backend && GEN_PROVIDER=local uvicorn app.main:app        # terminal 2
```

## Usarlo desde otro equipo

Por defecto el worker sólo escucha en `127.0.0.1`. Para que la app corra en otra máquina (o en un servidor) y use la GPU de tu PC:

```bash
set FORJA_WORKER_TOKEN=una-clave-larga
python -m worker --host 0.0.0.0
```

En la app, configura `LOCAL_WORKER_URL=http://IP-de-tu-pc:8001` y `LOCAL_WORKER_TOKEN=una-clave-larga`.

## API del worker

| Método | Ruta | |
|---|---|---|
| `GET` | `/health` | GPU, VRAM, modelos y capacidades |
| `POST` | `/images` | `{prompt, count, seed, negative}` → `{images: [base64 PNG]}` |
| `POST` | `/tasks` | `{kind: mesh\|texture, images: [base64], seed, quality, mesh_glb?}` → `{id}` |
| `GET` | `/tasks/{id}` | `{status, progress, stage, error}` |
| `GET` | `/tasks/{id}/result` | GLB |
| `DELETE` | `/tasks/{id}` | libera el resultado |

`python -m worker --backend dummy` arranca un worker falso, sin GPU, para pruebas. Los tests se ejecutan con `python -m pytest gpu_worker/tests`.

## Problemas frecuentes

- **"Sin memoria de GPU (VRAM)"**: usa calidad *Borrador* o *Estándar*. Cierra juegos, el navegador con aceleración o cualquier otro programa que use la GPU. No uses `--texture` y `--multiview` a la vez.
- **`no kernel image is available` o `sm_120 is not compatible`**: PyTorch no es la versión cu128. Repite `instalar.bat`, que reinstala la versión correcta.
- **El portátil usa la gráfica integrada**: en Panel de control de NVIDIA → Configuración 3D, asigna "Procesador NVIDIA de alto rendimiento" a `python.exe`. Conecta el cargador: en batería la GPU rinde mucho menos.

## Licencias de los modelos

Los modelos no forman parte de este proyecto y tienen licencias propias. Revísalas antes de cualquier uso comercial:

- **Hunyuan3D-2 / 2mini / 2mv / Paint**: *Tencent Hunyuan Community License*. **No concede derechos en la Unión Europea, el Reino Unido ni Corea del Sur.** Además pide una licencia aparte a partir de cierto número de usuarios.
- **SD-Turbo / SDXL-Turbo**: licencia de Stability AI, con condiciones para el uso comercial.
