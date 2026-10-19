// Spanish / English. The interface is written in Spanish; every visible text (and every text the
// server sends: labels, progress, errors) is looked up here when the language is English.

const KEY = "forja.lang";

export const lang = (() => {
  try {
    return localStorage.getItem(KEY) || ((navigator.language || "es").startsWith("es") ? "es" : "en");
  } catch {
    return "es";
  }
})();

export function setLang(next) {
  if (next === lang) return;
  try {
    localStorage.setItem(KEY, next);
  } catch {}
  location.reload(); // the whole UI (and the server's labels) comes back in the new language
}

const EN = {
  // ---------------------------------------------------------------- shell
  "Forja3D — Estudio 3D con IA": "Forja3D — AI 3D Studio",
  "Texto → 3D": "Text → 3D", "Imagen → 3D": "Image → 3D", "Rig + animación": "Rig + animation",
  "Texturas 8K": "8K textures", "Impresión 3D": "3D printing",
  "Motor de IA — clic para cambiar": "AI engine — click to change", "conectando…": "connecting…",
  "Ajustes": "Settings", "Idioma / Language": "Language",
  "Modo demo (sin IA) · configurar": "Demo mode (no AI) · set up", "sin conexión": "offline",
  "IA local · servidor sin conexión": "Local AI · server offline", "IA local": "Local AI", "IA nube": "Cloud AI",
  "No se pudo iniciar el motor elegido": "The chosen engine could not start",

  // ---------------------------------------------------------------- create
  "Crear": "Create", "Nuevo modelo": "New model",
  "Texto": "Text", "Imagen": "Image", "Multi-vista": "Multi-view", "Escena": "Scene", "Texto 3D": "3D text", "Subir": "Upload",
  "Describe tu objeto o personaje": "Describe your object or character",
  "Ej.: una guerrera samurái con armadura roja y katana, estilo anime": "E.g. a samurai warrior with red armour and a katana, anime style",
  "Personaje": "Character", "Criatura": "Creature", "Arma": "Weapon", "Edificio": "Building", "Accesorio": "Prop",
  "Imagen de referencia": "Reference image", "Arrastra una imagen o haz clic": "Drag an image or click",
  "PNG, JPG o WEBP · silueta clara y fondo limpio": "PNG, JPG or WEBP · clear silhouette, clean background",
  "Descripción (opcional)": "Description (optional)", "Ayuda a nombrar y guiar el modelo": "Helps name and guide the model",
  "De 2 a 4 vistas del mismo objeto": "2 to 4 views of the same object",
  "Frente, lateral, espalda… Más vistas = geometría más fiel por detrás.": "Front, side, back… More views = more faithful geometry at the back.",
  "Frente": "Front", "Lateral": "Side", "Espalda": "Back", "Lateral derecho": "Right side",
  "▦ Hoja de personaje: separar vistas automáticamente": "▦ Character sheet: split the views automatically",
  "Sube una hoja con el personaje de frente, de lado y de espaldas: se recortan solas.": "Upload a sheet with the character from the front, side and back: they are cut out for you.",
  "Buscando las vistas…": "Finding the views…", "Vistas detectadas: {v}": "Views found: {v}", " (+{n} sin usar, p. ej. 3/4)": " (+{n} unused, e.g. 3/4)",
  "No se encontraron figuras: usa una hoja con fondo liso": "No figures found: use a sheet with a plain background",
  "No se encontraron figuras completas en la imagen": "No full figures found in the image",
  "Tipo": "Type", "Escena 2.5D": "2.5D scene", "Relieve": "Relief", "Litofanía": "Lithophane",
  "Arrastra una foto (paisaje, habitación, retrato…)": "Drag a photo (landscape, room, portrait…)", "PNG, JPG o WEBP": "PNG, JPG or WEBP",
  "Detalle": "Detail", "Profundidad": "Depth", "Grosor": "Thickness",
  "Separar objetos del fondo (cortar bordes)": "Separate objects from the background (cut edges)",
  "Malla 3D con la foto como textura. Ideal como fondo o entorno en Blender, Unity o Unreal.": "3D mesh with the photo as its texture. Great as a backdrop or environment in Blender, Unity or Unreal.",
  "Placa sólida con la escena en relieve (bajorrelieve). Lista para imprimir en 3D.": "Solid plaque with the scene in relief (bas-relief). Ready to 3D print.",
  "Placa fina: al iluminarla por detrás aparece la foto. Imprime en blanco, de pie, al 100 % de relleno.": "Thin plaque: light it from behind and the photo appears. Print in white, upright, at 100 % infill.",
  "Texto (hasta 4 líneas)": "Text (up to 4 lines)", "Tu nombre, un logo, un cartel…": "Your name, a logo, a sign…",
  "Formato": "Format", "Letras": "Letters", "Placa": "Plaque", "Llavero": "Keychain", "Fuente": "Font",
  "Monoespaciada": "Monospace", "Negrita": "Bold", "Grosor de letra": "Letter depth", "Color letras": "Letter colour", "Color base": "Base colour",
  "Tu modelo 3D": "Your 3D model", "Arrastra un archivo 3D o haz clic": "Drag a 3D file or click", "GLB, OBJ, STL, PLY, OFF o 3MF": "GLB, OBJ, STL, PLY, OFF or 3MF",
  "Úsalo para ver, convertir, remallar, retexturizar, riggear o preparar para impresión.": "Use it to view, convert, remesh, retexture, rig or prepare for printing.",
  "Motor de IA": "AI engine", "Estilo": "Style", "Calidad": "Quality", "Variantes": "Variants",
  "Borrador · rápido": "Draft · fast", "Estándar · PBR": "Standard · PBR", "Full HD · 4K": "Full HD · 4K",
  "Opciones avanzadas": "Advanced options", "Forzar simetría": "Force symmetry", "Pose (personajes)": "Pose (characters)",
  "Semilla": "Seed", "Aleatoria": "Random", "✦ Conceptos": "✦ Concepts",
  "Generar modelo 3D": "Generate 3D model", "Importar modelo": "Import model", "Crear modelo 3D": "Create 3D model",
  "Elige un concepto": "Pick a concept", "↻ Otros": "↻ More", "Convertir concepto en 3D": "Turn concept into 3D", "Concepto": "Concept",
  "Escribe una descripción de al menos 3 caracteres": "Write a description of at least 3 characters",
  "Selecciona una imagen primero": "Select an image first", "Añade al menos 2 vistas": "Add at least 2 views",
  "Selecciona una foto": "Select a photo", "Escribe algún texto": "Type some text", "Selecciona un archivo 3D": "Select a 3D file",
  "Formato de archivo no soportado": "Unsupported file format", "No disponible con el proveedor actual": "Not available with the current provider",
  "Trabajo en cola": "Job queued", "{n} trabajos en cola": "{n} jobs queued",
  // styles & poses (server)
  "Automático": "Automatic", "Realista": "Realistic", "Cartoon": "Cartoon", "Pintado a mano": "Hand-painted",
  "Low-poly": "Low-poly", "Vóxel": "Voxel", "Escultura": "Sculpture", "Libre": "Free", "Estilo: {s}": "Style: {s}",

  // ---------------------------------------------------------------- viewer
  "Cuatro pasos. Un personaje.": "Four steps. One character.",
  "De una imagen a tu próxima": "From a single image to your next", "escena en Blender.": "Blender scene.",
  "Crea un modelo texturizado, con rig y animado desde una imagen o un texto, y llévalo a Blender, Unity, Unreal o Godot.":
    "Create a textured, rigged and animated model from an image or a prompt, then take it to Blender, Unity, Unreal or Godot.",
  "Empieza con una imagen o un texto": "Start with an image or a prompt", "Genera el modelo texturizado": "Generate the textured model",
  "Añade rig y animaciones": "Add a rig and animations", "Continúa en Blender o tu motor": "Continue in Blender or your engine",
  "FORJANDO": "FORGING", "Modo de vista": "View mode", "Textura": "Texture", "Malla + textura": "Wire + texture",
  "Arcilla": "Clay", "Normales": "Normals", "Malla": "Wireframe",
  "Inspeccionar los mapas del material": "Inspect the material maps", "Material final": "Final material",
  "Mapa: color": "Map: colour", "Mapa: normal": "Map: normal", "Mapa: rugosidad": "Map: roughness", "Mapa: metal": "Map: metal",
  "Mapa: oclusión (AO)": "Map: occlusion (AO)",
  "Iluminación y fondo": "Lighting and background", "Estudio": "Studio", "Neón Tokio": "Neon Tokyo", "Atardecer": "Sunset",
  "Sakura": "Sakura", "Exterior": "Outdoor", "Noche": "Night",
  "Arrastra los puntos a las articulaciones (vista frontal; rueda = zoom)": "Drag the dots onto the joints (front view; wheel = zoom)",
  "Vista lateral: cabeza a la derecha. Arrastra los puntos al hocico, lomo, cola y patas del lado visible.":
    "Side view: head on the right. Drag the dots to the snout, back, tail and the legs on the visible side.",
  "Simetría": "Symmetry", "Girar 90° si el personaje no mira al frente": "Turn 90° if the character is not facing front",
  "Auto": "Auto", "Cancelar": "Cancel", "✔ Crear rig": "✔ Create rig",
  "Entrada": "Input", "Modelo texturizado": "Textured model", "Rig y animación": "Rig & animation", "Blender y motores": "Blender & engines",
  "Rotación automática": "Auto-rotate", "Rejilla": "Grid", "Encuadrar": "Frame", "Captura PNG": "PNG snapshot", "Vídeo 360° (WebM)": "360° video (WebM)",
  "● Grabando…": "● Recording…", "Carga un modelo primero": "Load a model first",
  "Genera o selecciona un modelo para verlo aquí": "Generate or select a model to see it here",
  "La generación falló: {e}": "Generation failed: {e}", "Cargando modelo…": "Loading model…",
  "No se pudo cargar el modelo: {e}": "Could not load the model: {e}",
  "{t} triángulos · {v} vértices · {m} malla(s)": "{t} triangles · {v} vertices · {m} mesh(es)", "Textura {r}": "{r} texture",
  "Grabación no soportada en este navegador": "Recording is not supported in this browser", "No hay modelo cargado": "No model loaded",
  "Animación {n}": "Animation {n}",
  // workflow bar
  "pendiente": "pending", "generando…": "generating…", "falló": "failed", "colores por vértice": "vertex colours",
  "texturas": "textures", "texturas {r}": "{r} textures", "PBR {r}": "PBR {r}", "siguiente paso →": "next step →",
  "{n} animaciones": "{n} animations", "GLB · FBX · .blend": "GLB · FBX · .blend",

  // ---------------------------------------------------------------- gallery
  "Mis modelos": "My models", "Todos": "All", "Favoritos": "Favourites", "IA": "AI", "Editados": "Edited", "Rig": "Rig",
  "Personajes con rig": "Rigged characters", "En curso": "Running", "Buscar…": "Search…",
  "Aún no hay modelos.": "No models yet.", "Nada coincide con el filtro.": "Nothing matches the filter.",
  "En cola": "Queued", "Generando": "Generating", "Error": "Error", "Favorito": "Favourite",
  "Subido": "Uploaded", "Remallado": "Remeshed", "Retexturizado": "Retextured", "Reparado": "Repaired", "Piezas": "Parts",
  "Animado": "Animated", "Texturas HD": "HD textures",

  // ---------------------------------------------------------------- tools
  "Selecciona un modelo para editarlo, riggearlo, exportarlo o imprimirlo.": "Select a model to edit, rig, export or print it.",
  "Renombrar": "Rename", "Generación": "Generation", "Misma entrada y semilla en calidad Full HD": "Same input and seed in Full HD quality",
  "⇪ Refinar HD": "⇪ Refine HD", "Nuevas semillas": "New seeds", "⧉ 3 variaciones": "⧉ 3 variations",
  "↺ Cargar parámetros en el formulario": "↺ Load settings into the form", "Parámetros cargados en el formulario": "Settings loaded into the form",
  "Rigging y animación": "Rigging & animation", "Personaje completo en un clic": "Full character in one click",
  "Texturas HD + PBR → esqueleto → animaciones elegidas. Listo para Blender, Unity, Unreal y Godot.":
    "HD + PBR textures → skeleton → chosen animations. Ready for Blender, Unity, Unreal and Godot.",
  "Mejora de texturas antes del rig": "Texture upgrade before rigging", "Texturas 4K": "4K textures", "Texturas 2K": "2K textures",
  "Sin mejorar texturas": "Keep textures as they are", "⚡ Crear": "⚡ Create",
  "Tipo de cuerpo": "Body type", "Humano": "Human", "4 patas": "4 legs", "Animaciones": "Animations",
  "Biblioteca mocap": "Mocap library", "Buscar movimiento…": "Search moves…", "Todas": "All",
  "Tu mocap BVH (opcional)": "Your BVH mocap (optional)", "En el sitio (sin desplazamiento)": "In place (no root motion)",
  "Esqueleto y pesos automáticos": "Automatic skeleton and weights", "⚡ Solo rig": "⚡ Rig only",
  "Coloca tú las articulaciones, como en Mixamo": "Place the joints yourself, like Mixamo", "◎ Marcadores": "◎ Markers",
  "◎ Reajustar": "◎ Adjust", "⚡ Re-riggear": "⚡ Re-rig", "↻ Actualizar animaciones": "↻ Update animations",
  "Ver huesos": "Show bones", "Ver la influencia de un hueso sobre la malla": "See how much a bone moves the mesh",
  "Pesos: ninguno": "Weights: none", "Pesos: {b}": "Weights: {b}", "{n} elegidas · ": "{n} picked · ", "{n} clips": "{n} clips", "bucle": "loop",
  "humano": "Human", "cuadrúpedo": "Quadruped",
  "Esqueleto {t} de {n} huesos · {c} animaciones. ": "{t} skeleton with {n} bones · {c} animations. ",
  "Se creará un rig nuevo con el otro tipo de cuerpo.": "A new rig will be created with the other body type.",
  "Añade movimientos, sube mocap BVH o reajusta las articulaciones.": "Add moves, upload BVH mocap or adjust the joints.",
  "Animales de cuatro patas (perro, gato, caballo, dragón…) de perfil, de pie. Andar, trotar, galopar, sentarse…":
    "Four-legged animals (dog, cat, horse, dragon…) standing, seen from the side. Walk, trot, gallop, sit…",
  "Personajes de pie (pose A o T), mirando al frente. Esqueleto Mixamo compatible con Unity, Unreal y Blender.":
    "Standing characters (A or T pose) facing front. Mixamo skeleton for Unity, Unreal and Blender.",
  "Detecta si es una persona o un animal de cuatro patas. Los clips que no encajen se ignoran.":
    "Detects whether it is a person or a four-legged animal. Clips that don't fit are skipped.",
  "Elige al menos una animación o sube un BVH": "Pick at least one animation or upload a BVH",
  "Espera a que el modelo cargue en el visor": "Wait for the model to load in the viewer",
  "Texturas Full HD": "Full HD textures", "✦ Mejorar texturas": "✦ Upgrade textures",
  "Amplía las texturas con IA (Real-ESRGAN) y genera mapas PBR: normal, rugosidad y oclusión.":
    "Upscales the textures with AI (Real-ESRGAN) and builds PBR maps: normal, roughness and occlusion.",
  "Amplía las texturas (modo demo: reescalado de alta calidad sin IA) y genera mapas PBR: normal, rugosidad y oclusión.":
    "Upscales the textures (demo mode: high-quality resize without AI) and builds PBR maps: normal, roughness and occlusion.",
  "Este modelo no tiene texturas (colores por vértice). Usa «Retexturizado IA» o un motor con PBR para obtener texturas HD.":
    "This model has no textures (vertex colours). Use «AI retexture» or a PBR engine to get HD textures.",
  "Este modelo usa colores por vértice: retexturízalo con IA primero": "This model uses vertex colours: retexture it with AI first",
  "Retexturizado IA": "AI retexture", "Ej.: armadura lacada roja con detalles dorados, estilo anime": "E.g. red lacquered armour with gold details, anime style",
  "Resolución de la textura": "Texture resolution", "Retexturizar": "Retexture", "Describe la textura (mín. 3 caracteres)": "Describe the texture (min. 3 characters)",
  "Remallado y estilo": "Remesh & style", "Reducir polígonos": "Reduce polygons", "Quads limpios (para animar / esculpir)": "Clean quads (for animation / sculpting)",
  "Low-poly facetado": "Faceted low-poly", "Vóxeles": "Voxels", "Suavizar": "Smooth", "Caras objetivo": "Target faces",
  "Resolución": "Resolution", "Aplicar": "Apply",
  "Analizando…": "Analysing…", "✚ Reparar": "✚ Repair", "Cortar por altura": "Cut by height", "Separar piezas": "Separate parts",
  "Número de cortes": "Number of cuts", "✂ Dividir": "✂ Split", "Abre esta sección para analizar el modelo.": "Open this section to analyse the model.",
  "✔ Listo para imprimir": "✔ Ready to print", "⚠ Necesita reparación": "⚠ Needs repair",
  "{f} triángulos · {b} pieza(s) · {x}×{y}×{z} u": "{f} triangles · {b} part(s) · {x}×{y}×{z} u", " · vol. {v} u³": " · vol. {v} u³",
  "Exportar": "Export", "Destino": "Destination", "Qué medida fijar": "Which size to set", "Alto": "Height", "Ancho": "Width",
  "Fondo": "Depth", "Lado mayor": "Longest side", "Medida final en milímetros": "Final size in millimetres",
  "Colores / filamentos": "Colours / filaments", "⬇ Descargar": "⬇ Download",
  "◆ Complemento para Blender (importar con un clic)": "◆ Blender add-on (one-click import)",
  " (malla estática: sin esqueleto ni animaciones)": " (static mesh: no skeleton or animations)", " {f} por defecto: 100 mm.": " {f} defaults to 100 mm.",
  "No se pudo crear el USDZ: {e}": "Could not create the USDZ: {e}",
  "Universal: web, Unity, Unreal, Godot, Blender, Roblox.": "Universal: web, Unity, Unreal, Godot, Blender, Roblox.",
  "OBJ + MTL + textura en un ZIP.": "OBJ + MTL + texture in a ZIP.",
  "Impresión 3D: Z hacia arriba, en milímetros, apoyado en la cama.": "3D printing: Z up, in millimetres, resting on the bed.",
  "Impresión 3D moderna (Bambu Studio, PrusaSlicer, OrcaSlicer, Cura).": "Modern 3D printing (Bambu Studio, PrusaSlicer, OrcaSlicer, Cura).",
  "Colores por vértice; útil para MeshLab y escaneo.": "Vertex colours; handy for MeshLab and scanning.",
  "3MF multicolor (AMS / MMU)": "Multicolour 3MF (AMS / MMU)",
  "Un filamento por color, ya pintado: Bambu Studio, OrcaSlicer, PrusaSlicer.": "One filament per colour, already painted: Bambu Studio, OrcaSlicer, PrusaSlicer.",
  "STL por color (.zip)": "STL per colour (.zip)",
  "Un STL por color para cualquier laminador (asigna un extrusor a cada pieza).": "One STL per colour for any slicer (assign an extruder to each part).",
  "FBX (Unity, Unreal, Maya, 3ds Max)": "FBX (Unity, Unreal, Maya, 3ds Max)",
  "Convertido con Blender; conserva esqueleto, pesos y animaciones.": "Converted with Blender; keeps the skeleton, weights and animations.",
  "Blender (.blend)": "Blender (.blend)",
  "Archivo de Blender listo para abrir: armadura, materiales PBR y cada animación en su pista NLA.":
    "Blender file ready to open: armature, PBR materials and each animation on its own NLA track.",
  "USDZ (iPhone / Apple)": "USDZ (iPhone / Apple)",
  "AR Quick Look en iPhone/iPad. Se genera en tu navegador; conserva texturas, no colores por vértice.":
    "AR Quick Look on iPhone/iPad. Built in your browser; keeps textures, not vertex colours.",
  // export destinations
  "Impresora 3D": "3D printer", "Web / AR": "Web / AR",
  "Abre el .blend: armadura, materiales PBR y cada animación en su pista NLA (01, 02…).": "Open the .blend: armature, PBR materials and every animation on its own NLA track (01, 02…).",
  "Sin Blender instalado se descarga GLB: en Blender usa Archivo → Importar → glTF (o el complemento).":
    "Without Blender installed you get a GLB: in Blender use File → Import → glTF (or the add-on).",
  "FBX con esqueleto Humanoid y una toma por animación. En Unity: Rig → Humanoid.": "FBX with a Humanoid skeleton and one take per animation. In Unity: Rig → Humanoid.",
  "GLB: arrástralo a Unity con glTFast (o instala Blender para obtener FBX).": "GLB: drop it into Unity with glTFast (or install Blender to get FBX).",
  "FBX para Unreal: impórtalo como Skeletal Mesh con sus animaciones.": "FBX for Unreal: import it as a Skeletal Mesh with its animations.",
  "GLB: Unreal 5 lo importa con Interchange (o instala Blender para FBX).": "GLB: Unreal 5 imports it with Interchange (or install Blender for FBX).",
  "GLB: Godot 4 lo importa directamente con esqueleto y animaciones.": "GLB: Godot 4 imports it directly with skeleton and animations.",
  "FBX: Roblox Studio → Importador 3D (personajes con rig incluidos).": "FBX: Roblox Studio → 3D Importer (rigged characters included).",
  "OBJ: Roblox Studio → Importador 3D (instala Blender para FBX con rig).": "OBJ: Roblox Studio → 3D Importer (install Blender for rigged FBX).",
  "GLB para la web y USDZ para AR en iPhone.": "GLB for the web and USDZ for AR on iPhone.",
  "3MF en milímetros, apoyado en la cama. Indica la medida final.": "3MF in millimetres, resting on the bed. Set the final size.",
  "Historial de versiones": "Version history", "Compartir": "Share", "⧉ Copiar enlace": "⧉ Copy link", "↗ Abrir (AR)": "↗ Open (AR)",
  "Insertar en tu web": "Embed on your site", "Enlace copiado": "Link copied", "Copia el enlace:": "Copy the link:",
  "Eliminar modelo": "Delete model", "Modelo eliminado": "Model deleted", "¿Eliminar \"{t}\"?": "Delete \"{t}\"?", "Nuevo nombre:": "New name:",
  "de": "from", "de (eliminado)": "from (deleted)", "mapa de profundidad": "depth map", "Motor de IA usado": "AI engine used",

  // ---------------------------------------------------------------- printing
  "Impresora": "Printer", "Alto final (mm)": "Final height (mm)", "Estanco": "Watertight", "Volumen": "Volume",
  "Peso aprox. (PLA)": "Approx. weight (PLA)", "Huecos": "Holes", "Bordes no manifold": "Non-manifold edges", "Tamaño": "Size",
  "Cabe en la cama": "Fits the bed", "Sí": "Yes", "No": "No", "{f} triángulos · {b} pieza(s)": "{f} triangles · {b} part(s)",
  "✂ Dividir para que quepa": "✂ Split to fit", "Corta el modelo en piezas cerradas que caben en la cama, listas para pegar":
    "Cuts the model into closed pieces that fit the bed, ready to glue", "Directo a tu laminador": "Straight to your slicer",
  "▶ Abrir": "▶ Open", "Multicolor (AMS / MMU)": "Multicolour (AMS / MMU)", "Programa predeterminado para .3mf": "Default program for .3mf",
  "Abierto en {s}": "Opened in {s}", "tu laminador": "your slicer", "Publica tu 3MF en": "Publish your 3MF on",
  "Dividiendo para la cama de impresión": "Splitting for the print bed",
  "Abrir el laminador solo funciona en la app de escritorio": "Opening the slicer only works in the desktop app",

  // ---------------------------------------------------------------- share page
  "Cargando…": "Loading…", "Modelo 3D": "3D model", "👋 Ver en tu espacio (AR)": "👋 View in your space (AR)",
  "Hecho con Forja3D": "Made with Forja3D", "Falta el identificador del modelo": "The model id is missing",
  "Este modelo no existe o fue eliminado": "This model does not exist or was deleted", "El modelo aún no está listo": "The model is not ready yet",
  "Modelo 3D — Forja3D": "3D model — Forja3D",

  // ---------------------------------------------------------------- settings
  "¡Bienvenido a Forja3D!": "Welcome to Forja3D!",
  "Elige el motor de IA. Puedes cambiarlo cuando quieras desde Ajustes.": "Pick the AI engine. You can change it any time from Settings.",
  "Los ajustes están bloqueados en este servidor.": "Settings are locked on this server.", "Idioma": "Language",
  "Nube (fal.ai)": "Cloud (fal.ai)", "Máxima calidad: Hunyuan3D 3, TRELLIS 2, Tripo, Meshy. Necesita clave.": "Top quality: Hunyuan3D 3, TRELLIS 2, Tripo, Meshy. Needs a key.",
  "Mi GPU": "My GPU", "Gratis y privado en tu NVIDIA RTX con el servidor IA local.": "Free and private on your NVIDIA RTX with the local AI server.",
  "Demo": "Demo", "Sin IA: formas de prueba para conocer la app.": "No AI: sample shapes to get to know the app.",
  "Clave de fal.ai": "fal.ai key", "Pega tu clave (fal.ai/dashboard/keys)": "Paste your key (fal.ai/dashboard/keys)",
  "Clave guardada ({h}) — déjalo vacío para mantenerla": "Saved key ({h}) — leave empty to keep it",
  "Motor 3D por defecto": "Default 3D engine", "Dirección del servidor IA local": "Local AI server address",
  "Token (opcional)": "Token (optional)", "Sólo si lo configuraste": "Only if you set one",
  "Instálalo con scripts\\windows\\instalar.bat y ábrelo con iniciar.bat -SoloWorker.": "Install it with scripts\\windows\\instalar.bat and start it with iniciar.bat -SoloWorker.",
  "Blender (para FBX y .blend)": "Blender (for FBX and .blend)", "Detección automática": "Auto-detect",
  "✔ Blender encontrado: {p}": "✔ Blender found: {p}",
  "Blender no encontrado: instálalo (gratis, blender.org) para exportar FBX y .blend.": "Blender not found: install it (free, blender.org) to export FBX and .blend.",
  "El motor guardado no arrancó: {e}": "The saved engine did not start: {e}", "Pega tu clave de fal.ai para usar la nube": "Paste your fal.ai key to use the cloud",
  "Guardar": "Save", "Cerrar": "Close", "¡Ajustes guardados!": "Settings saved!",

  // ---------------------------------------------------------------- free local AI
  "Mi GPU · gratis": "My GPU · free",
  "Recomendado. Ilimitado y privado en tu NVIDIA RTX: se instala y arranca sola, sin claves.": "Recommended. Unlimited and private on your NVIDIA RTX: installs and starts by itself, no keys.",
  "Opcional y de pago por uso: Hunyuan3D 3, TRELLIS 2, Tripo, Meshy. Necesita clave.": "Optional, pay per use: Hunyuan3D 3, TRELLIS 2, Tripo, Meshy. Needs a key.",
  "Comprobando la IA local…": "Checking the local AI…", "Instalar IA gratis en mi GPU": "Install free AI on my GPU",
  "Una sola vez: descarga ≈20 GB (PyTorch, Hunyuan3D-2, SDXL-Turbo). Después arranca sola cada vez que generas: gratis, sin límites, sin claves.":
    "One time only: downloads ≈20 GB (PyTorch, Hunyuan3D-2, SDXL-Turbo). After that it starts by itself whenever you generate: free, unlimited, no keys.",
  "Avanzado: usar un servidor IA de otro equipo": "Advanced: use an AI server on another computer",
  "Instalando…": "Installing…", "La instalación falló: {e}": "The install failed: {e}",
  "✔ IA local instalada y funcionando": "✔ Local AI installed and running",
  "✔ IA local instalada: arranca sola al generar": "✔ Local AI installed: starts by itself when you generate",
  "La IA local necesita Windows o Linux con una tarjeta NVIDIA. En Mac usa la nube.": "The local AI needs Windows or Linux with an NVIDIA card. On a Mac use the cloud.",
  "No se detecta una tarjeta NVIDIA: actualiza el driver (nvidia.com/drivers).": "No NVIDIA card detected: update the driver (nvidia.com/drivers).",
  "La IA local aún no está instalada.": "The local AI is not installed yet.", "Reintentar instalación": "Retry install",
  "Instalando la IA gratis en tu GPU": "Installing the free AI on your GPU",
  "La instalación de la IA local falló: {e}": "The local AI install failed: {e}",
  "¡IA local instalada! Ya puedes generar gratis": "Local AI installed! You can generate for free now",
  "IA local gratis": "Free local AI", "IA local · instalando {p}%": "Local AI · installing {p}%",
  "IA local gratis · arranca al generar": "Free local AI · starts when you generate", "IA local · instalar (gratis)": "Local AI · install (free)",
  "Comprobando tu tarjeta gráfica": "Checking your graphics card", "Preparando el instalador": "Preparing the installer",
  "Instalando Python": "Installing Python", "Instalando PyTorch con CUDA (≈3 GB)": "Installing PyTorch with CUDA (≈3 GB)",
  "Instalando Hunyuan3D-2 y el servidor de IA": "Installing Hunyuan3D-2 and the AI server",
  "Comprobando que PyTorch ve tu GPU": "Checking that PyTorch sees your GPU",
  "Descargando los modelos de IA (≈15 GB, solo una vez)": "Downloading the AI models (≈15 GB, one time only)",
  "Terminando": "Finishing", "Listo": "Done", "Modelos listos": "Models ready",
  "La IA local aún no está instalada: abre Ajustes → «Mi GPU» → Instalar (gratis)": "The local AI is not installed yet: open Settings → “My GPU” → Install (free)",
  "No se encontró una tarjeta NVIDIA con su driver instalado: actualiza el driver (nvidia.com/drivers) y vuelve a intentarlo": "No NVIDIA card with its driver was found: update the driver (nvidia.com/drivers) and try again",
  "La IA local tarda demasiado en arrancar": "The local AI is taking too long to start",
  "No se pudo descargar el instalador uv": "Could not download the uv installer",

  // ---------------------------------------------------------------- AI agent
  "Agente IA": "AI agent", "Agente IA: crea y anima hablando": "AI agent: create and animate by chatting",
  "Ideas, modelos, animaciones e impresión, hablando": "Ideas, models, animations and printing, by chatting",
  "Nueva conversación": "New conversation",
  "Hola, soy tu asistente 3D. Cuéntame qué quieres crear y lo hago contigo: conceptos, modelo 3D, rig con animaciones o revisión para imprimir.":
    "Hi, I'm your 3D assistant. Tell me what you want to make and we'll build it together: concepts, 3D model, rig with animations or a print check.",
  "Un dragón bebé estilo cartoon para animar": "A cartoon baby dragon to animate",
  "Una maceta con forma de gato para imprimir en mi Bambu A1": "A cat-shaped planter to print on my Bambu A1",
  "Dame ideas para un personaje de videojuego": "Give me ideas for a video game character",
  "Escribe tu idea o pregunta…": "Type your idea or question…", "Enviar": "Send", "Pensando…": "Thinking…",
  "Concepto {n}": "Concept {n}", "Hacer este concepto en 3D": "Make this concept 3D",
  "Haz en 3D el concepto {n}": "Make concept {n} in 3D", "Toca uno para hacerlo en 3D": "Tap one to make it 3D",
  "en cola": "queued", "generando": "generating", "listo": "ready", "falló": "failed", "Ver": "View",
  "Agente IA: clave de Anthropic": "AI agent: Anthropic key",
  "Pega tu clave (console.anthropic.com)": "Paste your key (console.anthropic.com)",
  "Opcional: el agente ya funciona gratis. Con una clave de Anthropic entiende cualquier petición y conversa como un experto.":
    "Optional: the agent already works for free. With an Anthropic key it understands any request and chats like an expert.",
  "Añade tu clave de Anthropic en Ajustes para usar el agente": "Add your Anthropic key in Settings to use the agent",
  "La clave de Anthropic no es válida: revísala en Ajustes": "The Anthropic key is not valid: check it in Settings",
  "Tu clave de Anthropic no tiene permiso para usar el agente": "Your Anthropic key is not allowed to use the agent",
  "Tu clave de Anthropic no tiene modelos disponibles": "Your Anthropic key has no models available",
  "Demasiadas peticiones al agente: espera unos segundos": "Too many requests to the agent: wait a few seconds",
  "Sin conexión con el agente: revisa tu internet": "Cannot reach the agent: check your internet",
  "El agente no respondió": "The agent did not answer",
  "Borrador": "Draft", "Estándar": "Standard",
  "Tareas": "Tasks", "Animales": "Animals", "Escalar": "Climbing", "Ejercicio": "Exercise",

  // ---------------------------------------------------------------- server: animations, markers, mocap
  "Reposo (respirar)": "Idle (breathing)", "Caminar": "Walk", "Correr": "Run", "Saltar": "Jump", "Saludar": "Wave",
  "Bailar": "Dance", "Puñetazos": "Punches", "Victoria": "Victory", "Asentir (sí)": "Nod (yes)", "Negar (no)": "Shake head (no)",
  "Mirar alrededor": "Look around", "Sentado": "Sitting", "T-pose": "T-pose", "A-pose": "A-pose",
  "Reposo": "Idle", "Paso": "Walk", "Trote": "Trot", "Galope": "Gallop", "Contento (cola)": "Happy (tail wag)",
  "Barbilla": "Chin", "Ingle": "Groin", "Hombro izq.": "L shoulder", "Hombro der.": "R shoulder", "Codo izq.": "L elbow",
  "Codo der.": "R elbow", "Muñeca izq.": "L wrist", "Muñeca der.": "R wrist", "Rodilla izq.": "L knee", "Rodilla der.": "R knee",
  "Tobillo izq.": "L ankle", "Tobillo der.": "R ankle", "Hocico": "Snout", "Nuca": "Head", "Cruz (lomo delantero)": "Withers",
  "Grupa": "Rump", "Punta cola": "Tail tip", "Hombro": "Shoulder", "Codo": "Elbow", "Muñeca delantera": "Front paw",
  "Cadera": "Hip", "Rodilla": "Knee", "Corvejón": "Hock",
  "Baile": "Dance", "Combate": "Combat", "Acrobacias": "Acrobatics", "Deporte": "Sport", "Gestos": "Gestures", "Sentarse": "Sitting",
  "Otros": "Other",
  "Reposo natural": "Natural idle", "Esperar": "Wait", "Quieto de pie": "Standing still", "Esperar impaciente": "Impatient wait",
  "Caminar lento y parar": "Slow walk and stop", "Caminar girando": "Walk turning", "Caminar y girar 90°": "Walk and turn 90°",
  "Caminar triste": "Sad walk", "Caminar asustado": "Scared walk", "Caminar borracho": "Drunk walk", "Caminar feliz": "Happy walk",
  "Caminar sigiloso": "Sneak", "Caminar como dinosaurio": "Dinosaur walk", "Terreno irregular": "Uneven ground", "Trotar": "Jog",
  "Correr y frenar": "Run and stop", "Correr y girar": "Run and turn", "Correr en círculo": "Run in a circle", "Salto alto": "High jump",
  "Salto hacia delante": "Forward jump", "Salto con giro": "Spin jump", "Salto de 180°": "180° jump", "Charlestón": "Charleston",
  "Breakdance": "Breakdance", "Baile ruso": "Russian dance", "Bailar libre": "Freestyle dance", "Baile divertido": "Fun dance",
  "Danza con pirueta": "Pirouette dance", "Baile del pollito": "Chicken dance", "Lindy hop": "Lindy hop", "Boxeo": "Boxing",
  "Puñetazo y patada": "Punch and kick", "Patada voladora": "Flying kick", "Patada giratoria": "Spinning kick", "Golpes": "Strikes",
  "Espada": "Sword", "Patada frontal": "Front kick", "Mortal atrás": "Backflip", "Rueda lateral": "Cartwheel", "Mortal lateral": "Side flip",
  "Rueda y mortal": "Cartwheel and flip", "Mortal mono": "Monkey flip", "Tiro de baloncesto": "Basketball shot", "Tiro en suspensión": "Jump shot",
  "Driblar": "Dribble", "Lanzar (béisbol)": "Pitch (baseball)", "Batear": "Bat swing", "Lanzar": "Throw", "Disparar con arco": "Bow shot",
  "Saludar con la mano": "Hand wave", "Saludo": "Salute", "Reverencia": "Bow", "Muy feliz": "Very happy", "Triste": "Sad",
  "Asustado": "Scared", "Alegre": "Cheerful", "Discutir enfadado": "Angry argument", "Sentarse y levantarse": "Sit down and stand up",
  "Sentarse en silla": "Sit on a chair", "Sentarse en taburete": "Sit on a stool", "Zombi": "Zombie", "Mono": "Monkey",
  "Andar como gallina": "Chicken walk",

  // ---------------------------------------------------------------- server: progress, engines
  "Iniciando": "Starting", "Completado": "Done", "Esperando el paso anterior": "Waiting for the previous step",
  "Analizando entrada": "Analysing input", "Generando geometría": "Generating geometry", "Refinando malla": "Refining mesh",
  "Aplicando texturas": "Applying textures", "Exportando GLB": "Exporting GLB", "Analizando UVs": "Analysing UVs",
  "Pintando textura": "Painting texture", "Horneando materiales": "Baking materials", "Ampliando texturas": "Upscaling textures",
  "Ampliando texturas con IA": "Upscaling textures with AI", "Aplicando estilo": "Applying style", "Calculando grosores": "Computing thickness",
  "Colocando esqueleto y calculando pesos": "Placing the skeleton and computing weights", "Construyendo malla": "Building mesh",
  "Convirtiendo a GLB": "Converting to GLB", "Descargando modelo": "Downloading model", "Descargando texturas": "Downloading textures",
  "Enviando a Meshy 7 Retexture": "Sending to Meshy 7 Retexture", "Estimando profundidad": "Estimating depth",
  "Estimando profundidad con IA": "Estimating depth with AI", "Generando imagen de la textura": "Generating the texture image",
  "Generando imagen de referencia": "Generating a reference image", "Generando imagen de referencia en tu GPU": "Generating a reference image on your GPU",
  "Remallado en quads (Instant Meshes)": "Quad remeshing (Instant Meshes)", "Remallando": "Remeshing", "Reparando malla": "Repairing mesh",
  "Dividiendo en piezas": "Splitting into parts", "Importando modelo": "Importing model",
  "Demo (sin IA)": "Demo (no AI)", "Formas procedurales para probar la app.": "Procedural shapes to try the app.",
  "Hunyuan3D-2mini (tu GPU)": "Hunyuan3D-2mini (your GPU)", "Hunyuan3D-2 (tu GPU)": "Hunyuan3D-2 (your GPU)",
  "Texto o imagen → 3D nativo, PBR, hasta 1 M de caras, hasta 4 vistas.": "Text or image → native 3D, PBR, up to 1 M faces, up to 4 views.",
  "Geometría muy fiel a la imagen y texturas de hasta 4K.": "Geometry very faithful to the image and textures up to 4K.",
  "PBR y calidad «detailed» de geometría y textura.": "PBR and “detailed” geometry and texture quality.",
  "PBR, simetría y pose A/T para personajes listos para rig.": "PBR, symmetry and A/T pose for rig-ready characters.",
  "TRELLIS 1 (económico)": "TRELLIS 1 (budget)", "La generación anterior: rápida y barata.": "The previous generation: fast and cheap.",
  "CPU (Lanczos)": "CPU (Lanczos)", "IA local (Real-ESRGAN)": "Local AI (Real-ESRGAN)", "IA en la nube (Real-ESRGAN)": "Cloud AI (Real-ESRGAN)",

  // ---------------------------------------------------------------- server: errors
  "Error interno durante el procesamiento": "Internal error while processing", "Cancelado": "Cancelled",
  "Este modelo no tiene texturas (usa colores por vértice). Retexturízalo primero.": "This model has no textures (it uses vertex colours). Retexture it first.",
  "Mejora las texturas del modelo original y después crea el rig": "Upgrade the original model's textures, then create the rig",
  "Este personaje ya tiene rig: usa «Actualizar animaciones»": "This character is already rigged: use «Update animations»",
  "Primero crea el rig del personaje": "Create the character's rig first", "El modelo todavía no está listo": "The model is not ready yet",
  "Trabajo no encontrado": "Job not found", "Sube al menos una imagen": "Upload at least one image", "Máximo 4 imágenes": "4 images at most",
  "Formato no soportado: usa PNG, JPG o WEBP": "Unsupported format: use PNG, JPG or WEBP", "La imagen está vacía": "The image is empty",
  "La imagen está dañada o no es válida": "The image is damaged or invalid", "Concepto no encontrado": "Concept not found",
  "Sólo se pueden refinar o variar modelos generados por IA": "Only AI-generated models can be refined or varied",
  "El modelo original de este personaje ya no existe": "This character's original model no longer exists",
  "Para FBX/.blend instala Blender (o indica BLENDER_PATH en .env) y reinicia la app": "For FBX/.blend install Blender (or set its path in Settings)",
  "Los ajustes están bloqueados en este servidor": "Settings are locked on this server", "El BVH supera 20 MB": "The BVH is larger than 20 MB",
  "La captura de movimiento BVH solo se aplica a personajes humanoides": "BVH motion capture only applies to humanoid characters",
  "El BVH no contiene fotogramas de movimiento": "The BVH has no motion frames",
  "El BVH no tiene un esqueleto humanoide reconocible (caderas, piernas y brazos)": "The BVH has no recognisable humanoid skeleton (hips, legs and arms)",
  "El archivo no es un BVH válido (falta HIERARCHY)": "The file is not a valid BVH (HIERARCHY missing)",
  "No se pudo leer el BVH (formato inesperado)": "Could not read the BVH (unexpected format)",
  "El modelo no contiene mallas": "The model has no meshes", "El archivo no contiene ninguna malla": "The file has no mesh",
  "El modelo no contiene triángulos": "The model has no triangles", "El modelo no tiene altura": "The model has no height",
  "El modelo no tiene volumen": "The model has no volume", "La malla no tiene caras válidas": "The mesh has no valid faces",
  "No se encontraron piezas separables con ese modo": "No separable parts found with that mode",
  "Para cortar por altura la malla debe ser estanca: usa primero «Reparar»": "To cut by height the mesh must be watertight: use «Repair» first",
  "No se pudo generar la malla de quads (¿modelo demasiado pequeño o roto?)": "Could not build the quad mesh (model too small or broken?)",
  "La resolución es demasiado baja para este modelo": "The resolution is too low for this model",
  "La profundidad es demasiado irregular; desactiva «cortar bordes»": "The depth is too irregular; turn off «cut edges»",
  "Formato no soportado: usa GLB, OBJ, STL, PLY, OFF o 3MF": "Unsupported format: use GLB, OBJ, STL, PLY, OFF or 3MF",
  "El worker de IA local no respondió a tiempo": "The local AI worker did not answer in time", "La GPU local tardó demasiado": "The local GPU took too long",
  "Falta FAL_KEY: configura tu clave de fal.ai o usa GEN_PROVIDER=mock": "fal.ai key missing: add it in Settings or use demo mode",
  "El generador de imágenes no devolvió resultados": "The image generator returned nothing",
  "El retexturizado no devolvió ningún modelo": "Retexturing returned no model", "El ampliador no devolvió ninguna imagen": "The upscaler returned no image",
  "El modelo de profundidad no devolvió ningún mapa": "The depth model returned no map", "Mapa de profundidad no válido": "Invalid depth map",
  "No hay imagen de entrada": "No input image", "No se pudo leer la imagen": "Could not read the image",
  "El texto no contiene caracteres visibles": "The text has no visible characters", "Máximo 4 líneas de texto": "4 lines of text at most",
};

// server messages with variable parts: [regex, English template]
const PATTERNS = [
  [/^Descargando: (.+)$/, "Downloading: $1"],
  [/^La IA local se cerró al arrancar: (.*)$/s, "The local AI closed while starting: $1"],
  [/^No se puede conectar con la IA local en (\S+): ábrela desde Ajustes → «Mi GPU»$/, "Cannot reach the local AI at $1: open it from Settings → “My GPU”"],
  [/^Falló «(.+?)»: (.*)$/s, "“$1” failed: $2"],
  [/^El agente no respondió \((\d+)\)$/, "The agent did not answer ($1)"],
  [/^Generando con (.+)$/, "Generating with $1"],
  [/^Reconstruyendo con (.+)$/, "Rebuilding with $1"],
  [/^(.+) no devolvió ninguna malla$/, "$1 returned no mesh"],
  [/^Faltan marcadores: (.+)$/, "Missing markers: $1"],
  [/^Animaciones desconocidas: (.+)$/, "Unknown animations: $1"],
  [/^Animación no disponible para este tipo de esqueleto: (.+)$/, "Animation not available for this skeleton type: $1"],
  [/^Animación desconocida para este tipo de esqueleto: (.+)$/, "Unknown animation for this skeleton type: $1"],
  [/^Tipo de esqueleto desconocido: (.+)$/, "Unknown skeleton type: $1"],
  [/^Motor desconocido: (.+)$/, "Unknown engine: $1"],
  [/^La imagen supera (\d+) MB$/, "The image is larger than $1 MB"],
  [/^El modelo supera (\d+) MB$/, "The model is larger than $1 MB"],
  [/^(.+) no está disponible con el proveedor «(.+)»$/, "$1 is not available with the «$2» provider"],
  [/^No se pudo leer el archivo (\w+): (.+)$/, "Could not read the $1 file: $2"],
  [/^Tiempo de espera agotado en (.+)$/, "Timed out waiting for $1"],
  [/^Error (\d+)$/, "Error $1"],
];

// generated titles: "<name> · reducido" → "<name> · reduced"
const SUFFIX = { reducido: "reduced", "vóxel": "voxel", suavizado: "smoothed", reparado: "repaired", piezas: "parts" };

export function t(text, vars) {
  let out = lang === "es" || text == null ? text : (EN[text] ?? text);
  if (vars) for (const [k, v] of Object.entries(vars)) out = out.replaceAll(`{${k}}`, v);
  return out;
}

// text that came from the server
export function tr(text) {
  if (lang === "es" || !text) return text;
  if (EN[text]) return EN[text];
  for (const [re, en] of PATTERNS) if (re.test(text)) return text.replace(re, en);
  return text;
}

export function title(text) {
  if (lang === "es" || !text) return text;
  return text.replace(/ · ([^·]+)$/, (m, s) => ` · ${SUFFIX[s] || s}`);
}

// Translate the static page: text nodes, placeholders, titles and labels.
export function applyI18n(root = document.body) {
  if (lang !== "es") {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const raw = node.nodeValue;
      const key = raw.trim();
      if (key && EN[key]) node.nodeValue = raw.replace(key, EN[key]);
    }
    for (const el of root.querySelectorAll("[placeholder], [title], [aria-label], [alt]")) {
      for (const attr of ["placeholder", "title", "aria-label", "alt"]) {
        const v = el.getAttribute(attr);
        if (v && EN[v]) el.setAttribute(attr, EN[v]);
      }
    }
    if (root === document.body) document.title = t(document.title);
  }
  document.documentElement.classList.remove("i18n-wait");
  for (const b of document.querySelectorAll("[data-lang]")) {
    b.classList.toggle("on", b.dataset.lang === lang);
    b.addEventListener("click", () => setLang(b.dataset.lang));
  }
}
