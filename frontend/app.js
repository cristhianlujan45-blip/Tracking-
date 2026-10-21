import { applyI18n, lang, t, title, tr } from "./i18n.js";
import { clipLabel, initRig, renderRig } from "./rig.js";
import { initAgent } from "./agent.js";
import { initLocalAI } from "./localai.js";
import { initUpdates } from "./updates.js";
import { initSettings, openSettings } from "./settings.js";
import { Viewer } from "./viewer.js";

applyI18n(); // translate the static page before anything else renders

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

// prompt ideas by category, in the interface language (the engines understand both)
const EXAMPLES = {
  es: {
    Personaje: ["una guerrera samurái con armadura roja", "un orco guardián con armadura de bronce", "un robot mecha compacto", "una maga con capa y bastón"],
    Criatura: ["un zorro kitsune de nueve colas", "un dragón chibi", "un lobo de cristal", "un gato astronauta"],
    Arma: ["una katana con funda lacada", "un martillo de guerra rúnico", "una pistola de plasma sci-fi", "un arco élfico"],
    Edificio: ["un templo japonés con cerezos", "una taberna medieval", "una torre cyberpunk con neones", "una cabaña en el bosque"],
    Accesorio: ["un cofre del tesoro con remaches dorados", "una poción mágica brillante", "un casco de samurái", "una linterna de papel"],
  },
  en: {
    Personaje: ["a samurai warrior with red armour", "an orc guardian in bronze armour", "a compact mecha robot", "a mage with a cloak and staff"],
    Criatura: ["a nine-tailed kitsune fox", "a chibi dragon", "a crystal wolf", "an astronaut cat"],
    Arma: ["a katana with a lacquered sheath", "a runic war hammer", "a sci-fi plasma pistol", "an elven bow"],
    Edificio: ["a Japanese temple with cherry trees", "a medieval tavern", "a cyberpunk neon tower", "a cabin in the woods"],
    Accesorio: ["a treasure chest with gold rivets", "a glowing magic potion", "a samurai helmet", "a paper lantern"],
  },
}[lang === "es" ? "es" : "en"];
const STYLE_ICONS = { auto: "✧", realistic: "◉", cartoon: "☺", handpainted: "✎", lowpoly: "◆", voxel: "▦", sculpture: "◍" };
const MODE_LABEL = {
  text: "Texto", image: "Imagen", multi_image: "Multi-vista", upload: "Subido",
  remesh: "Remallado", retexture: "Retexturizado", repair: "Reparado", split: "Piezas",
  scene: "Escena", relief: "Relieve", lithophane: "Litofanía", text3d: "Texto 3D", rig: "Animado",
  enhance: "Texturas HD",
};
const MODE_ICON = {
  text: "✎", image: "▣", multi_image: "▤", upload: "⇪", remesh: "◇", retexture: "✺", repair: "✚", split: "✂",
  scene: "⛰", relief: "▨", lithophane: "◐", text3d: "A", rig: "⚡", enhance: "✦",
};
const TOOL_MODES = ["remesh", "retexture", "repair", "split", "enhance"];
const FLAT_MODES = ["relief", "lithophane"]; // plaques: size them by the longest side, not the thickness
const SCENE_INFO = {
  scene: ["Profundidad", "Malla 3D con la foto como textura. Ideal como fondo o entorno en Blender, Unity o Unreal."],
  relief: ["Relieve", "Placa sólida con la escena en relieve (bajorrelieve). Lista para imprimir en 3D."],
  lithophane: ["Grosor", "Placa fina: al iluminarla por detrás aparece la foto. Imprime en blanco, de pie, al 100 % de relleno."],
};
const STATUS_LABEL = { queued: "En cola", running: "Generando", succeeded: "", failed: "Error" };
const QUALITY_LABEL = { draft: "Borrador", standard: "Estándar", high: "Full HD" };
const GENERATIVE = ["text", "image", "multi_image"];
const FORMAT_INFO = {
  glb: ["GLB", "Universal: web, Unity, Unreal, Godot, Blender, Roblox."],
  obj: ["OBJ (.zip)", "OBJ + MTL + textura en un ZIP."],
  stl: ["STL", "Impresión 3D: Z hacia arriba, en milímetros, apoyado en la cama."],
  "3mf": ["3MF", "Impresión 3D moderna (Bambu Studio, PrusaSlicer, OrcaSlicer, Cura)."],
  ply: ["PLY", "Colores por vértice; útil para MeshLab y escaneo."],
  "3mf_color": ["3MF multicolor (AMS / MMU)", "Un filamento por color, ya pintado: Bambu Studio, OrcaSlicer, PrusaSlicer."],
  stl_colors: ["STL por color (.zip)", "Un STL por color para cualquier laminador (asigna un extrusor a cada pieza)."],
  fbx: ["FBX (Unity, Unreal, Maya, 3ds Max)", "Convertido con Blender; conserva esqueleto, pesos y animaciones."],
  blend: ["Blender (.blend)", "Archivo de Blender listo para abrir: armadura, materiales PBR y cada animación en su pista NLA."],
  usdz: ["USDZ (iPhone / Apple)", "AR Quick Look en iPhone/iPad. Se genera en tu navegador; conserva texturas, no colores por vértice."],
};
const CLIENT_FORMATS = ["usdz"];
const COLOR_FORMATS = ["3mf_color", "stl_colors"];
const SLOT_LABELS = ["Frente", "Lateral", "Espalda", "Lateral derecho"];
const SLOT_ROLE = { front: 0, left: 1, back: 2, right: 3 };
// export destinations: pick the best format available for each tool (FBX/.blend need Blender)
const TARGETS = [
  { id: "blender", label: "Blender", pick: (f) => (f.includes("blend") ? "blend" : "glb"),
    hint: (f) => (f.includes("blend") ? "Abre el .blend: armadura, materiales PBR y cada animación en su pista NLA (01, 02…)."
      : "Sin Blender instalado se descarga GLB: en Blender usa Archivo → Importar → glTF (o el complemento).") },
  { id: "unity", label: "Unity", pick: (f) => (f.includes("fbx") ? "fbx" : "glb"),
    hint: (f) => (f.includes("fbx") ? "FBX con esqueleto Humanoid y una toma por animación. En Unity: Rig → Humanoid."
      : "GLB: arrástralo a Unity con glTFast (o instala Blender para obtener FBX).") },
  { id: "unreal", label: "Unreal", pick: (f) => (f.includes("fbx") ? "fbx" : "glb"),
    hint: (f) => (f.includes("fbx") ? "FBX para Unreal: impórtalo como Skeletal Mesh con sus animaciones."
      : "GLB: Unreal 5 lo importa con Interchange (o instala Blender para FBX).") },
  { id: "godot", label: "Godot", pick: () => "glb", hint: () => "GLB: Godot 4 lo importa directamente con esqueleto y animaciones." },
  { id: "roblox", label: "Roblox", pick: (f) => (f.includes("fbx") ? "fbx" : "obj"),
    hint: (f) => (f.includes("fbx") ? "FBX: Roblox Studio → Importador 3D (personajes con rig incluidos)."
      : "OBJ: Roblox Studio → Importador 3D (instala Blender para FBX con rig).") },
  { id: "web", label: "Web / AR", pick: () => "glb", hint: () => "GLB para la web y USDZ para AR en iPhone." },
  { id: "print", label: "Impresora 3D", pick: () => "3mf", hint: () => "3MF en milímetros, apoyado en la cama. Indica la medida final." },
];

const state = {
  config: { capabilities: [], styles: [], poses: [], export_formats: [] },
  mode: "text",
  style: "auto",
  variants: 1,
  imageFile: null,
  multiFiles: [null, null, null, null],
  modelFile: null,
  sceneFile: null,
  sceneKind: "scene",
  textStyle: "letters",
  concepts: [],
  conceptPick: null,
  jobs: [],
  selected: null,
  loadedModel: null,
  filter: "all",
  search: "",
  toolsKey: null,
  analysis: {},
  pollTimer: null,
  target: null,
  exampleCat: "Personaje",
};
const viewer = new Viewer($("#viewer"));

// ================================================================ helpers

async function api(path, { json, ...options } = {}) {
  if (json !== undefined) {
    options.body = JSON.stringify(json);
    options.headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  }
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let msg = `Error ${resp.status}`;
    try {
      const body = await resp.json();
      msg = typeof body.detail === "string" ? body.detail : body.detail?.[0]?.msg || msg;
    } catch {}
    throw new Error(tr(msg));
  }
  return resp.status === 204 ? null : resp.json();
}

let toastTimer;
function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = true;
  void el.offsetWidth; // replay the pop-in animation
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 3500);
}

function downloadBlob(blob, name) {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

const has = (cap) => state.config.capabilities.includes(cap);
const slug = (s) => (s || "modelo").normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^\w]+/g, "-").replace(/^-|-$/g, "").toLowerCase().slice(0, 50);
const selectedJob = () => state.jobs.find((j) => j.id === state.selected);

function setupDropzone(zone, { accept, onFile }) {
  const input = zone.querySelector("input");
  input.addEventListener("change", (e) => e.target.files[0] && onFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => (e.preventDefault(), zone.classList.add("drag"))));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => (e.preventDefault(), zone.classList.remove("drag"))));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) accept(file) ? onFile(file) : showFormError(t("Formato de archivo no soportado"));
  });
}

// character sheet → cut-out views in the multi-view slots
async function splitSheet(file) {
  const hint = $("#sheetHint");
  $("#sheetBtn").classList.add("busy");
  hint.textContent = t("Buscando las vistas…");
  try {
    const body = new FormData();
    body.append("image", file);
    const { views } = await api("/api/sheet/split", { method: "POST", body });
    const zones = $$("#slots .dropzone");
    const used = [];
    for (const view of views) {
      const slot = SLOT_ROLE[view.role];
      if (slot === undefined) continue;
      const blob = await (await fetch(view.url)).blob();
      const f = new File([blob], `${view.role}.png`, { type: "image/png" });
      state.multiFiles[slot] = f;
      previewInto(zones[slot], f);
      used.push(t(SLOT_LABELS[slot]).toLowerCase());
    }
    const extra = views.length - used.length;
    hint.textContent = t("Vistas detectadas: {v}", { v: used.join(", ") }) +
      (extra > 0 ? t(" (+{n} sin usar, p. ej. 3/4)", { n: extra }) : "");
  } catch (err) {
    hint.textContent = err.message;
  } finally {
    $("#sheetBtn").classList.remove("busy");
    $("#sheetFile").value = "";
  }
}

function previewInto(zone, file) {
  const img = zone.querySelector("img");
  if (img) {
    img.src = URL.createObjectURL(file);
    img.hidden = false;
    zone.querySelector(".hint").hidden = true;
  } else {
    zone.querySelector(".hint").innerHTML = `📦 ${file.name}<br><small>${(file.size / 1048576).toFixed(1)} MB</small>`;
  }
  zone.classList.add("has-file");
  showFormError("");
}

const isImage = (f) => /^image\/(png|jpeg|webp)$/.test(f.type);

// ================================================================ create form

function renderExamples() {
  $$("#exampleCats button").forEach((b) => b.classList.toggle("on", b.dataset.cat === state.exampleCat));
  $("#examples").replaceChildren(
    ...EXAMPLES[state.exampleCat].map((text) => {
      const chip = Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: text });
      chip.onclick = () => ($("#prompt").value = text);
      return chip;
    }),
  );
}

function buildForm() {
  $("#exampleCats").replaceChildren(
    ...Object.keys(EXAMPLES).map((cat) => {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: t(cat) });
      b.dataset.cat = cat;
      b.onclick = () => ((state.exampleCat = cat), renderExamples());
      return b;
    }),
  );
  renderExamples();

  for (const { id, label } of state.config.styles) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "style" + (id === state.style ? " on" : "");
    b.dataset.style = id;
    b.innerHTML = `<span class="ico">${STYLE_ICONS[id] || "•"}</span>`;
    b.append(tr(label));
    b.onclick = () => setStyle(id);
    $("#styles").append(b);
    $("#retexStyle").append(new Option(t("Estilo: {s}", { s: tr(label) }), id));
  }
  for (const { id, label } of state.config.poses) $("#pose").append(new Option(tr(label), id));
  const engines = state.config.engines || [];
  for (const e of engines) $("#engine").append(new Option(tr(e.label), e.id));
  $("#engine").value = state.config.default_engine || engines[0]?.id || "";
  $("#engineRow").hidden = engines.length < 2;
  const engineHint = () => ($("#engineHint").textContent = tr(engines.find((e) => e.id === $("#engine").value)?.description || ""));
  $("#engine").addEventListener("change", engineHint);
  engineHint();
  for (const f of [...state.config.export_formats, ...CLIENT_FORMATS]) {
    $("#exportFormat").append(new Option(t(FORMAT_INFO[f]?.[0] || f), f));
  }
  setupPrinting();
  $("#targets").replaceChildren(
    ...TARGETS.map((target) => {
      const b = Object.assign(document.createElement("button"), { type: "button", textContent: t(target.label) });
      b.dataset.target = target.id;
      b.onclick = () => chooseTarget(target.id);
      return b;
    }),
  );

  $("#sheetFile").addEventListener("change", (e) => e.target.files[0] && splitSheet(e.target.files[0]));
  SLOT_LABELS.forEach((label, i) => {
    const zone = document.createElement("label");
    zone.className = "dropzone";
    zone.innerHTML = `<input type="file" accept="image/png,image/jpeg,image/webp" hidden /><img alt="" hidden /><span class="hint">${t(label)}</span>`;
    setupDropzone(zone, { accept: isImage, onFile: (f) => ((state.multiFiles[i] = f), previewInto(zone, f)) });
    $("#slots").append(zone);
  });
  setupDropzone($("#dropImage"), { accept: isImage, onFile: (f) => ((state.imageFile = f), previewInto($("#dropImage"), f)) });
  setupDropzone($("#dropScene"), { accept: isImage, onFile: (f) => ((state.sceneFile = f), previewInto($("#dropScene"), f)) });
  segmented("#sceneKind", "kind", (kind) => ((state.sceneKind = kind), updateSceneUi()));
  segmented("#enhanceTarget", "v", () => {});
  state.enhanceHint = t(has("upscale")
    ? "Amplía las texturas con IA (Real-ESRGAN) y genera mapas PBR: normal, rugosidad y oclusión."
    : "Amplía las texturas (modo demo: reescalado de alta calidad sin IA) y genera mapas PBR: normal, rugosidad y oclusión.");
  $("#enhanceHint").textContent = state.enhanceHint;
  segmented("#textStyle", "style", (style) => ((state.textStyle = style), ($("#plateColorRow").hidden = style === "letters")));
  ["sceneRes", "sceneStrength", "textDepth"].forEach((id) => $(`#${id}`).addEventListener("input", updateSceneUi));
  updateSceneUi();
  setupDropzone($("#dropModel"), {
    accept: (f) => state.config.upload_types.includes(f.name.split(".").pop().toLowerCase()),
    onFile: (f) => ((state.modelFile = f), previewInto($("#dropModel"), f)),
  });

  // capability gating
  $$("[data-cap]").forEach((el) => {
    if (has(el.dataset.cap)) return;
    if (el.classList.contains("tab")) {
      el.disabled = true;
      el.title = t("No disponible con el proveedor actual");
    } else el.hidden = true;
  });

  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  $$("#variants button").forEach((b) =>
    b.addEventListener("click", () => {
      state.variants = Number(b.dataset.v);
      $$("#variants button").forEach((x) => x.classList.toggle("on", x === b));
    }),
  );
  setMode("text");
}

function segmented(selector, key, onChange) {
  $$(`${selector} button`).forEach((b) =>
    b.addEventListener("click", () => {
      $$(`${selector} button`).forEach((x) => x.classList.toggle("on", x === b));
      onChange(b.dataset[key]);
    }),
  );
}

function updateSceneUi() {
  const [label, hint] = SCENE_INFO[state.sceneKind];
  $("#sceneHint").textContent = t(hint);
  $("#sceneStrengthLabel").textContent = t(label);
  $("#sceneCutRow").hidden = state.sceneKind !== "scene";
  $("#sceneResValue").textContent = `${$("#sceneRes").value} px`;
  $("#sceneStrengthValue").textContent = `${Math.round(Number($("#sceneStrength").value) * 100)} %`;
  $("#textDepthValue").textContent = `${Math.round(Number($("#textDepth").value) * 100)} %`;
}

function setStyle(id) {
  state.style = id;
  $$("#styles .style").forEach((b) => b.classList.toggle("on", b.dataset.style === id));
}

function setMode(mode) {
  state.mode = mode;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  $$("#form [data-for]").forEach((el) => {
    const visible = el.dataset.for.split(" ").includes(mode) && (!el.dataset.cap || has(el.dataset.cap));
    el.hidden = !visible;
  });
  $("#concepts").hidden = mode !== "text" || !state.concepts.length;
  $("#submit").textContent = t(
    mode === "upload" ? "Importar modelo" : ["scene", "text3d"].includes(mode) ? "Crear modelo 3D" : "Generar modelo 3D");
  showFormError("");
}

function showFormError(msg) {
  $("#formError").textContent = msg;
  $("#formError").hidden = !msg;
}

function generationOptions() {
  const seed = $("#seed").value.trim();
  return {
    quality: $("#quality").value,
    style: state.style,
    variants: state.variants,
    symmetry: $("#symmetry").checked,
    pose: $("#pose").value,
    ...(seed ? { seed: Number(seed) } : {}),
    ...($("#engineRow").hidden || !$("#engine").value ? {} : { engine: $("#engine").value }),
  };
}

function imageForm(opts, prompt) {
  const body = new FormData();
  body.append("quality", opts.quality);
  body.append("style", opts.style);
  body.append("variants", opts.variants);
  if (opts.seed !== undefined) body.append("seed", opts.seed);
  if (opts.engine) body.append("engine", opts.engine);
  if (prompt) body.append("prompt", prompt);
  return body;
}

async function withBusy(button, fn) {
  button.disabled = true;
  try {
    await fn();
  } catch (err) {
    showFormError(err.message);
  } finally {
    button.disabled = false;
  }
}

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  withBusy($("#submit"), async () => {
    const opts = generationOptions();
    let created;
    if (state.mode === "text") {
      const prompt = $("#prompt").value.trim();
      if (prompt.length < 3) throw new Error(t("Escribe una descripción de al menos 3 caracteres"));
      created = await api("/api/jobs/text", { method: "POST", json: { prompt, ...opts } });
    } else if (state.mode === "image") {
      if (!state.imageFile) throw new Error(t("Selecciona una imagen primero"));
      const body = imageForm(opts, $("#imagePrompt").value.trim());
      body.append("images", state.imageFile);
      created = await api("/api/jobs/image", { method: "POST", body });
    } else if (state.mode === "multi") {
      const ROLES = ["front", "left", "back", "right"];
      const slots = state.multiFiles.map((f, i) => [f, ROLES[i]]).filter(([f]) => f);
      if (slots.length < 2) throw new Error(t("Añade al menos 2 vistas"));
      const body = imageForm({ ...opts, style: "auto" });
      slots.forEach(([f, role]) => {  // each picture with its slot: front + back is not front + side
        body.append("images", f);
        body.append("roles", role);
      });
      created = await api("/api/jobs/image", { method: "POST", body });
    } else if (state.mode === "scene") {
      if (!state.sceneFile) throw new Error(t("Selecciona una foto"));
      const body = new FormData();
      body.append("image", state.sceneFile);
      body.append("kind", state.sceneKind);
      body.append("resolution", $("#sceneRes").value);
      body.append("strength", $("#sceneStrength").value);
      body.append("cut_edges", $("#sceneCut").checked);
      created = await api("/api/jobs/scene", { method: "POST", body });
    } else if (state.mode === "text3d") {
      const text = $("#text3d").value.trim();
      if (!text) throw new Error(t("Escribe algún texto"));
      created = await api("/api/jobs/text3d", {
        method: "POST",
        json: {
          text, style: state.textStyle, font: $("#textFont").value, bold: $("#textBold").checked,
          depth: Number($("#textDepth").value), color: $("#textColor").value, plate_color: $("#plateColor").value,
        },
      });
    } else {
      if (!state.modelFile) throw new Error(t("Selecciona un archivo 3D"));
      const body = new FormData();
      body.append("model", state.modelFile);
      created = await api("/api/jobs/upload", { method: "POST", body });
    }
    showFormError("");
    onCreated(created);
  });
});

// last: follow the final job of a chain (e.g. textures → rig) instead of the first one
function onCreated(created, { last = false } = {}) {
  const jobs = [].concat(created);
  toast(jobs.length > 1 ? t("{n} trabajos en cola", { n: jobs.length }) : t("Trabajo en cola"));
  state.selected = jobs[last ? jobs.length - 1 : 0].id;
  refreshJobs();
}

// ---------------- concepts (text -> image, pick, then 3D)

async function loadConcepts() {
  const prompt = $("#prompt").value.trim();
  if (prompt.length < 3) return showFormError(t("Escribe una descripción de al menos 3 caracteres"));
  const opts = generationOptions();
  state.conceptPick = null;
  $("#concepts").hidden = false;
  $("#btnConceptTo3d").disabled = true;
  $("#conceptGrid").innerHTML = '<div class="ph"></div>'.repeat(4);
  try {
    const res = await api("/api/concepts", {
      method: "POST",
      json: { prompt, style: opts.style, symmetry: opts.symmetry, pose: opts.pose, count: 4, ...(opts.seed !== undefined ? { seed: opts.seed } : {}) },
    });
    state.concepts = res.concepts;
    $("#conceptGrid").replaceChildren(
      ...res.concepts.map((c) => {
        const img = Object.assign(document.createElement("img"), { src: c.url, alt: t("Concepto") });
        img.onclick = () => {
          state.conceptPick = c.id;
          $$("#conceptGrid img").forEach((x) => x.classList.toggle("on", x === img));
          $("#btnConceptTo3d").disabled = false;
        };
        img.ondblclick = () => (img.onclick(), $("#btnConceptTo3d").click());
        return img;
      }),
    );
    showFormError("");
  } catch (err) {
    $("#concepts").hidden = true;
    showFormError(err.message);
  }
}
$("#btnConcepts").addEventListener("click", () => withBusy($("#btnConcepts"), loadConcepts));
$("#btnConceptsMore").addEventListener("click", () => ($("#seed").value = "", loadConcepts()));
$("#btnConceptTo3d").addEventListener("click", () =>
  withBusy($("#btnConceptTo3d"), async () => {
    const opts = generationOptions();
    const body = imageForm(opts, $("#prompt").value.trim());
    body.append("concept_id", state.conceptPick);
    onCreated(await api("/api/jobs/image", { method: "POST", body }));
  }),
);

// ================================================================ gallery

function matchesFilter(job) {
  const f = state.filter;
  if (f === "fav" && !job.favorite) return false;
  if (f === "gen" && !GENERATIVE.includes(job.mode)) return false;
  if (f === "tools" && !TOOL_MODES.includes(job.mode)) return false;
  if (f === "active" && !["queued", "running"].includes(job.status)) return false;
  if (f === "rig" && job.mode !== "rig") return false;
  return !state.search || title(job.title).toLowerCase().includes(state.search);
}

function renderJobs() {
  const list = $("#jobs");
  const visible = state.jobs.filter(matchesFilter);
  $("#jobsEmpty").hidden = visible.length > 0;
  $("#jobsEmpty").textContent = t(state.jobs.length ? "Nada coincide con el filtro." : "Aún no hay modelos.");
  list.replaceChildren(
    ...visible.map((job) => {
      const li = $("#jobTpl").content.firstElementChild.cloneNode(true);
      li.classList.add(job.status);
      li.classList.toggle("selected", job.id === state.selected);
      li.title = title(job.title);
      const thumb = li.querySelector(".thumb");
      const pic = job.thumbnail_url || job.image_url;
      if (pic) thumb.style.backgroundImage = `url("${pic}")`;
      else thumb.insertAdjacentText("afterbegin", MODE_ICON[job.mode] || "◆");
      li.querySelector(".badge").textContent =
        job.status === "running" ? `${job.progress}%` : t(STATUS_LABEL[job.status] || MODE_LABEL[job.mode] || job.mode);
      li.querySelector(".name").textContent = title(job.title);
      li.querySelector(".bar span").style.width = `${job.progress}%`;
      const star = li.querySelector(".star");
      star.classList.toggle("on", job.favorite);
      star.addEventListener("click", (e) => (e.stopPropagation(), toggleFavorite(job)));
      li.addEventListener("click", () => selectJob(job.id));
      return li;
    }),
  );
}

$$("#filters button").forEach((b) =>
  b.addEventListener("click", () => {
    state.filter = b.dataset.f;
    $$("#filters button").forEach((x) => x.classList.toggle("on", x === b));
    renderJobs();
  }),
);
$("#search").addEventListener("input", (e) => ((state.search = e.target.value.trim().toLowerCase()), renderJobs()));

async function refreshJobs() {
  try {
    state.jobs = await api("/api/jobs");
  } catch {
    return schedulePoll(5000);
  }
  renderJobs();
  const job = selectedJob();
  if (job) {
    showModel(job);
    renderTools(job);
  } else if (state.selected) {
    selectJob(null);
  }
  if (state.jobs.some((j) => j.status === "queued" || j.status === "running")) schedulePoll(1200);
}

function schedulePoll(ms) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(refreshJobs, ms);
}

function selectJob(id) {
  state.selected = id;
  renderJobs();
  const job = selectedJob();
  showModel(job);
  renderTools(job);
}

async function toggleFavorite(job) {
  await api(`/api/jobs/${job.id}`, { method: "PATCH", json: { favorite: !job.favorite } });
  refreshJobs();
}

// ================================================================ viewer

async function showModel(job) {
  const loading = $("#viewerLoading");
  const empty = $("#viewerEmpty");
  $("#viewerTitle").textContent = job ? title(job.title) : "";
  const emptyMsg = (msg) => {
    $("#viewerEmptyMsg").textContent = msg;
    $("#hero").hidden = !!msg;
  };

  if (!job || job.status !== "succeeded") {
    viewer.clear();
    state.loadedModel = null;
    $("#stats").textContent = "";
    $("#animBar").hidden = true;
    $("#mapView").hidden = true;
    const failed = job?.status === "failed";
    $("#btnDiagFailed").hidden = !failed;
    empty.hidden = !!job && !failed;
    emptyMsg(failed ? t("La generación falló: {e}", { e: tr(job.error) }) : "");
    loading.hidden = !job || failed;
    if (job && !failed) $("#viewerProgress").textContent = `${tr(job.stage) || t("En cola")} · ${job.progress}%`;
    return;
  }

  $("#btnDiagFailed").hidden = true;
  if (state.loadedModel === job.model_url) return;
  state.loadedModel = job.model_url;
  empty.hidden = true;
  loading.hidden = false;
  $("#viewerProgress").textContent = t("Cargando modelo…");
  try {
    const s = await viewer.load(job.model_url);
    if (state.loadedModel !== job.model_url) return; // another model was selected meanwhile
    const dims = s.size ? ` · ${s.size.x.toFixed(2)} × ${s.size.y.toFixed(2)} × ${s.size.z.toFixed(2)} u` : "";
    const res = textureLabel(viewer.textureSize());
    $("#stats").textContent = t("{t} triángulos · {v} vértices · {m} malla(s)",
      { t: s.triangles.toLocaleString(), v: s.vertices.toLocaleString(), m: s.meshes }) + dims +
      (res ? ` · ${t("Textura {r}", { r: res })}` : "");
    $("#animBar").hidden = !s.animations.length;
    $("#btnAnimPlay").textContent = "❚❚";
    $("#animClip").replaceChildren(...s.animations.map((n, i) => new Option(clipLabel(n), i)));
    const maps = viewer.availableMaps();
    $("#mapView").hidden = !maps.length;
    $("#mapView").value = "";
    [...$("#mapView").options].forEach((o) => (o.hidden = !!o.value && !maps.includes(o.value)));
    updateEnhance();
    renderFlow(job);
    // auto=1: the server's quick preview of a variant, replaced by the viewer's own picture
    if (!job.thumbnail_url || job.thumbnail_url.includes("auto=1")) setTimeout(() => uploadThumbnail(job.id), 700);
  } catch (err) {
    state.loadedModel = null;
    emptyMsg(t("No se pudo cargar el modelo: {e}", { e: err.message }));
    empty.hidden = false;
  } finally {
    loading.hidden = true;
  }
}

async function uploadThumbnail(jobId) {
  if (state.selected !== jobId) return;
  const blob = await viewer.snapshot({ size: 256, type: "image/jpeg", hideGrid: true });
  if (!blob) return;
  try {
    await fetch(`/api/jobs/${jobId}/thumbnail`, { method: "PUT", body: blob, headers: { "Content-Type": "image/jpeg" } });
    const job = state.jobs.find((j) => j.id === jobId);
    if (job) job.thumbnail_url = URL.createObjectURL(blob);
    renderJobs();
  } catch {}
}

function toggle(button, fn) {
  button.addEventListener("click", () => {
    const on = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(on));
    fn(on);
  });
}
toggle($("#btnRotate"), (on) => viewer.setAutoRotate(on));
toggle($("#btnGrid"), (on) => viewer.setGrid(on));
$("#btnReset").addEventListener("click", () => viewer.frame());
$$("#viewModes button").forEach((b) =>
  b.addEventListener("click", () => {
    viewer.setViewMode(b.dataset.view);
    $$("#viewModes button").forEach((x) => x.classList.toggle("on", x === b));
  }),
);
$("#lighting").addEventListener("change", (e) => viewer.setLighting(e.target.value));
$("#mapView").addEventListener("change", (e) => viewer.setMapView(e.target.value));

function textureLabel(px) {
  if (!px) return "";
  return px >= 8192 ? "8K" : px >= 4096 ? "4K" : px >= 2048 ? "2K" : px >= 1024 ? "1K" : `${px} px`;
}
$("#animClip").addEventListener("change", (e) => viewer.playClip(Number(e.target.value)));
$("#btnAnimPlay").addEventListener("click", () => ($("#btnAnimPlay").textContent = viewer.toggleAnimation() ? "❚❚" : "▶"));
$("#btnShot").addEventListener("click", async () => {
  if (!viewer.model) return toast(t("Carga un modelo primero"), true);
  downloadBlob(await viewer.snapshot(), `${slug(selectedJob()?.title)}.png`);
});
$("#btnVideo").addEventListener("click", async () => {
  const btn = $("#btnVideo");
  if (!viewer.model) return toast(t("Carga un modelo primero"), true);
  btn.disabled = true;
  btn.classList.add("recording");
  btn.textContent = t("● Grabando…");
  try {
    downloadBlob(await viewer.recordTurntable(6), `${slug(selectedJob()?.title)}-360.webm`);
  } catch (err) {
    toast(t(err.message), true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("recording");
    btn.textContent = "▶ 360°";
  }
});

// ================================================================ workflow bar: 01 input → 02 textured model → 03 rig → 04 Blender

function lineage(job) {
  const byId = Object.fromEntries(state.jobs.map((j) => [j.id, j]));
  const chain = [];
  for (let cur = job; cur && !chain.includes(cur); cur = byId[cur.parent_id]) chain.unshift(cur);
  return chain;
}

function renderFlow(job) {
  const steps = $$("#flow li");
  const set = (i, cls, note = "") => {
    steps[i].className = cls;
    steps[i].querySelector("small").textContent = note;
  };
  if (!job) {
    set(0, "next", t("siguiente paso →"));
    [1, 2, 3].forEach((i) => set(i, ""));
    return;
  }
  const chain = lineage(job);
  const ready = job.status === "succeeded";
  const rig = chain.find((j) => j.mode === "rig" && j.status === "succeeded");
  const hd = chain.find((j) => j.mode === "enhance" && j.status === "succeeded");
  set(0, "done", t(MODE_LABEL[chain[0].mode] || chain[0].mode));
  let textures = t("pendiente");
  if (job.status === "failed") textures = t("falló");
  else if (!ready) textures = t("generando…");
  else if (hd) textures = t("PBR {r}", { r: (hd.params?.target || "").toUpperCase() });
  else if (viewer.model && state.loadedModel === job.model_url) {
    const res = textureLabel(viewer.textureSize());
    textures = res ? t("texturas {r}", { r: res }) : t("colores por vértice");
  }
  set(1, ready ? "done" : "next", textures);
  set(2, rig ? "done" : ready ? "next" : "", rig ? t("{n} animaciones", { n: (rig.params?.clips || []).length })
    : ready ? t("siguiente paso →") : "");
  set(3, rig ? "next" : "", ready ? t("GLB · FBX · .blend") : "");
}

function openSection(section) {
  section.hidden = false;
  section.open = true;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

$$("#flow li").forEach((li) =>
  li.addEventListener("click", () => {
    const step = Number(li.dataset.step);
    if (step === 1 || !selectedJob()) {
      $(".create").scrollIntoView({ behavior: "smooth" });
      return $("#prompt").focus();
    }
    if (step === 2) return openSection($("#enhanceSection").hidden ? $("#rigSection") : $("#enhanceSection"));
    if (step === 3) return openSection($("#rigSection"));
    openSection($("#exportSection"));
    chooseTarget("blender");
  }),
);

// ================================================================ tools panel

function metaChips(job) {
  const p = job.params || {};
  const chips = [`<span>${t(MODE_LABEL[job.mode] || job.mode)}</span>`];
  if (GENERATIVE.includes(job.mode)) {
    chips.push(`<span>${t(QUALITY_LABEL[job.quality] || job.quality)}</span>`);
    const style = state.config.styles.find((s) => s.id === p.style);
    if (style && p.style !== "auto") chips.push(`<span>${tr(style.label)}</span>`);
    if (p.seed !== undefined) chips.push(`<span title="${t("Semilla")}">#${p.seed}</span>`);
    if (p.refined) chips.push("<span>HD</span>");
    const engine = (state.config.engines || []).find((e) => e.id === p.engine);
    if (engine) chips.push(`<span title="${t("Motor de IA usado")}">${escapeHtml(tr(engine.label))}</span>`);
  }
  if (job.mode === "enhance") {
    chips.push(`<span>${(p.target || "").toUpperCase()}${p.pbr ? " · PBR" : ""}</span>`);
    if (p.method) chips.push(`<span>${escapeHtml(tr(p.method))}</span>`);
    if (p.sizes?.length) chips.push(`<span>${p.sizes.map((s) => s.join("×")).slice(0, 3).join(", ")} px</span>`);
  }
  if (job.mode === "remesh") chips.push(`<span>${p.mode}</span>`);
  if (job.mode === "retexture" && p.resolution) chips.push(`<span>${p.resolution.toUpperCase()}</span>`);
  if (job.depth_url) chips.push(`<span><a href="${job.depth_url}" target="_blank" rel="noopener">${t("mapa de profundidad")}</a>${p.depth_source === "ia" ? " · IA" : ""}</span>`);
  if (job.parent_id) {
    const parent = state.jobs.find((j) => j.id === job.parent_id);
    chips.push(parent ? `<span>${t("de")} <a data-goto="${parent.id}">${escapeHtml(title(parent.title)).slice(0, 28)}</a></span>`
      : `<span>${t("de (eliminado)")}</span>`);
  }
  chips.push(`<span>${new Date(job.created_at * 1000).toLocaleString(lang)}</span>`);
  if (job.status === "failed") chips.push(`<span class="err">${escapeHtml(tr(job.error) || t("Error"))}</span>`);
  else if (job.status !== "succeeded") chips.push(`<span>${escapeHtml(tr(job.stage) || t("En cola"))} · ${job.progress}%</span>`);
  return chips.join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderTools(job) {
  $("#toolsEmpty").hidden = !!job;
  $("#toolsBody").hidden = !job;
  if (!job) {
    renderRig(null);
    renderFlow(null);
    return (state.toolsKey = null);
  }

  $("#toolTitle").textContent = title(job.title);
  $("#btnFav").textContent = job.favorite ? "★" : "☆";
  $("#btnFav").classList.toggle("on", job.favorite);
  $("#toolMeta").innerHTML = metaChips(job);
  $$("#toolMeta [data-goto]").forEach((a) => (a.onclick = () => selectJob(a.dataset.goto)));
  renderVariants(job);
  renderHistory(job);
  renderFlow(job);

  const key = `${job.id}|${job.status}`;
  if (state.toolsKey === key) return;
  state.toolsKey = key;

  const ready = job.status === "succeeded";
  $("#tools").classList.toggle("disabled-tools", !ready);
  $$('#toolsBody details[data-need="generative"]').forEach((d) => (d.hidden = !GENERATIVE.includes(job.mode)));
  $("#enhanceSection").hidden = job.mode === "rig";
  const flat = FLAT_MODES.includes(job.mode) || (job.mode === "text3d" && job.params?.style !== "letters");
  $("#exportFit").value = flat ? "max" : "height";
  updateExportLink();
  renderRig(job);
  const share = `${location.origin}/share.html?id=${job.id}`;
  $("#btnOpenShare").href = share;
  $("#embedCode").value = `<iframe src="${share}&embed=1" width="640" height="480" style="border:0;border-radius:12px" allow="xr-spatial-tracking; fullscreen" loading="lazy"></iframe>`;
  $("#printReport").innerHTML = `<span class="muted">${t("Abre esta sección para analizar el modelo.")}</span>`;
  if (ready && $("#printReport").closest("details").open) loadAnalysis(job);
}

async function loadAnalysis(job) {
  const el = $("#printReport");
  const key = `${job.id}|${$("#printSize").value}|${$("#printer").value}`;
  if (!state.analysis[key]) {
    el.textContent = t("Analizando…");
    try {
      const params = new URLSearchParams({ size_mm: $("#printSize").value || 100, fit: "height", bed: $("#printer").value || "bambu_a1" });
      state.analysis[key] = await api(`/api/jobs/${job.id}/analysis?${params}`);
    } catch (err) {
      el.textContent = err.message;
      return;
    }
  }
  if (state.selected !== job.id) return;
  const a = state.analysis[key];
  const p = a.print;
  const yes = (v) => (v ? `<b class="ok">${t("Sí")}</b>` : `<b class="bad">${t("No")}</b>`);
  const num = (v, good) => `<b class="${good ? "ok" : "bad"}">${v}</b>`;
  el.innerHTML = `
    <div class="${a.printable ? "ok" : "bad"} report-title">${t(a.printable ? "✔ Listo para imprimir" : "⚠ Necesita reparación")}</div>
    <dl class="print-grid">
      <dt>${t("Estanco")}</dt><dd>${yes(p.watertight)}</dd>
      <dt>${t("Volumen")}</dt><dd>${p.volume_cm3 != null ? `${p.volume_cm3.toLocaleString(lang)} cm³` : "—"}</dd>
      <dt>${t("Peso aprox. (PLA)")}</dt><dd>${p.weight_g != null ? `${p.weight_g.toLocaleString(lang)} g` : "—"}</dd>
      <dt>${t("Huecos")}</dt><dd>${num(p.holes, !p.holes)}</dd>
      <dt>${t("Bordes no manifold")}</dt><dd>${num(p.non_manifold_edges, !p.non_manifold_edges)}</dd>
      <dt>${t("Tamaño")}</dt><dd>${p.size_mm.join(" × ")} mm</dd>
      <dt>${t("Cabe en la cama")}</dt><dd>${yes(p.fits_bed)} <small class="muted">(${p.bed_mm.join("×")})</small></dd>
    </dl>
    <div class="muted small">${t("{f} triángulos · {b} pieza(s)", { f: a.faces.toLocaleString(lang), b: a.bodies })}</div>
    ${a.issues.length ? `<ul>${a.issues.map((i) => `<li>${escapeHtml(tr(i))}</li>`).join("")}</ul>` : ""}`;
  $("#btnFitBed").classList.toggle("attention", !p.fits_bed);
}
$("#printReport").closest("details").addEventListener("toggle", (e) => {
  const job = selectedJob();
  if (e.target.open && job?.status === "succeeded") loadAnalysis(job);
});

// shape variants of one input, side by side: open each one and keep the best
function renderVariants(job) {
  const group = job.params?.group;
  const items = group ? state.jobs.filter((j) => j.params?.group === group)
    .sort((a, b) => a.created_at - b.created_at || (a.params.variant || 0) - (b.params.variant || 0)) : [];
  $("#variantsBox").hidden = items.length < 2;
  if (items.length < 2) return;
  const ready = items.filter((j) => j.status === "succeeded").length;
  $("#variantsHint").textContent = ready < items.length
    ? t("{d} de {n} listas", { d: ready, n: items.length }) : t("Compara y quédate con una");
  $("#variantGrid").replaceChildren(
    ...items.map((j, i) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = `variant ${j.status}`;
      card.classList.toggle("on", j.id === job.id);
      const pic = document.createElement("span");
      pic.className = "pic";
      if (j.thumbnail_url) pic.style.backgroundImage = `url("${j.thumbnail_url}")`;
      else pic.textContent = j.status === "failed" ? "✕" : j.status === "running" ? `${j.progress}%` : "…";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = tr(j.params.variant_label) || t("Variante {n}", { n: i + 1 });
      card.append(pic, name);
      card.title = j.status === "failed" ? tr(j.error) || "" : name.textContent;
      card.addEventListener("click", () => selectJob(j.id));
      return card;
    }),
  );
  $("#btnKeep").disabled = job.status !== "succeeded";
}

$("#btnKeep").addEventListener("click", async (e) => {
  const job = selectedJob();
  if (!job?.params?.group) return;
  const others = state.jobs.filter((j) => j.params?.group === job.params.group && j.id !== job.id).length;
  if (!confirm(t("¿Quedarte con esta variante y borrar las otras {n}?", { n: others }))) return;
  const button = e.currentTarget;
  button.disabled = true;
  try {
    await api(`/api/jobs/${job.id}/keep`, { method: "POST" });
    toast(t("Variante elegida"));
    state.toolsKey = null;
    refreshJobs();
  } catch (err) {
    toast(tr(err.message), true);
    button.disabled = false;
  }
});

function renderHistory(job) {
  const byId = Object.fromEntries(state.jobs.map((j) => [j.id, j]));
  const chain = [];
  for (let cur = job; cur; cur = byId[cur.parent_id]) chain.unshift(cur);
  const rows = chain.map((j, depth) => [j, depth]);
  const children = state.jobs.filter((j) => j.parent_id === job.id).reverse();
  children.forEach((c) => rows.push([c, chain.length]));
  $("#historySection").hidden = rows.length < 2;
  $("#history").replaceChildren(
    ...rows.map(([j, depth]) => {
      const li = document.createElement("li");
      li.classList.toggle("current", j.id === job.id);
      li.innerHTML = `<span class="lvl">${depth ? "  ".repeat(depth - 1) + "└ " : ""}</span>${MODE_ICON[j.mode] || "◆"} `;
      li.append(document.createTextNode(title(j.title)));
      if (j.id !== job.id) li.onclick = () => selectJob(j.id);
      return li;
    }),
  );
}

function updateExportLink() {
  const job = selectedJob();
  if (!job) return;
  const fmt = $("#exportFormat").value;
  const size = Number($("#exportHeight").value);
  const fit = $("#exportFit").value;
  const isPrint = ["stl", "3mf", ...COLOR_FORMATS].includes(fmt);
  const colorFmt = COLOR_FORMATS.includes(fmt);
  $("#exportColorsRow").hidden = !colorFmt;
  $("#exportColorsValue").textContent = $("#exportColors").value;
  const fitLabel = $("#exportFit").selectedOptions[0].textContent.toLowerCase();
  const rigNote = job.mode === "rig" && !["glb", "fbx", "blend"].includes(fmt) ? t(" (malla estática: sin esqueleto ni animaciones)") : "";
  const target = TARGETS.find((x) => x.id === state.target);
  $("#exportHint").textContent = (target ? t(target.hint(state.config.export_formats)) : t(FORMAT_INFO[fmt]?.[1] || "")) + rigNote +
    (isPrint && !size ? t(" {f} por defecto: 100 mm.", { f: fitLabel[0].toUpperCase() + fitLabel.slice(1) }) : "");
  const client = CLIENT_FORMATS.includes(fmt);
  $("#exportFit").disabled = $("#exportHeight").disabled = client;
  $("#btnExport").href = client ? "#" : `/api/jobs/${job.id}/export?format=${fmt}${size > 0 ? `&size_mm=${size}&fit=${fit}` : `&fit=${fit}`}${colorFmt ? `&colors=${$("#exportColors").value}` : ""}`;
}
function chooseTarget(id) {
  const target = TARGETS.find((x) => x.id === id);
  state.target = target ? id : null;
  $$("#targets button").forEach((b) => b.classList.toggle("on", b.dataset.target === state.target));
  if (target) $("#exportFormat").value = target.pick(state.config.export_formats);
  updateExportLink();
}
$("#exportFormat").addEventListener("change", () => chooseTarget(null));
$("#exportFit").addEventListener("change", updateExportLink);
$("#exportHeight").addEventListener("input", updateExportLink);
$("#exportColors").addEventListener("input", updateExportLink);
$("#btnExport").addEventListener("click", async (e) => {
  if (!CLIENT_FORMATS.includes($("#exportFormat").value)) return;
  e.preventDefault();
  if (!viewer.model) return toast(t("Espera a que el modelo cargue en el visor"), true);
  try {
    downloadBlob(await viewer.exportUSDZ(), `${slug(selectedJob()?.title)}.usdz`);
  } catch (err) {
    toast(t("No se pudo crear el USDZ: {e}", { e: err.message }), true);
  }
});

// remesh controls: faces slider is logarithmic (200 .. 200k)
const facesFromSlider = () => Math.round(10 ** Number($("#remeshFaces").value) / 100) * 100;
function updateRemeshUi() {
  const mode = $("#remeshMode").value;
  $("#facesRow").hidden = !["decimate", "lowpoly", "quads"].includes(mode);
  $("#quadsBox").hidden = mode !== "quads";
  $$("#quadPresets button").forEach((b) => b.classList.toggle("on", b.dataset.faces === $("#remeshFaces").value));
  $("#resRow").hidden = mode !== "voxel";
  $("#facesValue").textContent = facesFromSlider().toLocaleString();
  $("#resValue").textContent = `${$("#remeshRes").value}³`;
}
["remeshMode", "remeshFaces", "remeshRes"].forEach((id) => $(`#${id}`).addEventListener("input", updateRemeshUi));
$$("#quadPresets button").forEach((b) =>
  b.addEventListener("click", () => {
    $("#remeshFaces").value = b.dataset.faces;
    updateRemeshUi();
  }),
);

async function derive(button, path, json) {
  const job = selectedJob();
  if (!job) return;
  button.disabled = true;
  try {
    onCreated(await api(`/api/jobs/${job.id}/${path}`, { method: "POST", json }));
  } catch (err) {
    toast(err.message, true);
  } finally {
    button.disabled = false;
  }
}

$("#btnRefine").addEventListener("click", (e) => derive(e.currentTarget, "refine"));
$("#btnVariations").addEventListener("click", (e) => derive(e.currentTarget, "variations", { count: 3 }));
$("#btnRemesh").addEventListener("click", (e) =>
  derive(e.currentTarget, "remesh", {
    mode: $("#remeshMode").value,
    target_faces: facesFromSlider(),
    resolution: Number($("#remeshRes").value),
  }),
);
$("#btnRetexture").addEventListener("click", (e) => {
  const prompt = $("#retexPrompt").value.trim();
  if (prompt.length < 3) return toast(t("Describe la textura (mín. 3 caracteres)"), true);
  derive(e.currentTarget, "retexture", { prompt, style: $("#retexStyle").value, resolution: $("#retexRes").value });
});
$("#btnRepair").addEventListener("click", (e) => derive(e.currentTarget, "repair"));
function updateEnhance() {
  const ok = !viewer.model || viewer.hasTextures();
  $("#btnEnhance").disabled = !ok;
  $("#btnEnhance").title = ok ? "" : t("Este modelo usa colores por vértice: retexturízalo con IA primero");
  $("#enhanceHint").textContent = ok ? state.enhanceHint
    : t("Este modelo no tiene texturas (colores por vértice). Usa «Retexturizado IA» o un motor con PBR para obtener texturas HD.");
}
$("#btnEnhance").addEventListener("click", (e) =>
  derive(e.currentTarget, "enhance", { target: $("#enhanceTarget .on").dataset.v, pbr: $("#enhancePbr").checked }),
);
async function setupPrinting() {
  let info = { beds: [], slicers: [] };
  try {
    info = await api("/api/printers");
  } catch {}
  $("#printer").replaceChildren(...info.beds.map((b) => new Option(`${b.label} · ${b.size_mm.join("×")}`, b.id)));
  try {
    $("#printer").value = localStorage.getItem("forja.printer") || "bambu_a1";
  } catch {}
  $("#slicer").replaceChildren(
    ...info.slicers.map((s) => new Option(s.label, s.id)),
    new Option(t("Programa predeterminado para .3mf"), ""),
  );
  // launching programs only makes sense when the app runs on this computer (desktop app)
  $("#slicerBox").hidden = !["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
}
const reanalyse = () => {
  const job = selectedJob();
  if (job?.status === "succeeded" && $("#printSection").open) loadAnalysis(job);
};
$("#printer").addEventListener("change", () => {
  try {
    localStorage.setItem("forja.printer", $("#printer").value);
  } catch {}
  reanalyse();
});
$("#printSize").addEventListener("change", reanalyse);

const UPLOAD_PAGES = {
  makerworld: "https://makerworld.com/es/my/models/upload",
  printables: "https://www.printables.com/model/create",
  thingiverse: "https://www.thingiverse.com/upload",
};
$("#publishRow").addEventListener("click", (e) => {
  const site = e.target.dataset?.site;
  const job = selectedJob();
  if (!site || !job) return;
  const params = new URLSearchParams({ size_mm: $("#printSize").value || 100, bed: $("#printer").value || "bambu_a1" });
  const link = Object.assign(document.createElement("a"), { href: `/api/jobs/${job.id}/publish?${params}` });
  document.body.append(link);
  link.click();  // the kit downloads while the upload page opens
  link.remove();
  window.open(UPLOAD_PAGES[site], "_blank", "noopener");
  toast(t("Preparando el paquete para {s}: sube los archivos y pega la descripción", { s: e.target.textContent }));
});
$("#btnFitBed").addEventListener("click", (e) =>
  derive(e.currentTarget, "split", { mode: "bed", bed: $("#printer").value, size_mm: Number($("#printSize").value) || 100, fit: "height" }),
);
$("#btnSlicer").addEventListener("click", async (e) => {
  const job = selectedJob();
  if (!job) return;
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const res = await api(`/api/jobs/${job.id}/slicer`, {
      method: "POST",
      json: { slicer: $("#slicer").value || null, format: $("#slicerColor").checked ? "3mf_color" : "3mf",
              size_mm: Number($("#printSize").value) || 100, fit: "height", bed: $("#printer").value },
    });
    toast(t("Abierto en {s}", { s: res.opened_with === "default" ? t("tu laminador") : res.opened_with }));
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
  }
});
$("#btnSplit").addEventListener("click", (e) =>
  derive(e.currentTarget, "split", { mode: $("#splitMode").value, parts: Number($("#splitParts").value) || 2 }),
);

$("#btnRemix").addEventListener("click", () => {
  const job = selectedJob();
  if (!job) return;
  const p = job.params || {};
  setMode("text");
  if (job.prompt) $("#prompt").value = job.prompt;
  setStyle(p.style || "auto");
  $("#quality").value = job.quality;
  $("#symmetry").checked = !!p.symmetry;
  $("#pose").value = p.pose || "none";
  $("#seed").value = p.seed ?? "";
  $(".advanced").open = true;
  toast(t("Parámetros cargados en el formulario"));
});

$("#btnFav").addEventListener("click", () => selectedJob() && toggleFavorite(selectedJob()));
$("#btnRename").addEventListener("click", async () => {
  const job = selectedJob();
  if (!job) return;
  const name = prompt(t("Nuevo nombre:"), job.title);
  if (name === null) return;
  await api(`/api/jobs/${job.id}`, { method: "PATCH", json: { name } });
  refreshJobs();
});
$("#btnDelete").addEventListener("click", async () => {
  const job = selectedJob();
  if (!job || !confirm(t("¿Eliminar \"{t}\"?", { t: title(job.title) }))) return;
  await api(`/api/jobs/${job.id}`, { method: "DELETE" });
  toast(t("Modelo eliminado"));
  selectJob(null);
  refreshJobs();
});
$("#btnMcpCopy").addEventListener("click", async () => {
  let command;
  try {
    command = (await api("/api/integrations/mcp/setup")).claude_code;
    await navigator.clipboard.writeText(command);
    toast(t("Comando copiado: pégalo en tu terminal (necesita uv)"));
  } catch (err) {
    if (!command) return toast(err.message, true);
    prompt(t("Copia el comando:"), command);
  }
});
$("#btnCopyLink").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#btnOpenShare").href);
    toast(t("Enlace copiado"));
  } catch {
    prompt(t("Copia el enlace:"), $("#btnOpenShare").href);
  }
});

// ================================================================ boot

function sakura(count = 14) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const box = $("#petals");
  for (let i = 0; i < count; i++) {
    const petal = document.createElement("i");
    petal.className = "petal";
    const size = 0.6 + Math.random() * 0.8;
    Object.assign(petal.style, {
      left: `${Math.random() * 100}%`,
      transform: `scale(${size})`,
      animationDuration: `${12 + Math.random() * 14}s`,
      animationDelay: `${-Math.random() * 24}s`,
      opacity: String(0.25 + Math.random() * 0.4),
    });
    box.append(petal);
  }
}

(async function boot() {
  sakura();
  const el = $("#provider");
  try {
    state.config = await api("/api/config");
    const gpu = state.config.provider_info?.gpu;
    const engine = (state.config.engines || []).find((e) => e.id === state.config.default_engine);
    const ai = state.config.local_ai || {};
    const localText = gpu ? `${t("IA local gratis")} · ${gpu}`
      : ai.installing ? t("IA local · instalando {p}%", { p: ai.progress })
      : ai.installed ? t("IA local gratis · arranca al generar")
      : t("IA local · instalar (gratis)");
    el.textContent =
      state.config.provider === "mock" ? t("Modo demo (sin IA) · configurar")
      : state.config.provider === "local" ? localText
      : `${t("IA nube")} · ${tr(engine?.label) || state.config.provider}`;
    el.classList.toggle("real", state.config.provider !== "mock");
    el.classList.toggle("bad", !!state.config.provider_error
      || (state.config.provider === "local" && !gpu && !ai.installed && !ai.installing));
    if (state.config.provider_error) el.title = `${t("No se pudo iniciar el motor elegido")}: ${tr(state.config.provider_error)}`;
  } catch {
    el.textContent = t("sin conexión");
    el.classList.add("bad");
  }
  buildForm();
  updateRemeshUi();
  await initRig({ api, viewer, toast, onCreated, selectedJob });
  await refreshJobs();
  const first = state.jobs.find((j) => j.status === "succeeded");
  if (first && !state.selected) selectJob(first.id);
  initUpdates({ api, toast, busy: () => state.jobs.some((j) => j.status === "queued" || j.status === "running") });
  initLocalAI({ api, toast, openSettings: () => openSettings(), config: () => state.config });
  initSettings({ api, toast });
  initAgent({
    api, onCreated, openSettings: () => openSettings(), config: () => state.config,
    selectJob: async (id) => {
      if (!state.jobs.some((j) => j.id === id)) await refreshJobs();
      selectJob(id);
    },
  });
})();
