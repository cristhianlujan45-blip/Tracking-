import { Viewer } from "./viewer.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const EXAMPLES = [
  "un árbol de pino estilizado",
  "un hongo rojo de cuento",
  "un muñeco de nieve con bufanda azul",
  "una roca volcánica",
  "una dona glaseada rosa",
];
const STYLE_ICONS = { auto: "✧", realistic: "◉", cartoon: "☺", handpainted: "✎", lowpoly: "◆", voxel: "▦", sculpture: "◍" };
const MODE_LABEL = {
  text: "Texto", image: "Imagen", multi_image: "Multi-vista", upload: "Subido",
  remesh: "Remallado", retexture: "Retexturizado", repair: "Reparado", split: "Piezas",
};
const MODE_ICON = { text: "✎", image: "▣", multi_image: "▤", upload: "⇪", remesh: "◇", retexture: "✺", repair: "✚", split: "✂" };
const STATUS_LABEL = { queued: "En cola", running: "Generando", succeeded: "", failed: "Error" };
const QUALITY_LABEL = { draft: "Borrador", standard: "Estándar", high: "Alta" };
const GENERATIVE = ["text", "image", "multi_image"];
const FORMAT_INFO = {
  glb: ["GLB", "Universal: web, Unity, Unreal, Godot, Blender, Roblox."],
  obj: ["OBJ (.zip)", "OBJ + MTL + textura en un ZIP."],
  stl: ["STL", "Impresión 3D: Z hacia arriba, en milímetros, apoyado en la cama."],
  "3mf": ["3MF", "Impresión 3D moderna (Bambu Studio, PrusaSlicer, OrcaSlicer, Cura)."],
  ply: ["PLY", "Colores por vértice; útil para MeshLab y escaneo."],
};
const SLOT_LABELS = ["Frente", "Lateral", "Espalda", "Otra vista"];

const state = {
  config: { capabilities: [], styles: [], poses: [], export_formats: [] },
  mode: "text",
  style: "auto",
  variants: 1,
  imageFile: null,
  multiFiles: [null, null, null, null],
  modelFile: null,
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
    throw new Error(msg);
  }
  return resp.status === 204 ? null : resp.json();
}

let toastTimer;
function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
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
    if (file) accept(file) ? onFile(file) : showFormError("Formato de archivo no soportado");
  });
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

function buildForm() {
  for (const text of EXAMPLES) {
    const chip = Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: text });
    chip.onclick = () => ($("#prompt").value = text);
    $("#examples").append(chip);
  }

  for (const { id, label } of state.config.styles) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "style" + (id === state.style ? " on" : "");
    b.dataset.style = id;
    b.innerHTML = `<span class="ico">${STYLE_ICONS[id] || "•"}</span>${label}`;
    b.onclick = () => setStyle(id);
    $("#styles").append(b);
    $("#retexStyle").append(new Option(`Estilo: ${label}`, id));
  }
  for (const { id, label } of state.config.poses) $("#pose").append(new Option(label, id));
  for (const f of state.config.export_formats) $("#exportFormat").append(new Option(FORMAT_INFO[f]?.[0] || f, f));

  SLOT_LABELS.forEach((label, i) => {
    const zone = document.createElement("label");
    zone.className = "dropzone";
    zone.innerHTML = `<input type="file" accept="image/png,image/jpeg,image/webp" hidden /><img alt="" hidden /><span class="hint">${label}</span>`;
    setupDropzone(zone, { accept: isImage, onFile: (f) => ((state.multiFiles[i] = f), previewInto(zone, f)) });
    $("#slots").append(zone);
  });
  setupDropzone($("#dropImage"), { accept: isImage, onFile: (f) => ((state.imageFile = f), previewInto($("#dropImage"), f)) });
  setupDropzone($("#dropModel"), {
    accept: (f) => state.config.upload_types.includes(f.name.split(".").pop().toLowerCase()),
    onFile: (f) => ((state.modelFile = f), previewInto($("#dropModel"), f)),
  });

  // capability gating
  $$("[data-cap]").forEach((el) => {
    if (has(el.dataset.cap)) return;
    if (el.classList.contains("tab")) {
      el.disabled = true;
      el.title = "No disponible con el proveedor actual";
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
  $("#submit").textContent = mode === "upload" ? "Importar modelo" : "Generar modelo 3D";
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
  };
}

function imageForm(opts, prompt) {
  const body = new FormData();
  body.append("quality", opts.quality);
  body.append("style", opts.style);
  body.append("variants", opts.variants);
  if (opts.seed !== undefined) body.append("seed", opts.seed);
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
      if (prompt.length < 3) throw new Error("Escribe una descripción de al menos 3 caracteres");
      created = await api("/api/jobs/text", { method: "POST", json: { prompt, ...opts } });
    } else if (state.mode === "image") {
      if (!state.imageFile) throw new Error("Selecciona una imagen primero");
      const body = imageForm(opts, $("#imagePrompt").value.trim());
      body.append("images", state.imageFile);
      created = await api("/api/jobs/image", { method: "POST", body });
    } else if (state.mode === "multi") {
      const files = state.multiFiles.filter(Boolean);
      if (files.length < 2) throw new Error("Añade al menos 2 vistas");
      const body = imageForm({ ...opts, style: "auto" });
      files.forEach((f) => body.append("images", f));
      created = await api("/api/jobs/image", { method: "POST", body });
    } else {
      if (!state.modelFile) throw new Error("Selecciona un archivo 3D");
      const body = new FormData();
      body.append("model", state.modelFile);
      created = await api("/api/jobs/upload", { method: "POST", body });
    }
    showFormError("");
    onCreated(created);
  });
});

function onCreated(created) {
  const jobs = [].concat(created);
  toast(jobs.length > 1 ? `${jobs.length} trabajos en cola` : "Trabajo en cola");
  state.selected = jobs[0].id;
  refreshJobs();
}

// ---------------- concepts (text -> image, pick, then 3D)

async function loadConcepts() {
  const prompt = $("#prompt").value.trim();
  if (prompt.length < 3) return showFormError("Escribe una descripción de al menos 3 caracteres");
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
        const img = Object.assign(document.createElement("img"), { src: c.url, alt: "Concepto" });
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
  if (f === "tools" && GENERATIVE.includes(job.mode)) return false;
  if (f === "active" && !["queued", "running"].includes(job.status)) return false;
  return !state.search || job.title.toLowerCase().includes(state.search);
}

function renderJobs() {
  const list = $("#jobs");
  const visible = state.jobs.filter(matchesFilter);
  $("#jobsEmpty").hidden = visible.length > 0;
  $("#jobsEmpty").textContent = state.jobs.length ? "Nada coincide con el filtro." : "Aún no hay modelos.";
  list.replaceChildren(
    ...visible.map((job) => {
      const li = $("#jobTpl").content.firstElementChild.cloneNode(true);
      li.classList.add(job.status);
      li.classList.toggle("selected", job.id === state.selected);
      li.title = job.title;
      const thumb = li.querySelector(".thumb");
      const pic = job.thumbnail_url || job.image_url;
      if (pic) thumb.style.backgroundImage = `url("${pic}")`;
      else thumb.insertAdjacentText("afterbegin", MODE_ICON[job.mode] || "◆");
      li.querySelector(".badge").textContent =
        job.status === "running" ? `${job.progress}%` : STATUS_LABEL[job.status] || MODE_LABEL[job.mode] || job.mode;
      li.querySelector(".name").textContent = job.title;
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
  $("#viewerTitle").textContent = job ? job.title : "";

  if (!job || job.status !== "succeeded") {
    viewer.clear();
    state.loadedModel = null;
    $("#stats").textContent = "";
    $("#animBar").hidden = true;
    const failed = job?.status === "failed";
    empty.hidden = !!job && !failed;
    empty.textContent = failed ? `La generación falló: ${job.error}` : "Genera o selecciona un modelo para verlo aquí";
    loading.hidden = !job || failed;
    if (job && !failed) $("#viewerProgress").textContent = `${job.stage || "En cola"} · ${job.progress}%`;
    return;
  }

  if (state.loadedModel === job.model_url) return;
  state.loadedModel = job.model_url;
  empty.hidden = true;
  loading.hidden = false;
  $("#viewerProgress").textContent = "Cargando modelo…";
  try {
    const s = await viewer.load(job.model_url);
    if (state.loadedModel !== job.model_url) return; // another model was selected meanwhile
    const dims = s.size ? ` · ${s.size.x.toFixed(2)} × ${s.size.y.toFixed(2)} × ${s.size.z.toFixed(2)} u` : "";
    $("#stats").textContent = `${s.triangles.toLocaleString()} triángulos · ${s.vertices.toLocaleString()} vértices · ${s.meshes} malla(s)${dims}`;
    $("#animBar").hidden = !s.animations.length;
    $("#animClip").replaceChildren(...s.animations.map((n, i) => new Option(n, i)));
    if (!job.thumbnail_url) setTimeout(() => uploadThumbnail(job.id), 700);
  } catch (err) {
    state.loadedModel = null;
    empty.textContent = `No se pudo cargar el modelo: ${err.message}`;
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
$("#animClip").addEventListener("change", (e) => viewer.playClip(Number(e.target.value)));
$("#btnAnimPlay").addEventListener("click", () => ($("#btnAnimPlay").textContent = viewer.toggleAnimation() ? "⏸" : "▶"));
$("#btnShot").addEventListener("click", async () => {
  if (!viewer.model) return toast("Carga un modelo primero", true);
  downloadBlob(await viewer.snapshot(), `${slug(selectedJob()?.title)}.png`);
});
$("#btnVideo").addEventListener("click", async () => {
  const btn = $("#btnVideo");
  if (!viewer.model) return toast("Carga un modelo primero", true);
  btn.disabled = true;
  btn.classList.add("recording");
  btn.textContent = "● Grabando…";
  try {
    downloadBlob(await viewer.recordTurntable(6), `${slug(selectedJob()?.title)}-360.webm`);
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.classList.remove("recording");
    btn.textContent = "🎬 360°";
  }
});

// ================================================================ tools panel

function metaChips(job) {
  const p = job.params || {};
  const chips = [`<span>${MODE_LABEL[job.mode] || job.mode}</span>`];
  if (GENERATIVE.includes(job.mode)) {
    chips.push(`<span>${QUALITY_LABEL[job.quality] || job.quality}</span>`);
    const style = state.config.styles.find((s) => s.id === p.style);
    if (style && p.style !== "auto") chips.push(`<span>${style.label}</span>`);
    if (p.seed !== undefined) chips.push(`<span title="Semilla">🎲 ${p.seed}</span>`);
    if (p.refined) chips.push("<span>HD</span>");
  }
  if (job.mode === "remesh") chips.push(`<span>${p.mode}</span>`);
  if (job.parent_id) {
    const parent = state.jobs.find((j) => j.id === job.parent_id);
    chips.push(parent ? `<span>de <a data-goto="${parent.id}">${escapeHtml(parent.title).slice(0, 28)}</a></span>` : "<span>de (eliminado)</span>");
  }
  chips.push(`<span>${new Date(job.created_at * 1000).toLocaleString()}</span>`);
  if (job.status === "failed") chips.push(`<span class="err">${escapeHtml(job.error || "Error")}</span>`);
  else if (job.status !== "succeeded") chips.push(`<span>${job.stage || "En cola"} · ${job.progress}%</span>`);
  return chips.join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderTools(job) {
  $("#toolsEmpty").hidden = !!job;
  $("#toolsBody").hidden = !job;
  if (!job) return (state.toolsKey = null);

  $("#toolTitle").textContent = job.title;
  $("#btnFav").textContent = job.favorite ? "★" : "☆";
  $("#btnFav").classList.toggle("on", job.favorite);
  $("#toolMeta").innerHTML = metaChips(job);
  $$("#toolMeta [data-goto]").forEach((a) => (a.onclick = () => selectJob(a.dataset.goto)));

  const key = `${job.id}|${job.status}`;
  if (state.toolsKey === key) return;
  state.toolsKey = key;

  const ready = job.status === "succeeded";
  $("#tools").classList.toggle("disabled-tools", !ready);
  $$('#toolsBody details[data-need="generative"]').forEach((d) => (d.hidden = !GENERATIVE.includes(job.mode)));
  updateExportLink();
  const share = `${location.origin}/share.html?id=${job.id}`;
  $("#btnOpenShare").href = share;
  $("#embedCode").value = `<iframe src="${share}&embed=1" width="640" height="480" style="border:0;border-radius:12px" allow="xr-spatial-tracking; fullscreen" loading="lazy"></iframe>`;
  $("#printReport").innerHTML = '<span class="muted">Abre esta sección para analizar el modelo.</span>';
  if (ready && $("#printReport").closest("details").open) loadAnalysis(job);
}

async function loadAnalysis(job) {
  const el = $("#printReport");
  if (!state.analysis[job.id]) {
    el.textContent = "Analizando…";
    try {
      state.analysis[job.id] = await api(`/api/jobs/${job.id}/analysis`);
    } catch (err) {
      el.textContent = err.message;
      return;
    }
  }
  if (state.selected !== job.id) return;
  const a = state.analysis[job.id];
  const [x, y, z] = a.extents.map((v) => v.toFixed(2));
  el.innerHTML = `
    <div class="${a.printable ? "ok" : "bad"}">${a.printable ? "✔ Listo para imprimir" : "⚠ Necesita reparación"}</div>
    <div class="muted small">${a.faces.toLocaleString()} triángulos · ${a.bodies} pieza(s) · ${x}×${y}×${z} u${
      a.volume ? ` · vol. ${a.volume.toFixed(3)} u³` : ""
    }</div>
    ${a.issues.length ? `<ul>${a.issues.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>` : ""}`;
}
$("#printReport").closest("details").addEventListener("toggle", (e) => {
  const job = selectedJob();
  if (e.target.open && job?.status === "succeeded") loadAnalysis(job);
});

function updateExportLink() {
  const job = selectedJob();
  if (!job) return;
  const fmt = $("#exportFormat").value;
  const h = Number($("#exportHeight").value);
  const isPrint = fmt === "stl" || fmt === "3mf";
  $("#exportHint").textContent = FORMAT_INFO[fmt]?.[1] + (isPrint && !h ? " Alto por defecto: 100 mm." : "");
  $("#btnExport").href = `/api/jobs/${job.id}/export?format=${fmt}${h > 0 ? `&height_mm=${h}` : ""}`;
}
$("#exportFormat").addEventListener("change", updateExportLink);
$("#exportHeight").addEventListener("input", updateExportLink);

// remesh controls: faces slider is logarithmic (200 .. 200k)
const facesFromSlider = () => Math.round(10 ** Number($("#remeshFaces").value) / 100) * 100;
function updateRemeshUi() {
  const mode = $("#remeshMode").value;
  $("#facesRow").hidden = !["decimate", "lowpoly"].includes(mode);
  $("#resRow").hidden = mode !== "voxel";
  $("#facesValue").textContent = facesFromSlider().toLocaleString();
  $("#resValue").textContent = `${$("#remeshRes").value}³`;
}
["remeshMode", "remeshFaces", "remeshRes"].forEach((id) => $(`#${id}`).addEventListener("input", updateRemeshUi));

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
  if (prompt.length < 3) return toast("Describe la textura (mín. 3 caracteres)", true);
  derive(e.currentTarget, "retexture", { prompt, style: $("#retexStyle").value });
});
$("#btnRepair").addEventListener("click", (e) => derive(e.currentTarget, "repair"));
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
  toast("Parámetros cargados en el formulario");
});

$("#btnFav").addEventListener("click", () => selectedJob() && toggleFavorite(selectedJob()));
$("#btnRename").addEventListener("click", async () => {
  const job = selectedJob();
  if (!job) return;
  const name = prompt("Nuevo nombre:", job.title);
  if (name === null) return;
  await api(`/api/jobs/${job.id}`, { method: "PATCH", json: { name } });
  refreshJobs();
});
$("#btnDelete").addEventListener("click", async () => {
  const job = selectedJob();
  if (!job || !confirm(`¿Eliminar "${job.title}"?`)) return;
  await api(`/api/jobs/${job.id}`, { method: "DELETE" });
  toast("Modelo eliminado");
  selectJob(null);
  refreshJobs();
});
$("#btnCopyLink").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("#btnOpenShare").href);
    toast("Enlace copiado");
  } catch {
    prompt("Copia el enlace:", $("#btnOpenShare").href);
  }
});

// ================================================================ boot

(async function boot() {
  try {
    state.config = await api("/api/config");
    const el = $("#provider");
    el.textContent = state.config.provider === "mock" ? "Modo demo (sin IA)" : `IA: ${state.config.provider}`;
    el.classList.toggle("real", state.config.provider !== "mock");
  } catch {
    $("#provider").textContent = "sin conexión";
  }
  buildForm();
  updateRemeshUi();
  await refreshJobs();
  const first = state.jobs.find((j) => j.status === "succeeded");
  if (first && !state.selected) selectJob(first.id);
})();
