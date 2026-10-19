import { Viewer } from "./viewer.js";

const $ = (sel) => document.querySelector(sel);
const EXAMPLES = [
  "un árbol de pino estilizado",
  "un hongo rojo de cuento",
  "un muñeco de nieve",
  "una roca volcánica",
  "una dona glaseada",
];
const STATUS_LABEL = { queued: "En cola", running: "Generando", succeeded: "Listo", failed: "Error" };

const state = { mode: "text", file: null, jobs: [], selected: null, loadedModel: null, pollTimer: null };
const viewer = new Viewer($("#viewer"));

// ---------- API ----------
async function api(path, options = {}) {
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

// ---------- create form ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    state.mode = tab.dataset.mode;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $(".mode-text").hidden = state.mode !== "text";
    $(".mode-image").hidden = state.mode !== "image";
    showFormError("");
  }),
);

for (const text of EXAMPLES) {
  const chip = Object.assign(document.createElement("button"), { type: "button", className: "chip", textContent: text });
  chip.onclick = () => ($("#prompt").value = text);
  $("#examples").append(chip);
}

function setFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) return showFormError("Usa una imagen PNG, JPG o WEBP");
  state.file = file;
  const preview = $("#preview");
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("#dropText").hidden = true;
  showFormError("");
}

const dropzone = $("#dropzone");
$("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => (e.preventDefault(), dropzone.classList.add("drag"))),
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => (e.preventDefault(), dropzone.classList.remove("drag"))),
);
dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function showFormError(msg) {
  $("#formError").textContent = msg;
  $("#formError").hidden = !msg;
}

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const quality = $("#quality").value;
  const button = $("#submit");
  try {
    let job;
    button.disabled = true;
    if (state.mode === "text") {
      const prompt = $("#prompt").value.trim();
      if (prompt.length < 3) throw new Error("Escribe una descripción de al menos 3 caracteres");
      job = await api("/api/jobs/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, quality }),
      });
    } else {
      if (!state.file) throw new Error("Selecciona una imagen primero");
      const body = new FormData();
      body.append("image", state.file);
      body.append("quality", quality);
      job = await api("/api/jobs/image", { method: "POST", body });
    }
    showFormError("");
    state.selected = job.id;
    await refreshJobs();
  } catch (err) {
    showFormError(err.message);
  } finally {
    button.disabled = false;
  }
});

// ---------- gallery ----------
function jobName(job) {
  return job.prompt || (job.mode === "image" ? "Desde imagen" : "Sin título");
}

function renderJobs() {
  const list = $("#jobs");
  list.replaceChildren();
  $("#jobsEmpty").hidden = state.jobs.length > 0;
  for (const job of state.jobs) {
    const li = $("#jobTpl").content.firstElementChild.cloneNode(true);
    li.classList.add(job.status);
    li.classList.toggle("selected", job.id === state.selected);
    const thumb = li.querySelector(".thumb");
    if (job.image_url) thumb.style.backgroundImage = `url(${job.image_url})`;
    else thumb.textContent = job.mode === "text" ? "✎" : "▣";
    li.querySelector(".name").textContent = jobName(job);
    const meta = li.querySelector(".meta");
    meta.textContent =
      job.status === "failed" ? job.error || "Error"
      : job.status === "running" ? `${job.stage || "Generando"} · ${job.progress}%`
      : `${STATUS_LABEL[job.status]} · ${new Date(job.created_at * 1000).toLocaleString()}`;
    meta.classList.toggle("failed", job.status === "failed");
    li.querySelector(".bar span").style.width = `${job.progress}%`;
    li.addEventListener("click", () => selectJob(job.id));
    li.querySelector(".del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`¿Eliminar "${jobName(job)}"?`)) return;
      await api(`/api/jobs/${job.id}`, { method: "DELETE" });
      if (state.selected === job.id) {
        state.selected = null;
        showModel(null);
      }
      refreshJobs();
    });
    list.append(li);
  }
}

async function refreshJobs() {
  try {
    state.jobs = await api("/api/jobs");
  } catch {
    return schedulePoll(5000);
  }
  renderJobs();
  const selected = state.jobs.find((j) => j.id === state.selected);
  if (selected) showModel(selected);
  if (state.jobs.some((j) => j.status === "queued" || j.status === "running")) schedulePoll(1500);
}

function schedulePoll(ms) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(refreshJobs, ms);
}

function selectJob(id) {
  state.selected = id;
  renderJobs();
  showModel(state.jobs.find((j) => j.id === id));
}

// ---------- viewer ----------
async function showModel(job) {
  const loading = $("#viewerLoading");
  const download = $("#btnDownload");
  $("#viewerTitle").textContent = job ? jobName(job) : "";

  if (!job || job.status !== "succeeded") {
    viewer.clear();
    state.loadedModel = null;
    download.hidden = true;
    $("#stats").textContent = "";
    $("#viewerEmpty").hidden = false;
    $("#viewerEmpty").textContent = !job
      ? "Genera o selecciona un modelo para verlo aquí"
      : job.status === "failed" ? `La generación falló: ${job.error}` : `${job.stage || "En cola"}… ${job.progress}%`;
    loading.hidden = !job || job.status === "failed";
    return;
  }

  loading.hidden = true;
  download.href = job.model_url;
  download.hidden = false;
  if (state.loadedModel === job.id) return;
  state.loadedModel = job.id;
  $("#viewerEmpty").hidden = true;
  loading.hidden = false;
  try {
    const s = await viewer.load(job.model_url);
    $("#stats").textContent =
      `${s.triangles.toLocaleString()} triángulos · ${s.vertices.toLocaleString()} vértices · ${s.meshes} malla(s)`;
  } catch (err) {
    state.loadedModel = null;
    $("#viewerEmpty").textContent = `No se pudo cargar el modelo: ${err.message}`;
    $("#viewerEmpty").hidden = false;
  } finally {
    loading.hidden = true;
  }
}

function toggle(button, fn) {
  button.addEventListener("click", () => {
    const on = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(on));
    fn(on);
  });
}
toggle($("#btnRotate"), (on) => viewer.setAutoRotate(on));
toggle($("#btnWire"), (on) => viewer.setWireframe(on));
$("#btnReset").addEventListener("click", () => viewer.frame());

// ---------- boot ----------
api("/api/health")
  .then(({ provider }) => {
    const el = $("#provider");
    el.textContent = provider === "mock" ? "Modo demo (sin IA)" : `IA: ${provider}`;
    el.classList.toggle("real", provider !== "mock");
  })
  .catch(() => ($("#provider").textContent = "sin conexión"));

refreshJobs().then(() => {
  const first = state.jobs.find((j) => j.status === "succeeded");
  if (first && !state.selected) selectJob(first.id);
});
