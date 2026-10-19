// Settings dialog: pick the AI engine (cloud fal.ai, your own GPU, demo), keys and Blender path.
// Saved on the server (user data folder) so the desktop app remembers them between launches.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

let ctx;
let current = null;
let provider = "mock";

export async function initSettings(context) {
  ctx = context;
  $$("#provCards button").forEach((b) => b.addEventListener("click", () => pick(b.dataset.p)));
  $("#falEngine").addEventListener("change", engineHint);
  $("#btnSaveSettings").addEventListener("click", save);
  $("#btnSettings").addEventListener("click", () => openSettings());
  $("#provider").addEventListener("click", () => openSettings());
  try {
    current = await ctx.api("/api/settings");
  } catch {
    $("#btnSettings").hidden = true; // older server without the settings API
    return;
  }
  if (current.first_run && !current.locked) openSettings(true);
}

export async function openSettings(welcome = false) {
  try {
    current = await ctx.api("/api/settings");
  } catch (err) {
    return ctx.toast(err.message, true);
  }
  $("#welcome").hidden = !welcome;
  $("#settingsLocked").hidden = !current.locked;
  $("#settingsForm").querySelectorAll("input, select, #provCards button, #btnSaveSettings")
    .forEach((el) => (el.disabled = current.locked));
  $("#falEngine").replaceChildren(...current.fal_engines.map((e) => new Option(e.label, e.id)));
  $("#falEngine").value = current.fal_engine;
  $("#falKey").value = "";
  $("#falKey").placeholder = current.fal_key_set
    ? `Clave guardada (${current.fal_key_hint}) — déjalo vacío para mantenerla`
    : "Pega tu clave (fal.ai/dashboard/keys)";
  $("#workerUrl").value = current.local_worker_url || "";
  $("#workerToken").value = "";
  $("#blenderPath").value = current.blender_path || "";
  $("#blenderStatus").textContent = current.blender_found
    ? `✔ Blender encontrado: ${current.blender_found}`
    : "Blender no encontrado: instálalo (gratis, blender.org) para exportar FBX y .blend.";
  showError(current.provider_error ? `El motor guardado no arrancó: ${current.provider_error}` : "");
  pick(welcome ? "fal" : current.provider);
  engineHint();
  const dlg = $("#settingsDlg");
  if (!dlg.open) dlg.showModal();
}

function pick(p) {
  provider = p;
  $$("#provCards button").forEach((b) => b.classList.toggle("on", b.dataset.p === p));
  $("#falGroup").hidden = p !== "fal";
  $("#localGroup").hidden = p !== "local";
}

function engineHint() {
  const engine = current?.fal_engines.find((e) => e.id === $("#falEngine").value);
  $("#falEngineHint").textContent = engine ? engine.description : "";
}

function showError(msg) {
  $("#settingsError").textContent = msg;
  $("#settingsError").hidden = !msg;
}

async function save() {
  const body = { provider, blender_path: $("#blenderPath").value.trim() };
  if (provider === "fal") {
    const key = $("#falKey").value.trim();
    if (!key && !current.fal_key_set) return showError("Pega tu clave de fal.ai para usar la nube");
    body.fal_key = key;
    body.fal_engine = $("#falEngine").value;
  }
  if (provider === "local") {
    body.local_worker_url = $("#workerUrl").value.trim() || "http://127.0.0.1:8001";
    const token = $("#workerToken").value.trim();
    if (token) body.local_worker_token = token;
  }
  const button = $("#btnSaveSettings");
  button.disabled = true;
  try {
    await ctx.api("/api/settings", { method: "PUT", json: body });
    $("#settingsDlg").close();
    ctx.toast("¡Ajustes guardados! 準備完了");
    setTimeout(() => location.reload(), 900); // engines and capabilities may have changed
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
  }
}
