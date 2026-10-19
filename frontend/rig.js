// Rigging & animation panel: body type (humanoid / quadruped), auto-rig, Mixamo-style marker editor
// (front view for people, side view for animals), procedural clips, the bundled mocap library,
// BVH mocap, skeleton and skin-weight inspection.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const NOT_RIGGABLE = ["scene", "relief", "lithophane", "text3d", "split"];
const TEMPLATE_LABEL = { humanoid: "humano", quadruped: "cuadrúpedo" };

let ctx;
let presets = { presets: [], default: [], markers: [], library: [], quadruped: { presets: [], default: [], markers: [] } };
let editing = null; // { jobId, yaw, markers, template }
let template = "auto"; // what the body-type selector shows
let job = null;

export async function initRig(context) {
  ctx = context;
  try {
    presets = await ctx.api("/api/rig/presets");
  } catch {
    $("#rigSection").hidden = true;
    return;
  }
  presets.library ||= [];
  presets.quadruped ||= { presets: [], default: [], markers: [] };

  const cats = [...new Set(presets.library.map((c) => c.category))];
  $("#libCat").append(...cats.map((c) => new Option(c, c)));
  $("#libList").replaceChildren(
    ...cats.flatMap((cat) => {
      const head = document.createElement("h4");
      head.textContent = cat;
      head.dataset.cat = cat;
      const rows = presets.library.filter((c) => c.category === cat).map((c) => {
        const label = document.createElement("label");
        label.dataset.cat = cat;
        label.innerHTML = `<input type="checkbox" value="${c.id}" /> `;
        label.append(document.createTextNode(c.label));
        label.insertAdjacentHTML("beforeend", `<small>${c.seconds.toFixed(1)} s</small>`);
        return label;
      });
      return [head, ...rows];
    }),
  );
  $("#libSearch").addEventListener("input", filterLibrary);
  $("#libCat").addEventListener("change", filterLibrary);
  $("#libList").addEventListener("change", countLibrary);

  $$("#rigTemplate button").forEach((b) => b.addEventListener("click", () => setTemplate(b.dataset.t)));
  $("#btnAutoRig").addEventListener("click", () => run($("#btnAutoRig"), autoRigOrAnimate));
  $("#btnMarkers").addEventListener("click", () => run($("#btnMarkers"), () => openEditor()));
  $("#btnRotateModel").addEventListener("click", () => run($("#btnRotateModel"), () => openEditor((editing.yaw + 90) % 360, true)));
  $("#btnResetMarkers").addEventListener("click", () => run($("#btnResetMarkers"), () => openEditor(editing.yaw, true)));
  $("#btnCancelMarkers").addEventListener("click", closeEditor);
  $("#btnApplyMarkers").addEventListener("click", () => run($("#btnApplyMarkers"), applyMarkers));
  $("#markerSym").addEventListener("change", (e) => ctx.viewer.setMarkerSymmetry(e.target.checked));
  $("#showBones").addEventListener("change", (e) => ctx.viewer.setSkeletonVisible(e.target.checked));
  $("#weightBone").addEventListener("pointerdown", fillBones);
  $("#weightBone").addEventListener("focus", fillBones);
  $("#weightBone").addEventListener("change", (e) => ctx.viewer.setWeightBone(e.target.value));
}

// Spanish label for a baked clip name (procedural presets are stored by id)
export function clipLabel(name) {
  const all = [...presets.presets, ...presets.quadruped.presets];
  return all.find((p) => p.id === name)?.label || name;
}

async function run(button, fn) {
  button.disabled = true;
  try {
    await fn();
  } catch (err) {
    ctx.toast(err.message, true);
  } finally {
    button.disabled = false;
  }
}

const isQuad = () => template === "quadruped";
const selectedAnimations = () =>
  [...$$("#animList input:checked"), ...(isQuad() ? [] : $$("#libList input:checked"))].map((i) => i.value);

function filterLibrary() {
  const q = $("#libSearch").value.trim().toLowerCase();
  const cat = $("#libCat").value;
  const visible = new Set();
  $$("#libList label").forEach((l) => {
    const show = (!cat || l.dataset.cat === cat) && (!q || l.textContent.toLowerCase().includes(q));
    l.hidden = !show;
    if (show) visible.add(l.dataset.cat);
  });
  $$("#libList h4").forEach((h) => (h.hidden = !visible.has(h.dataset.cat)));
}

function countLibrary() {
  const n = $$("#libList input:checked").length;
  $("#libCount").textContent = `${n ? `${n} elegidas · ` : ""}${presets.library.length} clips`;
}

// Fill the clip lists for the current body type, ticking `checked` (or that type's defaults).
function renderLists(checked) {
  const set = isQuad() ? presets.quadruped : presets;
  const ticked = new Set(checked ?? set.default);
  $("#animList").replaceChildren(
    ...set.presets.map((p) => {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${p.id}" ${ticked.has(p.id) ? "checked" : ""} /> `;
      label.append(document.createTextNode(p.label));
      label.title = `${p.duration.toFixed(1)} s${p.loop ? " · bucle" : ""}`;
      return label;
    }),
  );
  $$("#libList input").forEach((i) => (i.checked = ticked.has(i.value)));
  $("#libBox").hidden = isQuad() || !presets.library.length;
  countLibrary();
  $$("#rigTemplate button").forEach((b) => b.classList.toggle("on", b.dataset.t === template));
}

function setTemplate(t) {
  if (t === template) return;
  template = t;
  const isRig = job?.mode === "rig";
  renderLists(isRig && t === (job.params?.template || "humanoid") ? job.params.animations || [] : undefined);
  updateTexts();
}

function updateTexts() {
  const isRig = job?.mode === "rig";
  const params = job?.params || {};
  const rigTpl = params.template || "humanoid";
  const changing = isRig && template !== "auto" && template !== rigTpl;
  $("#btnAutoRig").textContent = !isRig ? "⚡ Auto-rig" : changing ? "⚡ Re-riggear" : "↻ Actualizar animaciones";
  $("#btnMarkers").textContent = isRig ? "◎ Reajustar" : "◎ Marcadores";
  $("#rigHint").textContent = isRig
    ? `Esqueleto ${TEMPLATE_LABEL[rigTpl]} de ${Object.keys(params.skeleton || {}).length || "—"} huesos · ` +
      `${(params.clips || []).length} animaciones. ` +
      (changing ? "Se creará un rig nuevo con el otro tipo de cuerpo." : "Añade movimientos, sube mocap BVH o reajusta las articulaciones.")
    : isQuad()
      ? "Animales de cuatro patas (perro, gato, caballo, dragón…) de perfil, de pie. Andar, trotar, galopar, sentarse…"
      : template === "humanoid"
        ? "Personajes de pie (pose A o T), mirando al frente. Esqueleto Mixamo compatible con Unity, Unreal y Blender."
        : "Detecta si es una persona o un animal de cuatro patas. Los clips que no encajen se ignoran.";
}

export function renderRig(next) {
  closeEditor();
  job = next;
  const section = $("#rigSection");
  section.hidden = !job || NOT_RIGGABLE.includes(job.mode) || !presets.presets.length;
  if (section.hidden) return;
  const isRig = job.mode === "rig";
  const params = job.params || {};
  template = isRig ? params.template || "humanoid" : "auto";
  renderLists(isRig ? params.animations || [] : undefined);
  updateTexts();

  $("#mocapBox").hidden = !isRig || template === "quadruped";
  $("#bvhFiles").value = "";
  $("#bvhKeep").replaceChildren(
    ...(params.bvh || []).map((b) => {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${b.name}" checked /> ▶ `;
      label.append(document.createTextNode(b.name));
      return label;
    }),
  );
  $("#rigView").hidden = !isRig || job.status !== "succeeded";
  $("#showBones").checked = false;
  $("#weightBone").replaceChildren(new Option("Pesos: ninguno", ""));
}

function fillBones() {
  const select = $("#weightBone");
  if (select.options.length > 1) return;
  for (const name of ctx.viewer.boneNames()) select.append(new Option(`Pesos: ${name}`, name));
}

async function autoRigOrAnimate() {
  const current = ctx.selectedJob();
  if (!current) return;
  const animations = selectedAnimations();
  const rigTpl = current.params?.template || "humanoid";
  if (current.mode !== "rig" || (template !== "auto" && template !== rigTpl)) {
    ctx.onCreated(await ctx.api(`/api/jobs/${current.id}/rig`, { method: "POST", json: { animations, template } }));
    return;
  }
  const body = new FormData();
  body.append("animations", animations.join(","));
  body.append("keep_bvh", $$("#bvhKeep input:checked").map((i) => i.value).join(","));
  body.append("in_place", $("#inPlace").checked);
  for (const file of $("#bvhFiles").files) body.append("bvh", file);
  if (!animations.length && !body.getAll("bvh").length && !body.get("keep_bvh")) {
    throw new Error("Elige al menos una animación o sube un BVH");
  }
  ctx.onCreated(await ctx.api(`/api/jobs/${current.id}/animate`, { method: "POST", body }));
}

// fresh: estimate again from the mesh (instead of a rigged character's saved markers)
async function openEditor(yaw, fresh = false) {
  const current = ctx.selectedJob();
  if (!current || current.status !== "succeeded") return;
  if (!ctx.viewer.model) throw new Error("Espera a que el modelo cargue en el visor");
  const json = { auto: fresh, template: editing?.template || template, ...(yaw === undefined ? {} : { yaw }) };
  const res = await ctx.api(`/api/jobs/${current.id}/rig/markers`, { method: "POST", json });
  const tpl = res.template || "humanoid";
  if (tpl !== template && template === "auto") {
    template = tpl; // show the clips that fit what was detected
    renderLists();
    updateTexts();
  }
  editing = { jobId: current.id, yaw: res.yaw, markers: res.markers, template: tpl };
  const side = res.view === "side";
  ctx.viewer.setWeightBone(null);
  ctx.viewer.enterMarkerEditor({
    markers: res.markers, labels: res.labels || Object.fromEntries(presets.markers.map((m) => [m.id, m.label])),
    yaw: res.view_yaw, view: res.view || "front", symmetric: $("#markerSym").checked,
    onChange: (m) => editing && (editing.markers = m),
  });
  $("#markerHelp").textContent = side
    ? "Vista lateral: cabeza a la derecha. Arrastra los puntos al hocico, lomo, cola y patas del lado visible."
    : "Arrastra los puntos a las articulaciones (vista frontal; rueda = zoom)";
  $("#markerSymRow").hidden = side;
  $("#rigEditorBar").hidden = false;
  $("#viewModes").hidden = true;
}

function closeEditor() {
  editing = null;
  ctx?.viewer.exitMarkerEditor();
  $("#rigEditorBar").hidden = true;
  $("#viewModes").hidden = false;
}

async function applyMarkers() {
  if (!editing) return;
  const { jobId, yaw, markers, template: tpl } = editing;
  const res = await ctx.api(`/api/jobs/${jobId}/rig`, {
    method: "POST",
    json: { markers, yaw, template: tpl, symmetric: false, animations: selectedAnimations() },
  });
  closeEditor();
  ctx.onCreated(res);
}
