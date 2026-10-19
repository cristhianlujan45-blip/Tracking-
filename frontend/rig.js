// Rigging & animation panel: auto-rig, Mixamo-style marker editor, animation picker, BVH mocap,
// skeleton and skin-weight inspection.

const $ = (sel) => document.querySelector(sel);
const NOT_RIGGABLE = ["scene", "relief", "lithophane", "text3d", "split"];

let ctx;
let presets = { presets: [], default: [], markers: [] };
let editing = null; // { jobId, yaw, markers }

export async function initRig(context) {
  ctx = context;
  try {
    presets = await ctx.api("/api/rig/presets");
  } catch {
    $("#rigSection").hidden = true;
    return;
  }
  $("#animList").replaceChildren(
    ...presets.presets.map((p) => {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${p.id}" /> `;
      label.append(document.createTextNode(p.label));
      label.title = `${p.duration.toFixed(1)} s${p.loop ? " · bucle" : ""}`;
      return label;
    }),
  );

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

const selectedAnimations = () => [...document.querySelectorAll("#animList input:checked")].map((i) => i.value);

export function renderRig(job) {
  closeEditor();
  const section = $("#rigSection");
  section.hidden = !job || NOT_RIGGABLE.includes(job.mode) || !presets.presets.length;
  if (section.hidden) return;
  const isRig = job.mode === "rig";
  const params = job.params || {};
  const checked = new Set(isRig ? params.animations || [] : presets.default);
  document.querySelectorAll("#animList input").forEach((i) => (i.checked = checked.has(i.value)));

  $("#mocapBox").hidden = !isRig;
  $("#bvhFiles").value = "";
  $("#bvhKeep").replaceChildren(
    ...(params.bvh || []).map((b) => {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${b.name}" checked /> 🎞 `;
      label.append(document.createTextNode(b.name));
      return label;
    }),
  );
  $("#btnAutoRig").textContent = isRig ? "↻ Actualizar animaciones" : "⚡ Auto-rig";
  $("#btnMarkers").textContent = isRig ? "🎯 Reajustar" : "🎯 Marcadores";
  $("#rigHint").textContent = isRig
    ? `Esqueleto de ${Object.keys(params.skeleton || {}).length || 27} huesos · ${(params.clips || []).length} animaciones. ` +
      "Añade movimientos, sube mocap BVH o reajusta las articulaciones."
    : "Personajes humanoides de pie (pose A o T). Esqueleto estándar compatible con Mixamo, Unity, Unreal y Blender.";
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
  const job = ctx.selectedJob();
  if (!job) return;
  const animations = selectedAnimations();
  if (job.mode !== "rig") {
    ctx.onCreated(await ctx.api(`/api/jobs/${job.id}/rig`, { method: "POST", json: { animations } }));
    return;
  }
  const body = new FormData();
  body.append("animations", animations.join(","));
  body.append("keep_bvh", [...document.querySelectorAll("#bvhKeep input:checked")].map((i) => i.value).join(","));
  body.append("in_place", $("#inPlace").checked);
  for (const file of $("#bvhFiles").files) body.append("bvh", file);
  if (!animations.length && !body.getAll("bvh").length && !body.get("keep_bvh")) {
    throw new Error("Elige al menos una animación o sube un BVH");
  }
  ctx.onCreated(await ctx.api(`/api/jobs/${job.id}/animate`, { method: "POST", body }));
}

// fresh: estimate again from the mesh (instead of a rigged character's saved markers)
async function openEditor(yaw, fresh = false) {
  const job = ctx.selectedJob();
  if (!job || job.status !== "succeeded") return;
  if (!ctx.viewer.model) throw new Error("Espera a que el modelo cargue en el visor");
  const json = { auto: fresh, ...(yaw === undefined ? {} : { yaw }) };
  const res = await ctx.api(`/api/jobs/${job.id}/rig/markers`, { method: "POST", json });
  const markers = res.markers;
  editing = { jobId: job.id, yaw: res.yaw, markers };
  const labels = Object.fromEntries(presets.markers.map((m) => [m.id, m.label]));
  ctx.viewer.setWeightBone(null);
  ctx.viewer.enterMarkerEditor({
    markers, labels, yaw: res.view_yaw, symmetric: $("#markerSym").checked,
    onChange: (m) => editing && (editing.markers = m),
  });
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
  const { jobId, yaw, markers } = editing;
  const res = await ctx.api(`/api/jobs/${jobId}/rig`, {
    method: "POST",
    json: { markers, yaw, symmetric: false, animations: selectedAnimations() },
  });
  closeEditor();
  ctx.onCreated(res);
}
