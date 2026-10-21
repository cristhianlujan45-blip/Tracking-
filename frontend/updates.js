// "New version" banner: the Windows app downloads and installs the update by itself and reopens.
// When the app is idle the update starts on its own after a short countdown ("Ahora no" waits).

import { t, tr } from "./i18n.js";

const $ = (sel) => document.querySelector(sel);
const AUTO_DELAY_S = 15;
const POSTPONE_MS = 3 * 60 * 60 * 1000;
let ctx;
let countdown = null;
let postponedUntil = 0;

export async function initUpdates(context) {
  ctx = context;
  $("#btnUpdateLater").addEventListener("click", () => {
    clearTimeout(countdown);
    countdown = null;
    postponedUntil = Date.now() + POSTPONE_MS;
    $("#updateBanner").hidden = true;
  });
  $("#btnUpdate").addEventListener("click", () => install());
  checkUpdates();
  setInterval(checkUpdates, 30 * 60 * 1000); // the app often stays open for days
}

// force: ask GitHub now (e.g. after a failure a newer version may already fix)
export async function checkUpdates(force = false) {
  if (!ctx || $("#btnUpdate").disabled) return; // not started, or already updating
  let s;
  try {
    s = await ctx.api(`/api/update${force ? "?force=true" : ""}`);
  } catch {
    return;
  }
  if (!s.available) return;
  if (!countdown) $("#updateText").textContent = t("Nueva versión {v} disponible", { v: s.latest });
  $("#btnUpdate").hidden = !s.can_install;
  $("#updateLink").hidden = s.can_install;
  $("#updateLink").href = s.url || "#";
  $("#updateLink").textContent = t("Descargar");
  if (Date.now() < postponedUntil) return;
  $("#updateBanner").hidden = false;
  if (s.can_install && !countdown) autoInstall(s.latest);
}

// install by itself once nothing is running (a model being made, the AI being installed)
function autoInstall(version) {
  let left = AUTO_DELAY_S;
  const tick = () => {
    if (ctx.busy?.()) {
      countdown = null;
      $("#updateText").textContent = t("Nueva versión {v}: se instalará sola cuando termine lo que está en curso",
        { v: version });
      setTimeout(checkUpdates, 2 * 60 * 1000);
      return;
    }
    if (left <= 0) {
      countdown = null;
      install(true);
      return;
    }
    $("#updateText").textContent = t("Actualizando a la versión {v} en {s} s (Forja3D se reabrirá sola)",
      { v: version, s: left });
    left -= 1;
    countdown = setTimeout(tick, 1000);
  };
  tick();
}

async function install(auto = false) {
  $("#btnUpdate").disabled = true;
  try {
    await ctx.api("/api/update/install", { method: "POST" });
  } catch (err) {
    $("#btnUpdate").disabled = false;
    if (auto) {  // busy after all (e.g. the AI is installing): try again later, quietly
      $("#updateText").textContent = tr(err.message);
      setTimeout(checkUpdates, 2 * 60 * 1000);
      return;
    }
    return ctx.toast(tr(err.message), true);
  }
  const poll = async () => {
    let s;
    try {
      s = await ctx.api("/api/update");
    } catch {  // the app already quit to install: it reopens by itself
      $("#updateText").textContent = t("Instalando la actualización… Forja3D se abrirá solo");
      return;
    }
    if (s.state === "error") {
      $("#btnUpdate").disabled = false;
      $("#updateText").textContent = t("La actualización falló: {e}", { e: tr(s.error) });
      return;
    }
    $("#updateText").textContent = s.state === "installing"
      ? t("Instalando la actualización… Forja3D se abrirá solo")
      : t("Descargando la versión {v}… {p}%", { v: s.latest, p: s.progress });
    setTimeout(poll, 1000);
  };
  poll();
}
