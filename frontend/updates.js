// "New version" banner: the Windows app downloads and installs the update by itself and reopens.

import { t, tr } from "./i18n.js";

const $ = (sel) => document.querySelector(sel);
let ctx;

export async function initUpdates(context) {
  ctx = context;
  $("#btnUpdateLater").addEventListener("click", () => ($("#updateBanner").hidden = true));
  $("#btnUpdate").addEventListener("click", install);
  let s;
  try {
    s = await ctx.api("/api/update");
  } catch {
    return;
  }
  if (!s.available) return;
  $("#updateText").textContent = t("Nueva versión {v} disponible", { v: s.latest });
  $("#btnUpdate").hidden = !s.can_install;
  $("#updateLink").hidden = s.can_install;
  $("#updateLink").href = s.url || "#";
  $("#updateLink").textContent = t("Descargar");
  $("#updateBanner").hidden = false;
}

async function install() {
  $("#btnUpdate").disabled = true;
  try {
    await ctx.api("/api/update/install", { method: "POST" });
  } catch (err) {
    $("#btnUpdate").disabled = false;
    return ctx.toast(err.message, true);
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
