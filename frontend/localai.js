// Free local AI: shows whether the app's own GPU AI is installed, installs it with one click and
// follows the install (a banner stays visible while it downloads, even with Settings closed).

import { t, tr } from "./i18n.js";
import { checkUpdates } from "./updates.js";

const $ = (sel) => document.querySelector(sel);
let ctx;
let timer = null;
let wasInstalling = false;

export function initLocalAI(context) {
  ctx = context;
  $("#btnInstallAI").addEventListener("click", install);
  $("#btnDiag").addEventListener("click", copyDiagnostics);
  $("#btnDiagFailed").addEventListener("click", copyDiagnostics);
  $("#btnDiagCopy").addEventListener("click", async () => {
    const copied = await copyText($("#diagText").value);
    $("#diagNote").textContent = copied
      ? t("✔ Copiado. Pégalo en tu mensaje (Ctrl+V o mantén pulsado → Pegar).")
      : t("Selecciona todo el texto de abajo (Ctrl+A) y cópialo (Ctrl+C).");
  });
  $("#aiTextures").addEventListener("change", async (e) => {
    try {
      render(await ctx.api(`/api/local-ai/textures?enabled=${e.target.checked}`, { method: "PUT" }));
      ctx.toast(e.target.checked ? t("Texturas con IA activadas: se aplican desde la próxima generación") : t("Texturas con IA desactivadas"));
    } catch (err) {
      e.target.checked = !e.target.checked;
      ctx.toast(err.message, true);
    }
  });
  $("#aiBanner").addEventListener("click", () => ctx.openSettings());
  const s = ctx.config()?.local_ai;
  if (s?.installing) refreshLocalAI();
}

async function install() {
  const button = $("#btnInstallAI");
  button.disabled = true;
  try {
    render(await ctx.api("/api/local-ai/install", { method: "POST" }));
    // the app's own AI is the engine from now on
    if (ctx.config()?.provider !== "local") await ctx.api("/api/settings", { method: "PUT", json: { provider: "local" } });
    refreshLocalAI();
  } catch (err) {
    ctx.toast(err.message, true);
  } finally {
    button.disabled = false;
  }
}

export async function copyDiagnostics() {
  let text;
  try {
    ({ text } = await ctx.api("/api/diagnostics"));
  } catch (err) {
    return ctx.toast(tr(err.message), true);
  }
  // always shown too: whatever happens with the clipboard, the text can be selected and copied by hand
  $("#diagText").value = text;
  $("#diagNote").textContent = t("Copiando…");
  if (!$("#diagDlg").open) $("#diagDlg").showModal();
  const copied = await copyText(text);
  $("#diagNote").textContent = copied
    ? t("✔ Copiado. Pégalo en tu mensaje (Ctrl+V o mantén pulsado → Pegar).")
    : t("Selecciona todo el texto de abajo (Ctrl+A) y cópialo (Ctrl+C).");
  $("#diagText").focus();
  $("#diagText").select();
  if (copied) ctx.toast(t("Diagnóstico copiado: pégalo en tu mensaje"));
}

async function copyText(text) {
  try {  // the app itself (same PC): works even when the window blocks the clipboard
    if ((await ctx.api("/api/clipboard", { method: "POST", json: { text } })).ok) return true;
  } catch {}
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {}
  try {
    $("#diagText").select();
    return document.execCommand("copy");
  } catch {
    return false;
  }
}

let checkedAfterError = false;

export async function refreshLocalAI() {
  clearTimeout(timer);
  let s;
  try {
    s = await ctx.api("/api/local-ai");
  } catch {
    return;
  }
  render(s);
  if (s.error && !checkedAfterError) {
    checkedAfterError = true;
    checkUpdates(true);
  }
  if (s.installing) {
    wasInstalling = true;
    timer = setTimeout(refreshLocalAI, 2000);
  } else if (wasInstalling) {
    wasInstalling = false;
    if (s.error) {
      ctx.toast(t("La instalación de la IA local falló: {e}", { e: tr(s.error) }), true);
      checkUpdates(true); // a newer version may already fix it: show its banner
    } else {
      ctx.toast(t("¡IA local instalada! Ya puedes generar gratis"));
      setTimeout(() => location.reload(), 1500);
    }
  } else if ($("#settingsDlg").open && s.installed && !s.worker_up) {
    timer = setTimeout(refreshLocalAI, 3000); // starting up
  }
}

function render(s) {
  const state = $("#aiState");
  const pct = `${s.progress || 0}%`;
  state.className = "ai-state";
  if (s.installing) {
    state.textContent = `${tr(s.step) || t("Instalando…")} · ${pct}`;
  } else if (s.error) {
    state.textContent = t("La instalación falló: {e}", { e: tr(s.error) });
    state.classList.add("bad");
  } else if (s.installed) {
    state.textContent = s.worker_up ? t("✔ IA local instalada y funcionando") : t("✔ IA local instalada: arranca sola al generar");
    state.classList.add("ok");
  } else if (!s.supported) {
    state.textContent = t("La IA local necesita Windows o Linux con una tarjeta NVIDIA. En Mac usa la nube.");
  } else if (!s.gpu) {
    state.textContent = t("No se detecta una tarjeta NVIDIA: actualiza el driver (nvidia.com/drivers).");
    state.classList.add("bad");
  } else {
    state.textContent = t("La IA local aún no está instalada.");
  }
  $("#aiBar").hidden = !s.installing;
  $("#aiBar span").style.width = pct;
  $("#aiLog").hidden = !s.installing && !s.error;
  $("#aiLog").textContent = (s.log || []).slice(-2).join("\n");
  const button = $("#btnInstallAI");
  button.hidden = s.installing || !s.supported || (s.installed && !s.error);
  button.textContent = s.error ? t("Reintentar instalación") : t("Instalar IA gratis en mi GPU");
  $("#aiTexturesBox").hidden = !s.ai_textures_available;
  $("#aiTextures").checked = !!s.ai_textures;
  $("#aiBanner").hidden = !s.installing;
  $("#aiBannerText").textContent = `${t("Instalando la IA gratis en tu GPU")} — ${tr(s.step) || ""}`;
  $("#aiBannerPct").textContent = `· ${pct}`;
}
