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
  $("#btnReport").addEventListener("click", copyDiagnostics);
  $("#btnDiagGithub").addEventListener("click", sendOnGithub);
  $("#btnDiagCopy").addEventListener("click", async () => {
    const copied = await copyText(report());
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

// "Reportar error": what happened (typed by the user) + the diagnostics, to copy or send on GitHub
export async function copyDiagnostics() {
  $("#diagNote").textContent = t("Preparando el diagnóstico…");
  $("#diagText").value = "";
  if (!$("#diagDlg").open) $("#diagDlg").showModal();
  let text;
  try {
    ({ text } = await ctx.api("/api/diagnostics"));
  } catch (err) {
    $("#diagNote").textContent = tr(err.message);
    return;
  }
  $("#diagText").value = text;
  const copied = await copyText(report());
  $("#diagNote").textContent = copied
    ? t("✔ Copiado. Pégalo en tu mensaje (Ctrl+V o mantén pulsado → Pegar).")
    : t("Selecciona todo el texto de abajo (Ctrl+A) y cópialo (Ctrl+C).");
  $("#diagWhat").focus();
}

function report() {
  const what = $("#diagWhat").value.trim();
  return (what ? `${t("Qué pasó")}: ${what}\n\n` : "") + $("#diagText").value;
}

export async function openExternal(url) {
  try {  // the desktop window does not open browser tabs itself: the app asks the PC's browser
    if ((await ctx.api("/api/open", { method: "POST", json: { url } })).ok) return;
  } catch {}
  window.open(url, "_blank", "noopener");
}

async function sendOnGithub() {
  const base = ctx.config()?.report_url;
  if (!base) return;
  const what = $("#diagWhat").value.trim();
  const title = `Error: ${what.split("\n")[0].slice(0, 80) || $("#diagText").value.split("\n").find((l) => l.startsWith("Trabajo fallido") || l.startsWith("Error")) || "Forja3D"}`;
  const body = report().slice(0, 6000);  // links have a length limit; the full text is on the clipboard
  await copyText(report());
  await openExternal(`${base}?title=${encodeURIComponent(title.slice(0, 120))}&body=${encodeURIComponent("```\n" + body + "\n```")}`);
  $("#diagNote").textContent = t("Se abrió GitHub con el reporte escrito: pulsa «Submit new issue». El texto completo también está copiado.");
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
