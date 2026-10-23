// AI agent chat: a drawer where the user talks about what to make; the backend agent calls the app's own
// tools (concepts, 3D, rig + animations, printability) and returns cards to show here.

import { lang, t, tr } from "./i18n.js";

const $ = (sel) => document.querySelector(sel);
const STATUS = { queued: "en cola", running: "generando", succeeded: "listo", failed: "falló" };

let ctx;
let conversationId = null;
let busy = false;

export function initAgent(context) {
  ctx = context;
  $("#agentBtn").addEventListener("click", () => toggle(true));
  $("#agentClose").addEventListener("click", () => toggle(false));
  $("#agentNew").addEventListener("click", reset);
  $("#agentChips").addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") send(e.target.textContent);
  });
  $("#agentForm").addEventListener("submit", (e) => {
    e.preventDefault();
    send($("#agentInput").value);
  });
  $("#agentInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send($("#agentInput").value);
    }
  });
}

function toggle(open) {
  $("#agentPanel").hidden = !open;
  $("#agentBtn").hidden = open;
  if (open) $("#agentInput").focus();
  document.dispatchEvent(new CustomEvent("agent-toggle", { detail: { open } }));
}

function reset() {
  conversationId = null;
  [...$("#agentLog").children].forEach((el) => el.classList.contains("agent-hello") || el.remove());
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  // plain text with line breaks and **bold** only: never inject HTML from the reply
  for (const [i, line] of text.split("\n").entries()) {
    if (i) el.append(document.createElement("br"));
    line.split(/\*\*(.+?)\*\*/g).forEach((part, j) => {
      if (j % 2) el.append(Object.assign(document.createElement("b"), { textContent: part }));
      else el.append(document.createTextNode(part));
    });
  }
  $("#agentLog").append(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
  return el;
}

async function send(raw) {
  const message = raw.trim();
  if (!message || busy) return;
  busy = true;
  $("#agentSend").disabled = true;
  $("#agentInput").value = "";
  bubble("user", message);
  const thinking = bubble("bot typing", t("Pensando…"));
  try {
    const res = await ctx.api("/api/agent/chat", {
      method: "POST",
      json: { message, conversation_id: conversationId, lang },
    });
    conversationId = res.conversation_id;
    thinking.remove();
    if (res.reply) bubble("bot", res.reply);
    res.artifacts.forEach(renderArtifact);
    const created = res.artifacts.filter((a) => a.type === "job" && a.job.status === "queued").map((a) => a.job);
    if (created.length) ctx.onCreated(created, { last: true });
  } catch (err) {
    thinking.remove();
    bubble("bot error", err.message);
  } finally {
    busy = false;
    $("#agentSend").disabled = false;
  }
}

function renderArtifact(a) {
  const card = document.createElement("div");
  card.className = "msg card";
  if (a.type === "concepts") {
    const grid = document.createElement("div");
    grid.className = "agent-concepts";
    a.concepts.forEach((c, i) => {
      const img = Object.assign(document.createElement("img"), { src: c.url, alt: t("Concepto {n}", { n: i + 1 }) });
      img.title = t("Hacer este concepto en 3D");
      img.addEventListener("click", () => send(t("Haz en 3D el concepto {n}", { n: i + 1 })));
      grid.append(img);
    });
    card.append(grid, Object.assign(document.createElement("small"), { textContent: t("Toca uno para hacerlo en 3D") }));
  } else if (a.type === "job") {
    const job = a.job;
    const name = Object.assign(document.createElement("b"), { textContent: tr(job.title) });
    const status = Object.assign(document.createElement("small"), { textContent: t(STATUS[job.status] || job.status) });
    const open = Object.assign(document.createElement("button"), { type: "button", textContent: t("Ver") });
    open.addEventListener("click", () => ctx.selectJob(job.id));
    card.classList.add("job-card");
    card.append(name, status, open);
  } else return;
  $("#agentLog").append(card);
  card.scrollIntoView({ block: "end", behavior: "smooth" });
}
