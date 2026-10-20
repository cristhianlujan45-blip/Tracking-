"""Forja3D's AI agent: a chat that brainstorms with you, answers 3D / printing questions and drives
the app through tools (concepts, 3D generation, rig and animation, printability).

Runs on Claude through the official `anthropic` SDK with a manual tool loop, so the whole
conversation (thinking blocks and tool calls included) is kept and sent back unchanged each turn.
"""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anthropic

MAX_TOOL_ROUNDS = 8
MAX_CONVERSATIONS = 50

SYSTEM = """You are the assistant inside Forja3D, a desktop studio that turns text and images into 3D models \
for games, animation, Blender and 3D printing. You help the user go from an idea to a finished asset.

What you can do with your tools:
- make_concepts: draw 4 concept images for an idea (cheap and quick). Offer this before spending on 3D when \
the idea is still vague, and let the user pick one.
- generate_3d: create a 3D model from a text prompt or from a concept the user picked.
- find_animations and rig_character: give a finished character a skeleton and animations (humans and \
four-legged animals) in one step, with HD textures when the model has them.
- check_print: printability of a model at a given size on a given printer (watertight, holes, volume, \
weight, whether it fits the bed).
- list_models and job_status: see what the user has made and how a job is going.

How to work:
- Reply in the user's language ({language}). Be warm, concrete and brief; this is a chat panel, not a document.
- When the user describes something to make, write a strong English prompt for the 3D engines (subject, \
shape, materials, colours, style) and tell the user in one line what you are creating.
- Characters meant for rigging come out best in an A-pose, standing, facing front, full body.
- Generation takes from seconds to a few minutes; after starting a job say so and do not poll it in a loop.
- Answer questions about 3D, Blender, game engines and 3D printing directly from your knowledge.
- Never invent job ids or results: use what the tools return."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties, "required": required,
                             "additionalProperties": False}}


STYLE = {"type": "string", "enum": ["auto", "realistic", "cartoon", "handpainted", "lowpoly", "voxel", "sculpture"]}

TOOLS = [
    _tool("make_concepts", "Draw 4 concept images for an idea so the user can pick one before making it 3D.",
          {"prompt": {"type": "string", "description": "English description of the object or character"},
           "style": STYLE}, ["prompt", "style"]),
    _tool("generate_3d", "Start generating a 3D model. Use concept_id when the user picked a concept image "
          "(then prompt only names it); otherwise prompt describes the model.",
          {"prompt": {"type": "string"}, "style": STYLE,
           "quality": {"type": "string", "enum": ["draft", "standard", "high"]},
           "concept_id": {"type": ["string", "null"], "description": "id returned by make_concepts, or null"},
           "a_pose": {"type": "boolean", "description": "true for characters that will be rigged"}},
          ["prompt", "style", "quality", "concept_id", "a_pose"]),
    _tool("find_animations", "Search the animation library (procedural clips and 600+ motion-capture clips).",
          {"query": {"type": "string", "description": "words in Spanish or English, e.g. 'dance', 'caminar triste'"},
           "body": {"type": "string", "enum": ["humanoid", "quadruped"]}}, ["query", "body"]),
    _tool("rig_character", "Rig a finished model and bake animations, upgrading its textures to HD first when it "
          "has textures. Animation ids come from find_animations.",
          {"job_id": {"type": "string"}, "animations": {"type": "array", "items": {"type": "string"}},
           "body": {"type": "string", "enum": ["auto", "humanoid", "quadruped"]}},
          ["job_id", "animations", "body"]),
    _tool("check_print", "Printability report of a model printed at size_mm tall on the given printer.",
          {"job_id": {"type": "string"}, "size_mm": {"type": "number"},
           "printer": {"type": "string", "description": "printer id, e.g. bambu_a1, prusa_mk4, ender3"}},
          ["job_id", "size_mm", "printer"]),
    _tool("list_models", "The user's most recent models, newest first.",
          {"limit": {"type": "integer", "description": "1 to 20"}}, ["limit"]),
    _tool("job_status", "Status, progress and type of one model or job.", {"job_id": {"type": "string"}}, ["job_id"]),
]


class AgentError(Exception):
    """User-facing error."""


@dataclass
class Turn:
    reply: str
    conversation_id: str
    artifacts: list[dict] = field(default_factory=list)  # concepts and jobs to show in the chat


ToolFn = Callable[[dict], Awaitable[tuple[Any, list[dict]]]]  # → (result for Claude, artifacts for the UI)


class Agent:
    """One per API key. Conversations live in memory (the newest MAX_CONVERSATIONS are kept)."""

    def __init__(self, api_key: str, model: str = "", client: anthropic.AsyncAnthropic | None = None):
        if not api_key and client is None:
            raise AgentError("Añade tu clave de Anthropic en Ajustes para usar el agente")
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.conversations: dict[str, list[dict]] = {}

    async def _resolve_model(self) -> str:
        """No model configured: the newest active Opus-class model this key can use (else the newest one)."""
        if not self.model:
            ids = [m.id async for m in self.client.models.list(limit=50, lifecycle=["active"])]
            if not ids:
                raise AgentError("Tu clave de Anthropic no tiene modelos disponibles")
            self.model = next((i for i in ids if "opus" in i), ids[0])  # the API lists newest first
        return self.model

    async def chat(self, message: str, conversation_id: str | None, tools: dict[str, ToolFn],
                   language: str = "es") -> Turn:
        cid = conversation_id if conversation_id in self.conversations else uuid.uuid4().hex
        history = self.conversations.setdefault(cid, [])
        start = len(history)
        history.append({"role": "user", "content": message})
        system = SYSTEM.format(language="Spanish" if language == "es" else "English")
        artifacts: list[dict] = []
        try:
            model = await self._resolve_model()
            for _ in range(MAX_TOOL_ROUNDS):
                response = await self.client.beta.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    tools=TOOLS,
                    messages=history,
                    output_config={"effort": "medium"},
                    cache_control={"type": "ephemeral"},
                    # a declined request is retried server-side on a fallback model instead of failing
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                history.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "refusal":
                    del history[start:]  # a refused turn cannot be continued
                    return Turn(_declined(language), cid, artifacts)
                if response.stop_reason != "tool_use":
                    return Turn(_text(response), cid, artifacts)
                results = [await self._run_tool(block, tools, artifacts)
                           for block in response.content if block.type == "tool_use"]
                history.append({"role": "user", "content": results})
            # too many tool rounds: close the turn so the conversation stays valid
            history.append({"role": "assistant", "content": "…"})
            return Turn(_text(response) or "…", cid, artifacts)
        except anthropic.APIError as exc:
            del history[start:]
            raise AgentError(_api_error(exc)) from exc
        finally:
            while len(self.conversations) > MAX_CONVERSATIONS:
                self.conversations.pop(next(iter(self.conversations)))

    async def _run_tool(self, block, tools: dict[str, ToolFn], artifacts: list[dict]) -> dict:
        fn = tools.get(block.name)
        try:
            if fn is None:
                raise AgentError(f"unknown tool {block.name}")
            result, extra = await fn(dict(block.input))
            artifacts.extend(extra)
            content, is_error = json.dumps(result, ensure_ascii=False, default=str), False
        except Exception as exc:  # noqa: BLE001 - every failure goes back to Claude as a tool error
            content, is_error = f"Error: {getattr(exc, 'detail', None) or exc}", True
        return {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}


def _api_error(exc: anthropic.APIError) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "La clave de Anthropic no es válida: revísala en Ajustes"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "Tu clave de Anthropic no tiene permiso para usar el agente"
    if isinstance(exc, anthropic.RateLimitError):
        return "Demasiadas peticiones al agente: espera unos segundos"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Sin conexión con el agente: revisa tu internet"
    if isinstance(exc, anthropic.APIStatusError):
        return f"El agente no respondió ({exc.status_code})"
    return "El agente no respondió"


def fold(text: str) -> str:
    """Lower case without accents, for loose search ("canción" matches "cancion")."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _text(response) -> str:
    return "\n\n".join(b.text for b in response.content if b.type == "text").strip()


def _declined(language: str) -> str:
    return ("No puedo ayudar con eso. ¿Probamos con otra idea?" if language == "es"
            else "I can't help with that. Shall we try another idea?")


# ---------------------------------------------------------------- built-in assistant (no key needed)

_CONCEPT_N = re.compile(r"\b(?:concepto|concept|imagen|image|opcion|option|numero|number)?\s*#?([1-4])\b")
_SIZE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm)\b")
_ANIMATE = ("anim", "rig", "esquelet", "skeleton", "bail", "danc", "camin", "walk", "corr", "run", "salt", "jump",
            "pele", "fight", "mover", "move")
_PRINT = ("imprim", "imprime", "impresion", "print", "3mf", "stl", "laminador", "slicer", "bambu", "prusa", "ender")
_IDEAS = ("idea", "concept", "boceto", "sketch", "dibuj", "draw", "opciones", "options", "inspir")
_LIST = ("mis modelos", "my models", "que modelos", "what models", "estado", "status", "como va", "how is")
_FILLER = re.compile(r"^(?:por favor|please|quiero|quisiera|i want|i'd like|hazme|haz|crea|creame|genera|generame|"
                     r"make|create|generate|dame|give me|me haces|puedes hacer|can you make|un modelo 3d de|"
                     r"a 3d model of|un modelo de|a model of|modelo de|model of)\s+")
_PRINTERS = [("bambu_a1mini", ("a1 ?mini",)), ("bambu_h2d", ("h2d",)), ("bambu_a1", ("a1", "p1s", "p1p", "x1c", "x1", "bambu")),
             ("prusa_mini", ("prusa mini",)), ("prusa_mk4", ("mk4", "mk3", "core one", "prusa")),
             ("creality_k2", ("k2",)), ("creality_k1", ("k1c?",)), ("ender3", ("ender ?3", "ender")),
             ("elegoo_n4", ("neptune", "elegoo")), ("anycubic_kobra3", ("kobra", "anycubic")),
             ("saturn4", ("saturn",)), ("photon_m5", ("photon",))]
# "make me a …": a new object for sure (unlike "quiero sacarlo…", which may be about the last model)
_MAKE = re.compile(r"^(?:por favor |please )?(?:crea|creame|hazme|haz un|haz una|genera|generame|make|create|"
                   r"generate|quiero un|quiero una|i want a|i want an|dame un|dame una|give me a|give me an)\b")
_CHARACTER = ("personaje", "character", "hero", "heroe", "guerrer", "warrior", "chica", "chico", "girl", "boy",
              "robot", "monstruo", "monster", "caballero", "knight", "mago", "wizard", "animal", "perro", "dog",
              "gato", "cat", "dragon", "criatura", "creature", "humano", "human", "soldado", "soldier")


class OfflineAgent:
    """The chat without an Anthropic key: understands the usual requests (make, ideas, animate, print)
    and runs the same tools. Answers in the user's language; free and offline."""

    def __init__(self):
        self.memory: dict[str, dict] = {}

    async def chat(self, message: str, conversation_id: str | None, tools: dict[str, ToolFn],
                   language: str = "es") -> Turn:
        cid = conversation_id if conversation_id in self.memory else uuid.uuid4().hex
        mem = self.memory.setdefault(cid, {})
        while len(self.memory) > MAX_CONVERSATIONS:
            self.memory.pop(next(iter(self.memory)))
        es = language == "es"
        artifacts: list[dict] = []

        async def call(name, args):
            result, extra = await tools[name](args)
            artifacts.extend(extra)
            return result

        try:
            intent = await self._intent(message, mem, es)
            reply = await self._act(intent, mem, call, es)
        except LookupError as exc:
            reply = str(exc)
        except Exception as exc:  # noqa: BLE001 - shown in the chat
            reply = ("No pude hacerlo: " if es else "I couldn't do it: ") + str(getattr(exc, "detail", None) or exc)
        mem.setdefault("history", []).extend([{"role": "user", "content": message},
                                              {"role": "assistant", "content": reply}])
        mem["history"] = mem["history"][-8:]
        return Turn(reply, cid, artifacts)

    async def _intent(self, message: str, mem: dict, es: bool) -> dict:
        return rule_intent(message, mem)

    async def _act(self, intent: dict, mem: dict, call, es: bool) -> str:
        action = intent["action"]
        if action == "answer":
            return intent.get("reply") or ("¿Qué quieres crear?" if es else "What would you like to make?")
        if action == "pick" and mem.get("concepts"):
            n = max(1, min(int(intent.get("concept") or 1), len(mem["concepts"])))
            jobs = await call("generate_3d", {"prompt": mem["prompt"], "style": mem.get("style", "auto"),
                                              "quality": "high", "concept_id": mem["concepts"][n - 1],
                                              "a_pose": mem.get("a_pose", False)})
            mem["job"] = jobs[0]["job_id"]
            return (f"¡Vamos con el concepto {n}! Estoy creando el modelo 3D en calidad alta; aparece en "
                    "«Mis modelos» en unos minutos. Después puedo animarlo o revisar si se imprime bien."
                    if es else f"Going with concept {n}! I'm building the 3D model in high quality; it shows up "
                    "in “My models” in a few minutes. Then I can animate it or check it for printing.")
        if action == "list":
            models = await call("list_models", {"limit": 5})
            lines = [f"• {m['title']} ({m['status']})" for m in models] or ["—"]
            return ("Tus últimos modelos:\n" if es else "Your latest models:\n") + "\n".join(lines)
        if action == "print":
            job = await self._last_model(call, mem, es)
            mm = float(intent.get("size_mm") or 100)
            r = await call("check_print", {"job_id": job, "size_mm": mm,
                                           "printer": intent.get("printer") or "bambu_a1"})
            return _print_reply(r, mm, es)
        if action == "animate":
            job = await self._last_model(call, mem, es)
            query = intent.get("query") or ""
            found = await call("find_animations", {"query": query, "body": "humanoid"}) if query else []
            clips = ["idle", "walk", "run", "jump", "wave"] + [c["id"] for c in found[:5]]
            jobs = await call("rig_character", {"job_id": job, "animations": list(dict.fromkeys(clips)),
                                                 "body": "auto"})
            mem["job"] = jobs[-1]["job_id"]
            names = ", ".join(c["label"] for c in found[:5])
            return ("¡Hecho! Le pongo esqueleto, texturas HD y animaciones (reposo, caminar, correr, saltar, "
                    f"saludar{', ' + names if names else ''}). Lo verás en «Mis modelos», listo para Blender, "
                    "Unity, Unreal o Godot." if es else
                    "Done! Adding a skeleton, HD textures and animations (idle, walk, run, jump, wave"
                    f"{', ' + names if names else ''}). It will appear in “My models”, ready for Blender, "
                    "Unity, Unreal or Godot.")
        prompt, style, a_pose = intent["prompt"], intent.get("style") or "auto", bool(intent.get("a_pose"))
        mem.update(prompt=prompt, style=style, a_pose=a_pose)
        if action == "concepts":
            result = await call("make_concepts", {"prompt": prompt, "style": style})
            mem["concepts"] = result["concept_ids"]
            return ("Te dibujé 4 conceptos. Toca el que más te guste (o escribe «el 2») y lo hago en 3D."
                    if es else "I drew 4 concepts. Tap the one you like (or type “number 2”) and I'll make it 3D.")
        jobs = await call("generate_3d", {"prompt": prompt, "style": style, "quality": "high",
                                          "concept_id": None, "a_pose": a_pose})
        mem["job"] = jobs[0]["job_id"]
        name = intent.get("name") or prompt
        extra_es = " Como es un personaje, va en A-pose para poder animarlo después." if a_pose else ""
        extra_en = " Since it's a character, it comes in an A-pose so it can be animated later." if a_pose else ""
        return (f"¡Creando «{name}» en 3D, calidad alta!{extra_es} Cuando termine dime «anímalo» o "
                "«¿se puede imprimir a 15 cm?»." if es else
                f"Creating “{name}” in 3D, high quality!{extra_en} When it's done tell me “animate it” "
                "or “can I print it at 15 cm?”.")

    @staticmethod
    async def _last_model(call, mem, es: bool) -> str:
        if mem.get("job"):
            return mem["job"]
        models = await call("list_models", {"limit": 20})
        ready = next((m for m in models if m["status"] == "succeeded"), None)
        if not ready:
            raise LookupError("Primero crea un modelo: dime qué quieres hacer." if es
                              else "First make a model: tell me what you want to create.")
        return ready["job_id"]


STYLES = ("auto", "realistic", "cartoon", "handpainted", "lowpoly", "voxel", "sculpture")
_STYLE_WORDS = (("cartoon", "cartoon"), ("caricatura", "cartoon"), ("anime", "cartoon"), ("realista", "realistic"),
                ("realistic", "realistic"), ("low poly", "lowpoly"), ("lowpoly", "lowpoly"), ("voxel", "voxel"),
                ("pintado", "handpainted"), ("hand painted", "handpainted"), ("escultura", "sculpture"),
                ("sculpture", "sculpture"))


def _printer_in(text: str) -> str:
    return next((pid for pid, words in _PRINTERS if any(re.search(rf"\b{w}\b", text) for w in words)), "bambu_a1")


def _size_in(text: str) -> float:
    size = _SIZE.search(text)
    return float(size.group(1).replace(",", ".")) * (10 if size.group(2) == "cm" else 1) if size else 100.0


def rule_intent(message: str, mem: dict) -> dict:
    """Keyword understanding of a request (used without a text model, and as its safety net)."""
    text = fold(message).strip()
    creating = bool(_FILLER.match(text)) and not text.startswith(("haz que", "make it"))
    if mem.get("concepts") and (m := _CONCEPT_N.search(text)) and len(text) < 40:
        return {"action": "pick", "concept": int(m.group(1))}
    if any(k in text for k in _LIST):
        return {"action": "list"}
    if not creating and any(k in text for k in _PRINT):
        return {"action": "print", "size_mm": _size_in(text), "printer": _printer_in(text)}
    if not creating and any(k in text for k in _ANIMATE):
        return {"action": "animate",
                "query": " ".join(w for w in text.split() if len(w) > 3 and not w.startswith(("anim", "rig")))}
    prompt = _FILLER.sub("", message.strip().rstrip(".!?"), count=3) or message.strip()
    style = next((s for k, s in _STYLE_WORDS if k in text), "auto")
    action = "concepts" if any(k in text for k in _IDEAS) or len(text.split()) <= 3 else "create"
    return {"action": action, "prompt": prompt, "style": style, "a_pose": any(k in text for k in _CHARACTER),
            "explicit": bool(_MAKE.match(text))}


LOCAL_SYSTEM = """You turn a user's message for a 3D studio app into ONE JSON object, nothing else.
Fields: "action": one of "create" (make a 3D model now), "concepts" (draw 4 concept images first, when \
the idea is vague or they ask for ideas/options), "pick" (they chose concept number N), "animate" (rig and \
animate their last model), "print" (check 3D printing), "list" (show their models), "answer" (a question \
about 3D, Blender, game engines or printing: put the answer in "reply").
"prompt": for create/concepts, a short vivid ENGLISH description of the object (subject, shape, materials, \
colours). "name": the object in the user's language, 2-5 words. "style": one of auto, realistic, cartoon, \
handpainted, lowpoly, voxel, sculpture. "a_pose": true for characters/creatures that will be animated. \
"concept": number for pick. "query": English motion words for animate (e.g. "dance happy"). "size_mm": \
number for print (convert cm to mm; default 100). "printer": one of bambu_a1mini, bambu_a1, bambu_h2d, \
prusa_mk4, prusa_mini, ender3, creality_k1, creality_k2, elegoo_n4, anycubic_kobra3, saturn4, photon_m5. \
"reply": for answer only, 1-4 sentences in the user's language ({language}).
A new object to make is always create or concepts, even if they also want to animate or print it later."""

# few-shot examples: small local models follow examples better than rules
LOCAL_EXAMPLES = [
    ("quiero un dragón bebé morado para animarlo en mi juego",
     {"action": "create", "prompt": "cute baby dragon, purple scales, small wings, big eyes, chubby body",
      "name": "dragón bebé morado", "style": "cartoon", "a_pose": True}),
    ("dame ideas para una espada mágica",
     {"action": "concepts", "prompt": "magic sword with glowing runes on the blade and an ornate golden hilt",
      "name": "espada mágica", "style": "auto", "a_pose": False}),
    ("anímalo bailando", {"action": "animate", "query": "dance happy"}),
    ("¿cabe en mi bambu a1 mini a 20 cm?", {"action": "print", "size_mm": 200, "printer": "bambu_a1mini"}),
    ("¿qué formato uso para Unity?", {"action": "answer", "reply": "Usa FBX o GLB: los dos llevan el esqueleto y las animaciones."}),
]


class LocalModelAgent(OfflineAgent):
    """Free assistant powered by the small text model on the user's GPU: it understands free-form
    requests and turns them into the same actions; keyword rules remain the safety net."""

    def __init__(self, llm: Callable[[list[dict]], Awaitable[str]]):
        super().__init__()
        self.llm = llm

    async def _intent(self, message: str, mem: dict, es: bool) -> dict:
        fallback = rule_intent(message, mem)
        if fallback["action"] == "pick":  # "el 2" is unambiguous
            return fallback
        context = mem.get("history", [])[-4:]
        if mem.get("concepts"):
            context = context + [{"role": "assistant", "content": f"(4 concept images shown for: {mem['prompt']})"}]
        shots = [m for q, a in LOCAL_EXAMPLES for m in ({"role": "user", "content": q},
                                                         {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})]
        messages = [{"role": "system", "content": LOCAL_SYSTEM.format(language="Spanish" if es else "English")},
                    *shots, *context, {"role": "user", "content": message}]
        try:
            raw = await self.llm(messages)
            data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:  # noqa: BLE001 - model down, slow or not JSON: keyword rules
            return fallback
        action = data.get("action")
        if action not in ("create", "concepts", "pick", "animate", "print", "list", "answer"):
            return fallback
        if action == "answer":
            data["reply"] = data.get("reply") or data.get("prompt")  # small models sometimes swap the fields
        # an explicit "make me…" is a new object, whatever the model says
        if fallback.get("explicit") and action in ("animate", "print", "list", "answer"):
            data = {**data, "action": fallback["action"]}
            action = fallback["action"]
        if action in ("create", "concepts") and not str(data.get("prompt") or "").strip():
            return fallback
        if action == "pick" and not mem.get("concepts"):
            return fallback
        intent = {**fallback, **{k: v for k, v in data.items() if v not in (None, "")}}
        if intent.get("style") not in STYLES:
            intent["style"] = fallback.get("style", "auto")
        if intent.get("printer") not in {p for p, _ in _PRINTERS}:
            intent["printer"] = fallback.get("printer", "bambu_a1")
        try:
            intent["size_mm"] = float(intent.get("size_mm") or 100)
        except (TypeError, ValueError):
            intent["size_mm"] = 100.0
        return intent


def _print_reply(r: dict, mm: float, es: bool) -> str:
    x, y, z = r["size_mm"]
    lines = [f"{r['printer']} · {x:.0f} × {y:.0f} × {z:.0f} mm"]
    if es:
        lines.append("✔ Cerrado (watertight)" if r["watertight"] else f"✘ Tiene {r['holes']} agujeros: usa «Reparar»")
        if r["volume_cm3"]:
            lines.append(f"Volumen {r['volume_cm3']} cm³ · ≈{r['weight_g']} g de PLA")
        lines.append("✔ Cabe en la cama" if r["fits_bed"] else "✘ No cabe: usa «Dividir para la cama»")
    else:
        lines.append("✔ Watertight" if r["watertight"] else f"✘ It has {r['holes']} holes: use “Repair”")
        if r["volume_cm3"]:
            lines.append(f"Volume {r['volume_cm3']} cm³ · ≈{r['weight_g']} g of PLA")
        lines.append("✔ Fits the bed" if r["fits_bed"] else "✘ Too big: use “Split to fit the bed”")
    return "\n".join(lines)
