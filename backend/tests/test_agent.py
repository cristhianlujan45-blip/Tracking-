"""The AI agent's tool loop and its API, with a scripted stand-in for the Claude client."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from app import agent as ai_agent
from app.config import Settings
from app.main import create_app

from .test_api import wait_done


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(name, args, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


class FakeClient:
    """Answers each create() with the next scripted response and records what it was sent."""

    def __init__(self, script, models=("claude-haiku-x", "claude-opus-x")):
        self.script = list(script)
        self.calls = []
        self.model_ids = models
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)

    async def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        stop, content = item
        return SimpleNamespace(stop_reason=stop, content=content)

    def _list(self, **kwargs):
        async def gen():
            for model_id in self.model_ids:
                yield SimpleNamespace(id=model_id)

        return gen()


def run(coro):
    import asyncio

    return asyncio.run(coro)


def test_tool_loop_runs_tools_and_keeps_history():
    calls = []

    async def list_models(args):
        calls.append(args)
        return [{"job_id": "abc"}], [{"type": "job", "job": {"id": "abc"}}]

    client = FakeClient([("tool_use", [text("Miro…"), tool_use("list_models", {"limit": 3})]),
                         ("end_turn", [text("Tienes 1 modelo.")])])
    bot = ai_agent.Agent("", client=client)
    turn = run(bot.chat("¿Qué modelos tengo?", None, {"list_models": list_models}))
    assert turn.reply == "Tienes 1 modelo."
    assert turn.artifacts == [{"type": "job", "job": {"id": "abc"}}]
    assert calls == [{"limit": 3}]
    # newest Opus-class model picked from the key's model list
    assert client.calls[0]["model"] == "claude-opus-x"
    assert client.calls[0]["fallbacks"] == "default"
    # the tool result goes back paired with its tool_use id, after the unchanged assistant content
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][1].name == "list_models"
    result = second[2]["content"][0]
    assert result["tool_use_id"] == "tu_1" and not result["is_error"] and "abc" in result["content"]

    # the conversation continues with the same id
    client.script.append(("end_turn", [text("¡Hola otra vez!")]))
    again = run(bot.chat("hola", turn.conversation_id, {}))
    assert again.conversation_id == turn.conversation_id
    assert len(client.calls[2]["messages"]) == 5


def test_tool_errors_go_back_to_claude():
    async def broken(args):
        raise RuntimeError("sin modelo")

    client = FakeClient([("tool_use", [tool_use("job_status", {"job_id": "x"})]),
                         ("end_turn", [text("Ese modelo no existe.")])])
    turn = run(ai_agent.Agent("", client=client).chat("estado", None, {"job_status": broken}))
    assert turn.reply == "Ese modelo no existe."
    result = client.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] and "sin modelo" in result["content"]


def test_refusal_and_api_errors():
    client = FakeClient([("refusal", [])], models=("claude-opus-x",))
    bot = ai_agent.Agent("", client=client)
    turn = run(bot.chat("algo", None, {}, language="en"))
    assert "can't help" in turn.reply
    assert bot.conversations[turn.conversation_id] == []  # nothing left that cannot be continued

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    client.script.append(anthropic.AuthenticationError("bad key", response=response, body=None))
    with pytest.raises(ai_agent.AgentError, match="clave"):
        run(bot.chat("hola", turn.conversation_id, {}))
    assert bot.conversations[turn.conversation_id] == []

    with pytest.raises(ai_agent.AgentError, match="Ajustes"):
        ai_agent.Agent("")


@pytest.fixture
def client(tmp_path):
    settings = Settings(provider="mock", data_dir=tmp_path, mock_step_delay=0)
    with TestClient(create_app(settings)) as c:
        yield c


def test_free_assistant_without_a_key(client):
    """No key: the built-in assistant still makes concepts, 3D, print checks and animations."""
    def say(message, cid=None, lang="es"):
        resp = client.post("/api/agent/chat", json={"message": message, "conversation_id": cid, "lang": lang})
        assert resp.status_code == 200, resp.text
        return resp.json()

    first = say("un robot bonito")
    assert first["artifacts"][0]["type"] == "concepts" and "concepto" in first["reply"]
    cid = first["conversation_id"]
    second = say("el 2", cid)
    job = second["artifacts"][0]["job"]
    assert job["mode"] == "image" and job["quality"] == "high"
    assert wait_done(client, job["id"])["status"] == "succeeded"
    printed = say("¿se puede imprimir a 12 cm en mi ender 3?", cid)
    assert "Ender-3" in printed["reply"] and "mm" in printed["reply"]
    animated = say("anímalo bailando", cid)
    rig = animated["artifacts"][-1]["job"]
    assert rig["mode"] == "rig" and "esqueleto" in animated["reply"]
    assert wait_done(client, rig["id"], timeout=90)["status"] == "succeeded"
    # a direct request makes the model right away, in English too
    made = say("create a medieval wooden treasure chest with gold trim", lang="en")
    assert made["artifacts"][0]["job"]["mode"] == "text" and "Creating" in made["reply"]


def test_agent_settings_key_is_never_returned(client):
    resp = client.put("/api/settings", json={"anthropic_key": "sk-ant-secret-1234"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["anthropic_key_set"] and body["anthropic_key_hint"] == "…1234"
    assert "sk-ant-secret" not in resp.text
    # an empty field keeps the saved key
    assert client.put("/api/settings", json={"anthropic_key": ""}).json()["anthropic_key_set"]


def test_agent_drives_the_app(client):
    """Concepts → 3D from the chosen concept → animations → rig, all through the chat endpoint."""
    fake = FakeClient([
        ("tool_use", [tool_use("make_concepts", {"prompt": "a cute robot", "style": "cartoon"})]),
        ("end_turn", [text("Elige un concepto.")]),
    ])
    client.app.state.agent = ai_agent.Agent("", client=fake)
    client.app.state.agent_key = ""
    resp = client.post("/api/agent/chat", json={"message": "un robot bonito", "lang": "es"})
    assert resp.status_code == 200, resp.text
    first = resp.json()
    concepts = first["artifacts"][0]
    assert concepts["type"] == "concepts" and len(concepts["concepts"]) == 4
    concept_id = concepts["concepts"][0]["id"]

    fake.script += [
        ("tool_use", [tool_use("generate_3d", {"prompt": "cute robot", "style": "cartoon", "quality": "draft",
                                               "concept_id": concept_id, "a_pose": True})]),
        ("end_turn", [text("Generando.")]),
    ]
    resp = client.post("/api/agent/chat", json={"message": "el primero",
                                                "conversation_id": first["conversation_id"]})
    job = resp.json()["artifacts"][0]["job"]
    assert job["mode"] == "image" and job["params"]["concept_id"] == concept_id
    assert wait_done(client, job["id"])["status"] == "succeeded"

    fake.script += [
        ("tool_use", [tool_use("find_animations", {"query": "baile", "body": "humanoid"}, "tu_a"),
                      tool_use("check_print", {"job_id": job["id"], "size_mm": 120, "printer": "bambu_a1"}, "tu_b")]),
        ("tool_use", [tool_use("rig_character", {"job_id": job["id"], "animations": ["walk"], "body": "humanoid"})]),
        ("end_turn", [text("Listo.")]),
    ]
    resp = client.post("/api/agent/chat", json={"message": "anímalo bailando y dime si se imprime",
                                                "conversation_id": first["conversation_id"]})
    assert resp.status_code == 200, resp.text
    results = fake.calls[-2]["messages"][-1]["content"]
    assert not results[0]["is_error"] and "Bail" in results[0]["content"]
    assert not results[1]["is_error"] and '"fits_bed": true' in results[1]["content"]
    rig = resp.json()["artifacts"][-1]["job"]
    assert rig["mode"] == "rig"
    assert wait_done(client, rig["id"], timeout=60)["status"] == "succeeded"


def test_local_text_model_understands_free_form_requests():
    """The GPU's small text model turns a free-form message into an action; bad output falls back."""
    import json as _json

    answers = [_json.dumps({"action": "print", "size_mm": 150, "printer": "prusa_mk4"}),
               "lo siento, no sé",  # not JSON: keyword rules take over
               _json.dumps({"action": "answer", "reply": "Exporta en FBX para Unity."})]

    async def llm(messages):
        assert messages[0]["role"] == "system" and "JSON" in messages[0]["content"]
        return answers.pop(0)

    calls = []

    def tool(name, result):
        async def fn(args):
            calls.append((name, args))
            return result, []
        return fn

    tools = {"list_models": tool("list_models", [{"job_id": "j1", "title": "Robot", "status": "succeeded"}]),
             "check_print": tool("check_print", {"printer": "Prusa MK4", "size_mm": [80, 60, 150], "watertight": True,
                                                 "holes": 0, "volume_cm3": 40.0, "weight_g": 22.3, "fits_bed": True}),
             "generate_3d": tool("generate_3d", [{"job_id": "j2", "status": "queued"}])}
    bot = ai_agent.LocalModelAgent(llm)
    turn = run(bot.chat("quiero sacarlo en mi impresora grandecito, como de 15 centímetros", None, tools))
    assert calls[-1] == ("check_print", {"job_id": "j1", "size_mm": 150.0, "printer": "prusa_mk4"})
    assert "Prusa MK4" in turn.reply
    run(bot.chat("crea un cofre del tesoro de madera con detalles dorados", turn.conversation_id, tools))
    assert calls[-1][0] == "generate_3d"  # rules understood it although the model failed
    answer = run(bot.chat("¿qué formato uso para Unity?", turn.conversation_id, tools))
    assert answer.reply == "Exporta en FBX para Unity."
