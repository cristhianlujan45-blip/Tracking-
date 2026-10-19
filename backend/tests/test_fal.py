"""fal.ai provider against a simulated fal queue API (no network, no cost)."""

import asyncio
import json

import httpx
import pytest

from app.providers.base import GenerationRequest, InputImage
from app.providers.fal import ENGINES, FalProvider

GLB = b"glTF fake"
PNG = b"\x89PNG fake"


class FakeFal:
    def __init__(self, result_key="model_glb"):
        self.calls: list[tuple[str, dict]] = []
        self.result_key = result_key

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://queue.fal.run/") and request.method == "POST":
            model = url.removeprefix("https://queue.fal.run/")
            self.calls.append((model, json.loads(request.content)))
            return httpx.Response(200, json={"status_url": f"https://queue.fal.run/s/{model}",
                                             "response_url": f"https://queue.fal.run/r/{model}"})
        if "/s/" in url:
            return httpx.Response(200, json={"status": "COMPLETED"})
        if "/r/" in url:
            if "flux" in url:
                return httpx.Response(200, json={"images": [{"url": "https://cdn.fal/ref.png"}]})
            if self.result_key == "model_urls":
                return httpx.Response(200, json={"model_urls": {"glb": {"url": "https://cdn.fal/m.glb"}}})
            return httpx.Response(200, json={self.result_key: {"url": "https://cdn.fal/m.glb"}})
        if url.endswith(".glb"):
            return httpx.Response(200, content=GLB)
        if url.endswith(".png"):
            return httpx.Response(200, content=PNG)
        return httpx.Response(404, json={"detail": "unexpected " + url})


def provider(fake: FakeFal) -> FalProvider:
    return FalProvider("key", transport=httpx.MockTransport(fake.handler), poll_interval=0)


async def _noop(*_):
    pass


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("engine,model,image_key", [
    ("hunyuan3d_v3", "fal-ai/hunyuan3d-v3/image-to-3d", "input_image_url"),
    ("trellis2", "fal-ai/trellis-2", "image_url"),
    ("tripo_h31", "tripo3d/h3.1/image-to-3d", "image_url"),
    ("meshy6", "fal-ai/meshy/v6-preview/image-to-3d", "image_url"),
    ("trellis", "fal-ai/trellis", "image_url"),
])
def test_image_engines(engine, model, image_key):
    fake = FakeFal("model_mesh" if engine == "tripo_h31" else "model_glb")
    req = GenerationRequest("image", None, "high", images=[InputImage(PNG, "image/png")], engine=engine, seed=5,
                            pose="a-pose", symmetry=True)
    result = run(provider(fake).generate(req, _noop))
    assert result.model_glb == GLB
    (called, payload), = fake.calls
    assert called == model
    assert payload[image_key].startswith("data:image/png;base64,")
    if engine == "hunyuan3d_v3":
        assert payload["enable_pbr"] is True and payload["face_count"] == 1_000_000
    if engine == "trellis2":
        assert payload["texture_size"] == 4096 and payload["resolution"] == 1536
    if engine == "tripo_h31":
        assert payload["texture_quality"] == "detailed" and payload["pbr"] is True
    if engine == "meshy6":
        assert payload["pose_mode"] == "a-pose" and payload["symmetry_mode"] == "on" and payload["enable_pbr"]


def test_hunyuan_text_is_native_and_draft_skips_pbr():
    fake = FakeFal()
    req = GenerationRequest("text", "un dragón de jade", "draft", engine="hunyuan3d_v3", style="lowpoly")
    result = run(provider(fake).generate(req, _noop))
    assert result.model_glb == GLB and result.reference_image is None
    (called, payload), = fake.calls
    assert called == "fal-ai/hunyuan3d-v3/text-to-3d"
    assert payload["enable_pbr"] is False and payload["generate_type"] == "LowPoly"
    assert payload["prompt"].startswith("un dragón de jade")


def test_text_without_native_mode_goes_through_flux():
    fake = FakeFal()
    req = GenerationRequest("text", "una silla", "standard", engine="trellis2")
    result = run(provider(fake).generate(req, _noop))
    assert [c[0] for c in fake.calls] == ["fal-ai/flux/schnell", "fal-ai/trellis-2"]
    assert result.reference_image == PNG


def test_hunyuan_multiview_maps_extra_views():
    fake = FakeFal()
    views = [InputImage(PNG, "image/png") for _ in range(4)]
    run(provider(fake).generate(GenerationRequest("multi_image", None, "standard", images=views,
                                                  engine="hunyuan3d_v3"), _noop))
    payload = fake.calls[0][1]
    assert {"back_image_url", "left_image_url", "right_image_url"} <= payload.keys()
    # app slots: front, side, back, right
    views = [InputImage(bytes([i]) + PNG, "image/png") for i in range(3)]
    fake = FakeFal()
    run(provider(fake).generate(GenerationRequest("multi_image", None, "standard", images=views,
                                                  engine="hunyuan3d_v3"), _noop))
    payload = fake.calls[0][1]
    assert set(payload) >= {"left_image_url", "back_image_url"} and "right_image_url" not in payload


def test_retexture_uses_meshy7_at_8k():
    fake = FakeFal()
    req = GenerationRequest("text", "armadura dorada", "high", style="realistic")
    glb = run(provider(fake).retexture(b"glTF source", req, _noop))
    assert glb == GLB
    (called, payload), = fake.calls
    assert called == "meshy/v7/retexture"
    assert payload["texture_resolution"] == "8k" and payload["enable_pbr"] and payload["enable_original_uv"]
    assert payload["model_url"].startswith("data:model/gltf-binary;base64,")
    assert payload["text_style_prompt"].startswith("armadura dorada")


def test_model_urls_fallback_and_errors():
    fake = FakeFal("model_urls")
    req = GenerationRequest("image", None, "standard", images=[InputImage(PNG)], engine="meshy6")
    assert run(provider(fake).generate(req, _noop)).model_glb == GLB

    def failing(request):
        return httpx.Response(402, json={"detail": "saldo insuficiente"})

    p = FalProvider("key", transport=httpx.MockTransport(failing), poll_interval=0)
    from app.providers import GenerationError

    with pytest.raises(GenerationError, match="saldo insuficiente"):
        run(p.generate(req, _noop))


def test_engine_catalogue_is_public():
    p = FalProvider("key")
    ids = [e["id"] for e in p.engines]
    assert ids == list(ENGINES) and p.default_engine == "hunyuan3d_v3"
    assert next(e for e in p.engines if e["id"] == "hunyuan3d_v3")["native_text"] is True
