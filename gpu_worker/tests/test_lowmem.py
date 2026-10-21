"""The low-memory Hunyuan loader builds the same model as Hunyuan's own loader (needs PyTorch)."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
yaml = pytest.importorskip("yaml")

from worker import lowmem  # noqa: E402


class Model(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.inp = torch.nn.Linear(4, width)
        self.norm = torch.nn.LayerNorm(width)
        self.register_buffer("scale", torch.ones(width))


class Vae(Model):
    def __init__(self, width):
        super().__init__(width)
        self.extra = torch.nn.Linear(2, 2)  # not in the file (Hunyuan loads the VAE with strict=False)


class Pipeline:
    def __init__(self, vae, model, scheduler, conditioner, image_processor, device="cuda", dtype=None, **kwargs):
        self.vae, self.model, self.conditioner = vae, model, conditioner
        self.scheduler, self.image_processor, self.kwargs = scheduler, image_processor, kwargs
        self.device, self.dtype = device, dtype


def instantiate(config):
    return {"model": Model, "vae": Vae, "cond": Model, "none": lambda: object()}[config["target"]](
        **config.get("params", {}))


@pytest.fixture
def hunyuan(monkeypatch, tmp_path):
    pipelines = types.ModuleType("hy3dgen.shapegen.pipelines")
    pipelines.instantiate_from_config = instantiate
    for name, module in (("hy3dgen", types.ModuleType("hy3dgen")),
                         ("hy3dgen.shapegen", types.ModuleType("hy3dgen.shapegen")),
                         ("hy3dgen.shapegen.pipelines", pipelines)):
        monkeypatch.setitem(sys.modules, name, module)
    config = {"model": {"target": "model", "params": {"width": 8}}, "vae": {"target": "vae", "params": {"width": 6}},
              "conditioner": {"target": "cond", "params": {"width": 5}},
              "scheduler": {"target": "none"}, "image_processor": {"target": "none"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    torch.manual_seed(0)
    reference = {"model": Model(8), "vae": Vae(6), "conditioner": Model(5)}
    from safetensors.torch import save_file

    weights = {f"{name}.{k}": v.to(torch.float16).contiguous() for name, part in reference.items()
               for k, v in part.state_dict().items() if not k.startswith("extra")}
    save_file(weights, str(tmp_path / "model.fp16.safetensors"))
    return tmp_path, reference


def test_loads_every_part_in_fp16_with_the_file_weights(hunyuan):
    folder, reference = hunyuan
    pipe = lowmem.load_pipeline(Pipeline, str(folder), "tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0", device="cpu")
    for name in ("model", "vae", "conditioner"):
        part = getattr(pipe, name)
        for key, value in reference[name].state_dict().items():
            if key.startswith("extra"):
                continue
            loaded = part.state_dict()[key]
            assert loaded.dtype == torch.float16 and torch.equal(loaded, value.to(torch.float16)), (name, key)
    assert pipe.kwargs["from_pretrained_kwargs"]["model_path"] == "tencent/Hunyuan3D-2"  # FlashVDM needs it


def test_a_file_that_does_not_match_is_refused(hunyuan):
    folder, _ = hunyuan
    config = yaml.safe_load((folder / "config.yaml").read_text())
    config["model"]["params"]["width"] = 16  # another model size than the weights
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(lowmem.LoadError):
        lowmem.load_pipeline(Pipeline, str(folder), "r", "s", device="cpu")
