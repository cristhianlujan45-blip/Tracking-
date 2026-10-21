"""Load the big Hunyuan3D models on PCs with little memory (8 GB GPU, 16 GB RAM).

Hunyuan's own loader builds every part of the model in system RAM in 32-bit floats and reads the
whole weights file next to it: about 10 GB for a ~3 GB model, which is what makes Windows answer
"the paging file is too small" (os error 1455). Here each part (shape model, VAE, image encoder)
is built on its own, halved to fp16 and filled with its weights one tensor at a time from the file:
on the GPU when it fits there with room to spare (system RAM barely moves), otherwise in RAM, one
part at a time (a few GB instead of ~10).
The result is the same pipeline object Hunyuan builds (same class, same settings).
"""

from __future__ import annotations

import gc
import logging
import os

log = logging.getLogger(__name__)

PARTS = (("model", True), ("vae", False), ("conditioner", True))  # (name, every weight must be in the file)


class LoadError(RuntimeError):
    """The file does not match what this loader expects: use Hunyuan's own loader instead."""


def load_pipeline(cls, folder: str, repo: str, subfolder: str, device: str = "cuda", dtype=None, progress=None):
    """`cls`: Hunyuan3DDiTFlowMatchingPipeline. `folder` holds config.yaml and model.fp16.safetensors.
    `progress(fraction, text)` follows the load."""
    import torch
    import yaml
    from safetensors import safe_open

    from hy3dgen.shapegen.pipelines import instantiate_from_config

    dtype = dtype or torch.float16
    config_path = os.path.join(folder, "config.yaml")
    weights = os.path.join(folder, "model.fp16.safetensors")
    if not (os.path.isfile(config_path) and os.path.isfile(weights)):
        raise LoadError(f"faltan config.yaml o model.fp16.safetensors en {folder}")
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    parts = {}
    say = progress or (lambda fraction, text: None)
    with safe_open(weights, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        sizes = {name: sum(_numel(f.get_slice(k).get_shape()) for k in keys if k.startswith(f"{name}."))
                 for name, _ in PARTS}
        total, done = max(sum(sizes.values()), 1), 0
        for name, strict in PARTS:
            where = _build_device(device, sizes[name])
            say(done / total, f"{PART_LABELS[name]} ({'GPU' if where != 'cpu' else 'RAM'})")
            parts[name] = _build_part(instantiate_from_config, config[name], f, keys, name, strict, where, dtype)
            parts[name].to(device)  # the CPU copy (when built there) is freed here
            gc.collect()
            done += sizes[name]
            say(done / total, PART_LABELS[name])
    kwargs = {"from_pretrained_kwargs": {"model_path": repo, "subfolder": subfolder, "use_safetensors": True,
                                         "variant": "fp16", "dtype": dtype, "device": device}}
    return cls(vae=parts["vae"], model=parts["model"], conditioner=parts["conditioner"],
               scheduler=instantiate_from_config(config["scheduler"]),
               image_processor=instantiate_from_config(config["image_processor"]),
               device=device, dtype=dtype, **kwargs)


PART_LABELS = {"model": "modelo de forma", "vae": "decodificador", "conditioner": "lector de imágenes"}


def _numel(shape) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _build_device(device: str, numel: int) -> str:
    """Build a part on the GPU only when its fp32 version fits with room to spare. On Windows a full
    GPU does not fail: the driver quietly spills into system RAM and everything slows down ~100×."""
    if not str(device).startswith("cuda"):
        return device
    import torch

    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - unknown: the safe choice
        return "cpu"
    return device if free > numel * 4 * 1.3 + 512 * 1024**2 else "cpu"


def _build_part(instantiate, part_config, f, keys, name, strict, device, dtype):
    import torch

    with torch.device(device):  # parameters are created on the GPU, not in RAM
        module = instantiate(part_config)
    module.to(dtype=dtype)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    state = module.state_dict()  # tensors that share storage with the module: filled in place
    prefix = f"{name}."
    from_file = {key[len(prefix):]: key for key in keys if key.startswith(prefix)}
    if strict and from_file:
        missing = [k for k in state if k not in from_file]
        if missing:
            raise LoadError(f"{name}: faltan {len(missing)} pesos en el archivo (p. ej. {missing[0]})")
    with torch.no_grad():
        for short, key in from_file.items():
            target = state.get(short)
            if target is None:
                if strict:
                    raise LoadError(f"{name}: el archivo trae un peso desconocido ({short})")
                continue
            tensor = f.get_tensor(key)
            if tuple(tensor.shape) != tuple(target.shape):
                raise LoadError(f"{name}.{short}: forma {tuple(tensor.shape)} en vez de {tuple(target.shape)}")
            target.copy_(tensor.to(target.dtype))
            del tensor
    module.eval()
    return module
