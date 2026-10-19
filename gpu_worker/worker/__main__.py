"""Start the GPU worker:  python -m worker --backend hunyuan [--texture] [--multiview]"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .backends import build_backend, prefetch
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Forja3D: servidor de IA local en tu GPU")
    parser.add_argument("--backend", choices=["hunyuan", "dummy"], default="hunyuan")
    parser.add_argument("--host", default="127.0.0.1", help="usa 0.0.0.0 para exponerlo en tu red (pon un token)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--image-model", default="stabilityai/sd-turbo",
                        help="modelo texto→imagen (stabilityai/sd-turbo o stabilityai/sdxl-turbo)")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="profundidad para escenas/relieves (…-Base-hf = más detalle, ~400 MB de VRAM)")
    parser.add_argument("--shape-model", default="tencent/Hunyuan3D-2mini")
    parser.add_argument("--shape-subfolder", default="hunyuan3d-dit-v2-mini")
    parser.add_argument("--texture", action="store_true",
                        help="activa Hunyuan3D-Paint (texturas y retexturizado). Lento con 8 GB de VRAM")
    parser.add_argument("--multiview", action="store_true", help="activa Hunyuan3D-2mv para varias vistas")
    parser.add_argument("--dummy-delay", type=float, default=0.3)
    parser.add_argument("--prefetch", action="store_true", help="solo descarga los modelos y termina")
    args = parser.parse_args()

    if args.prefetch:
        if args.backend == "dummy":
            print("@@ 100 Modelos listos", flush=True)
            return
        prefetch(args.shape_model, args.shape_subfolder, args.image_model, args.depth_model, args.multiview,
                 args.texture, lambda pct, text: print(f"@@ {pct} {text}", flush=True))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.backend == "dummy":
        backend = build_backend("dummy", delay=args.dummy_delay, texture=args.texture, multiview=args.multiview)
    else:
        backend = build_backend(
            "hunyuan", shape_model=args.shape_model, shape_subfolder=args.shape_subfolder,
            image_model=args.image_model, depth_model=args.depth_model,
            texture=args.texture, multiview=args.multiview,
        )
    info = backend.info()
    logging.info("Backend %s · GPU: %s (%s GB) · capacidades: %s", backend.name, info["gpu"], info["vram_gb"],
                 ", ".join(sorted(backend.capabilities())))
    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
