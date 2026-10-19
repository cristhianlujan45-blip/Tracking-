"""Bundled motion-capture library (CMU Graphics Lab Motion Capture Database, BVH by B. Hahne).

CMU places no restrictions on the data's use. Clips are trimmed to 10 s at 30 fps; see
tools/build_mocap_library.py.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

LIBRARY = Path(__file__).with_name("mocap") / "cmu.zip"


@lru_cache(maxsize=1)
def catalogue() -> dict[str, dict]:
    if not LIBRARY.is_file():
        return {}
    with zipfile.ZipFile(LIBRARY) as zf:
        return {entry["id"]: entry for entry in json.loads(zf.read("index.json"))}


def clip_text(clip_id: str) -> str:
    entry = catalogue()[clip_id]
    with zipfile.ZipFile(LIBRARY) as zf:
        return zf.read(entry["file"]).decode("utf-8")
