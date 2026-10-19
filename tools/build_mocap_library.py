"""Build the bundled mocap library from the CMU Motion Capture Database (BVH by B. Hahne/cgspeed).

CMU places no restrictions on the use of the dataset; the BVH conversion adds none either.
Usage: python tools/build_mocap_library.py <cmubvh checkout> [tools/mocap_selection.py]
Writes backend/app/rigging/mocap/cmu.zip (one trimmed 30 fps BVH per clip + index.json).
"""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.rigging.animations import parse_bvh  # noqa: E402

MAX_SECONDS, FPS = 10.0, 30


def write_bvh(joints, motion, frame_time) -> str:
    children = {i: [j for j, jt in enumerate(joints) if jt.parent == i] for i in range(len(joints))}
    lines = ["HIERARCHY"]

    def emit(i, depth):
        jt, ind = joints[i], "  " * depth
        off = " ".join(f"{v:.4f}" for v in jt.offset)
        if jt.name.endswith("_End") and not jt.channels:
            lines.extend([f"{ind}End Site", ind + "{", f"{ind}  OFFSET {off}", ind + "}"])
            return
        lines.extend([f"{ind}{'ROOT' if jt.parent < 0 else 'JOINT'} {jt.name}", ind + "{", f"{ind}  OFFSET {off}",
                      f"{ind}  CHANNELS {len(jt.channels)} {' '.join(jt.channels)}"])
        for c in children[i]:
            emit(c, depth + 1)
        lines.append(ind + "}")

    emit(0, 0)
    lines += ["MOTION", f"Frames: {len(motion)}", f"Frame Time: {frame_time:.6f}"]
    lines += [" ".join(f"{v:.2f}" for v in row) for row in motion]
    return "\n".join(lines) + "\n"


def main(src: Path, selection_file: Path) -> None:
    spec = importlib.util.spec_from_file_location("selection", selection_file)
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)
    catalogue = {r["id"]: r for r in json.loads(Path("/tmp/claude-0/cmu_index.json").read_text())}
    out = ROOT / "backend/app/rigging/mocap/cmu.zip"
    index = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for clip_id, label, label_en, category in selection.SELECTION:
            row = catalogue[clip_id]
            with zipfile.ZipFile(src / row["path"]) as inner:
                name = next(n for n in inner.namelist() if n.lower().endswith(".bvh"))
                text = inner.read(name).decode("utf-8", errors="replace")
            joints, motion, frame_time = parse_bvh(text)
            step = max(1, round((1 / FPS) / frame_time))
            motion = motion[1:][::step][: int(MAX_SECONDS * FPS)]  # frame 1 is a calibration T-pose
            zf.writestr(f"{clip_id}.bvh", write_bvh(joints, motion, frame_time * step))
            index.append({"id": f"cmu_{clip_id}", "file": f"{clip_id}.bvh", "label": label, "label_en": label_en,
                          "category": category,
                          "source": f"CMU {clip_id}: {row['desc']}", "seconds": round(len(motion) / FPS, 1)})
        zf.writestr("index.json", json.dumps(index, ensure_ascii=False, indent=1))
    print(f"{len(index)} clips → {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "tools" / "mocap_selection.py")
