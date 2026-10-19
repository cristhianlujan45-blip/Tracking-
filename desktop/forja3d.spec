# PyInstaller spec for the Forja3D desktop app (Windows .exe folder, macOS .app, Linux folder).
#   pyinstaller desktop/forja3d.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
ICON = ROOT / "desktop" / "assets" / ("forja3d.ico" if sys.platform == "win32" else "forja3d.png")

datas = [
    (str(ROOT / "frontend"), "frontend"),
    (str(ROOT / "integrations" / "blender" / "forja3d_blender.py"), "integrations/blender"),
    # run by Blender as a script, so it must exist as a real .py file
    (str(ROOT / "backend" / "app" / "blender_convert.py"), "app"),
    (str(ROOT / "backend" / "app" / "rigging" / "mocap" / "cmu.zip"), "app/rigging/mocap"),
    # the GPU worker is installed by the app into its own Python (see app/local_ai.py)
    (str(ROOT / "gpu_worker" / "worker"), "gpu_worker/worker"),
    (str(ROOT / "gpu_worker" / "requirements.txt"), "gpu_worker"),
]
datas += collect_data_files("trimesh")

hiddenimports = (
    collect_submodules("app")
    + collect_submodules("uvicorn")
    + collect_submodules("webview")
    + ["fast_simplification", "pynanoinstantmeshes", "mapbox_earcut", "rtree", "shapely", "lxml.etree",
       "networkx", "cv2", "multipart", "python_multipart", "scipy.sparse.linalg", "scipy.ndimage"]
    + collect_submodules("anthropic")
)

a = Analysis(
    [str(ROOT / "desktop" / "forja3d_desktop.py")],
    pathex=[str(ROOT / "backend")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "torch", "IPython", "notebook", "PyQt5", "PyQt6", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Forja3D",
    console=False,
    icon=str(ICON),
    target_arch=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Forja3D")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Forja3D.app",
        icon=str(ICON),
        bundle_identifier="app.forja3d.desktop",
        info_plist={
            "CFBundleName": "Forja3D",
            "CFBundleDisplayName": "Forja3D",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
