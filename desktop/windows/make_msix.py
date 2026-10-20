"""Package the built app (dist/Forja3D) as an MSIX for the Microsoft Store.

The Store signs the package itself, so a Store install shows no SmartScreen warning and updates
on its own. Identity values come from Partner Center (Product identity page) and are passed as
environment variables by the release workflow:

    MSIX_IDENTITY_NAME, MSIX_PUBLISHER (e.g. CN=XXXXXXXX-...), MSIX_PUBLISHER_DISPLAY, APP_VERSION

    python desktop/windows/make_msix.py
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dist" / "Forja3D"
LAYOUT = ROOT / "build" / "msix"
OUT = ROOT / "dist-installer"

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap rescap">
  <Identity Name="{name}" Publisher="{publisher}" Version="{version}" ProcessorArchitecture="x64" />
  <Properties>
    <DisplayName>Forja3D</DisplayName>
    <PublisherDisplayName>{publisher_display}</PublisherDisplayName>
    <Logo>assets\\StoreLogo.png</Logo>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0" />
  </Dependencies>
  <Resources>
    <Resource Language="es-es" />
    <Resource Language="en-us" />
  </Resources>
  <Applications>
    <Application Id="Forja3D" Executable="Forja3D.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="Forja3D" Description="Estudio 3D con IA: texto e imágenes a modelos 3D, rig, animación e impresión"
                          BackgroundColor="transparent" Square150x150Logo="assets\\Square150x150Logo.png"
                          Square44x44Logo="assets\\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="assets\\Wide310x150Logo.png" />
      </uap:VisualElements>
    </Application>
  </Applications>
  <Capabilities>
    <Capability Name="internetClient" />
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"""

LOGOS = {"StoreLogo.png": (50, 50), "Square44x44Logo.png": (44, 44), "Square150x150Logo.png": (150, 150),
         "Wide310x150Logo.png": (310, 150)}


def store_version(app_version: str) -> str:
    """The Store wants four numbers with the last one 0: 1.0.12 -> 1.0.12.0."""
    parts = [int(p) for p in app_version.split(".")[:3]]
    return ".".join(map(str, parts + [0] * (3 - len(parts)) + [0]))


def logos(target: Path) -> None:
    icon = Image.open(ROOT / "desktop" / "assets" / "forja3d.png").convert("RGBA")
    target.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in LOGOS.items():
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        side = min(w, h)
        canvas.alpha_composite(icon.resize((side, side), Image.LANCZOS), ((w - side) // 2, (h - side) // 2))
        canvas.save(target / name)


def makeappx() -> str:
    found = sorted(glob.glob(r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\makeappx.exe"))
    if not found:
        raise SystemExit("makeappx.exe no encontrado (Windows SDK)")
    return found[-1]


def main() -> None:
    env = os.environ
    missing = [k for k in ("MSIX_IDENTITY_NAME", "MSIX_PUBLISHER", "APP_VERSION") if not env.get(k)]
    if missing:
        raise SystemExit(f"Faltan variables: {', '.join(missing)}")
    if not (SRC / "Forja3D.exe").is_file():
        raise SystemExit("Primero compila la app: pyinstaller desktop/forja3d.spec")
    shutil.rmtree(LAYOUT, ignore_errors=True)
    shutil.copytree(SRC, LAYOUT)
    logos(LAYOUT / "assets")
    (LAYOUT / "AppxManifest.xml").write_text(MANIFEST.format(
        name=env["MSIX_IDENTITY_NAME"], publisher=env["MSIX_PUBLISHER"],
        publisher_display=env.get("MSIX_PUBLISHER_DISPLAY") or "Forja3D",
        version=store_version(env["APP_VERSION"])), encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    target = OUT / f"Forja3D-{env['APP_VERSION']}-store.msix"
    subprocess.run([makeappx(), "pack", "/o", "/d", str(LAYOUT), "/p", str(target)], check=True)
    print(f"MSIX listo para subir a Partner Center: {target}")


if __name__ == "__main__":
    sys.exit(main())
