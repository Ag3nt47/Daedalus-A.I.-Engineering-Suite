from __future__ import annotations

import struct
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ICON_SIZES = {16, 24, 32, 48, 64, 128, 256}


def _ico_sizes(payload: bytes) -> set[int]:
    reserved, image_type, count = struct.unpack_from("<HHH", payload)
    assert reserved == 0
    assert image_type == 1
    sizes: set[int] = set()
    for index in range(count):
        width, height = struct.unpack_from("BB", payload, 6 + index * 16)
        decoded_width = width or 256
        decoded_height = height or 256
        assert decoded_width == decoded_height
        sizes.add(decoded_width)
    return sizes


def test_application_icon_assets_are_packaged_and_multiresolution() -> None:
    package = files("daedalus").joinpath("assets")
    png = package.joinpath("daedalus-app-icon.png").read_bytes()
    ico = package.joinpath("daedalus.ico").read_bytes()

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 10_000
    assert _ico_sizes(ico) == EXPECTED_ICON_SIZES


def test_windows_launcher_sources_use_the_branded_identity() -> None:
    launcher_source = (ROOT / "tools" / "launcher" / "DaedalusLauncher.cs").read_text(
        encoding="utf-8"
    )
    installer = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8")
    run_script = (ROOT / "Run-Daedalus.bat").read_text(encoding="utf-8")

    assert 'AssemblyTitle("Daedalus AI Engineering Suite")' in launcher_source
    assert "Daedalus.exe" in installer
    assert 'Icon="$Launcher,0"' in installer
    assert "Daedalus.exe" in run_script
