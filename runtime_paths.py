"""
Runtime path helpers.

These functions centralize "where am I running from?" logic so the app works both:
- as normal Python scripts, and
- when frozen into a single-file EXE (e.g., PyInstaller --onefile).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a frozen executable (PyInstaller, etc.)."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """
    Directory where the executable lives (frozen) or project directory (script).
    This is a good place to look for optional portable dependencies (vendor/*).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    """
    Directory where bundled resources live.
    For PyInstaller onefile, this is the temporary extraction folder.
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".__write_test__"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def user_data_dir(app_name: str = "OCR_PDF2Table") -> Path:
    """
    Resolve a writable directory for settings and other user data.

    Preference order:
    - executable directory (portable mode) if writable
    - %APPDATA%\\<app_name> (Windows)
    - ~/.config/<app_name> (fallback)
    """
    portable = app_dir()
    if _is_writable_dir(portable):
        return portable

    appdata = os.environ.get("APPDATA")
    if appdata:
        p = Path(appdata) / app_name
        if _is_writable_dir(p):
            return p

    p = Path.home() / ".config" / app_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_vendor_poppler_bin() -> str | None:
    """
    Return a poppler bin directory if shipped alongside the app, else None.

    Expected layouts (relative to app_dir()):
    - vendor/poppler/bin
    - vendor/poppler/Library/bin   (common in some Windows builds)
    """
    base = app_dir()
    candidates = [
        base / "vendor" / "poppler" / "bin",
        base / "vendor" / "poppler" / "Library" / "bin",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return None


def find_vendor_tesseract_exe() -> tuple[str | None, str | None]:
    """
    Return (tesseract_exe_path, tessdata_dir) if shipped alongside the app.

    Expected layout (relative to app_dir()):
    - vendor/tesseract/tesseract.exe
    - vendor/tesseract/tessdata/
    """
    base = app_dir()
    exe = base / "vendor" / "tesseract" / "tesseract.exe"
    tessdata = base / "vendor" / "tesseract" / "tessdata"
    exe_path = str(exe) if exe.exists() else None
    tessdata_path = str(tessdata) if tessdata.exists() and tessdata.is_dir() else None
    return exe_path, tessdata_path

