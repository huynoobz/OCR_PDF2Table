"""
Module 3: OCR (Tesseract)

Pure logic module. No UI code.

Responsibilities:
- Provide OCR helpers (Tesseract via pytesseract) that can be used by Module 2.
- Keep OCR concerns out of Module 1 (PDF/image preprocessing) and out of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class OCRConfig:
    lang: str = "eng"
    psm: int = 6
    oem: Optional[int] = None
    config_extra: str = ""


def ocr_image_pil(pil_image: Image.Image, *, config: OCRConfig) -> str:
    """
    Run Tesseract OCR on a PIL image.

    Requires:
    - `pytesseract` package
    - Tesseract OCR installed on the system (and available on PATH, or configured by pytesseract)
    """
    try:
        import pytesseract  # type: ignore
    except Exception as e:
        raise RuntimeError("pytesseract is not installed. Run `pip install pytesseract`.") from e

    cfg = f"--psm {int(config.psm)}"
    if config.oem is not None:
        cfg += f" --oem {int(config.oem)}"
    if config.config_extra:
        cfg += f" {config.config_extra}"

    text = pytesseract.image_to_string(pil_image, lang=config.lang, config=cfg)
    return text

