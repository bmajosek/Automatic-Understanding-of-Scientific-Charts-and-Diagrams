"""Reliable Tesseract paths on Windows (fixes intermittent tessdata errors)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TesseractSettings:
    cmd: str
    tessdata_dir: str
    ocr_config: str


def _normalize(path: str) -> str:
    return str(Path(path).resolve())


def _windows_short_path(path: str) -> str:
    """8.3 short path avoids quoting bugs in Tesseract --tessdata-dir on Windows."""
    if os.name != "nt":
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512):
            return buf.value
    except Exception:
        pass
    return path


def resolve_tesseract(
    tesseract_cmd: Optional[str] = None,
    tessdata_prefix: Optional[str] = None,
) -> TesseractSettings:
    """Resolve executable and tessdata directory; verify eng.traineddata exists."""
    cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD", "")
    if not cmd:
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if Path(candidate).is_file():
                cmd = candidate
                break
    if not cmd:
        raise RuntimeError(
            "Tesseract not found. Install Tesseract OCR and set TESSERACT_CMD in .env "
            "or pass --tesseract-cmd."
        )
    cmd = _normalize(cmd)

    tessdata = tessdata_prefix or os.environ.get("TESSDATA_PREFIX", "")
    if not tessdata:
        tessdata = str(Path(cmd).parent / "tessdata")
    tessdata_path = Path(tessdata)
    if tessdata_path.name != "tessdata":
        candidate = tessdata_path / "tessdata"
        if candidate.is_dir():
            tessdata_path = candidate
    if not tessdata_path.is_dir():
        raise RuntimeError(f"Tessdata directory not found: {tessdata_path}")

    eng = tessdata_path / "eng.traineddata"
    if not eng.is_file():
        raise RuntimeError(
            f"Missing {eng}. Install English language pack for Tesseract."
        )

    tessdata_dir = _normalize(str(tessdata_path))
    tessdata_for_cli = _windows_short_path(tessdata_dir).replace("\\", "/")

    os.environ["TESSERACT_CMD"] = cmd
    # Must point at the tessdata folder (where eng.traineddata lives).
    os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return TesseractSettings(
        cmd=cmd,
        tessdata_dir=tessdata_dir,
        # Short path on Windows avoids quoting bugs in --tessdata-dir.
        ocr_config=f"--psm 6 --tessdata-dir {tessdata_for_cli}",
    )


def apply_pytesseract(settings: TesseractSettings) -> None:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.cmd
