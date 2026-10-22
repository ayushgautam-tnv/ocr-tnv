from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from app.config import settings

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_text(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in PDF_SUFFIXES:
        return _extract_from_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return _extract_from_image(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_from_image(path: Path) -> str:
    with Image.open(path) as image:
        return pytesseract.image_to_string(image)


def _extract_from_pdf(path: Path) -> str:
    kwargs = {"dpi": settings.pdf_dpi}
    if settings.poppler_path:
        kwargs["poppler_path"] = settings.poppler_path
    pages = convert_from_path(str(path), **kwargs)
    texts = [pytesseract.image_to_string(page) for page in pages]
    return "\n".join(texts)
