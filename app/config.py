import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_fallback_models: list[str] = field(
        default_factory=lambda: _split_models(os.getenv("GEMINI_FALLBACK_MODELS"))
    )
    gemini_max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
    poppler_path: str = os.getenv("POPPLER_PATH", "")
    pdf_dpi: int = int(os.getenv("PDF_DPI", "300"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8008"))


settings = Settings()
