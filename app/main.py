import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.gemini_client import GeminiError
from app.ocr import extract_text
from app.scope import suggest_scope

app = FastAPI(title="OCR Scope Suggestion Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/suggest-scope")
async def suggest_scope_endpoint(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        try:
            ocr_text = extract_text(tmp.name)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"OCR failed: {error}") from error

        try:
            result = suggest_scope(ocr_text)
        except GeminiError as error:
            raise HTTPException(status_code=502, detail=error.message) from error

    return {
        "file_name": file.filename,
        "ocr_text_length": len(ocr_text),
        **result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
