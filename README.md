# OCR Scope Suggestion Service

Standalone Python service. OCRs an uploaded certificate/registration document with Tesseract, then asks Gemini to find (or, if none is printed, suggest) the Scope of Certificate — the same Gemini usage pattern (model + fallback models, retries, JSON responses) as `tnvmis/OCR`.

## Setup

```bash
brew install tesseract poppler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY` in `.env`.

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8008
```

## API

`POST /suggest-scope` — multipart form field `file` (PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP).

```json
{
  "file_name": "certificate.pdf",
  "ocr_text_length": 1284,
  "standard_name": "ISO 9001:2015",
  "certified_entity_name": "Example Manufacturing Pvt Ltd",
  "scope_of_certificate": "Quality management for widget manufacturing.",
  "scope_is_suggested": false,
  "confidence": 0.86,
  "possible_issues": []
}
```

`scope_is_suggested: true` means no scope statement was printed on the document and Gemini inferred one from the described business activity instead.
