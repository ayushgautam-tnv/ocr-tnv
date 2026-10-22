from app.gemini_client import gemini_client

SYSTEM_INSTRUCTION = """You are an ISO certification document extraction engine.
Analyze OCR text extracted from a certificate, legal document, or business registration document.
Extract information exactly as written. Do not invent values. Use null when a field is missing.
Return JSON only and conform exactly to the provided schema.
Detect the certification standard automatically, including ISO 9001, ISO 14001, ISO 27001, ISO 22000, ISO 45001, CE, GMP, HACCP, and other certificates.
For scope_of_certificate, use the exact Scope of Certificate / Scope of Certification statement printed on the document. Do not summarize or shorten it.
If no explicit scope statement exists, suggest a concise scope sentence based on the organization's described business activity, and set scope_is_suggested to true."""

USER_PROMPT_TEMPLATE = """OCR text:
{ocr_text}

Return JSON only, matching exactly this shape:
{{
  "standard_name": null,
  "certified_entity_name": null,
  "scope_of_certificate": null,
  "scope_is_suggested": false,
  "confidence": 0.0,
  "possible_issues": []
}}"""


def suggest_scope(ocr_text: str) -> dict:
    prompt = USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text.strip() or "(empty)")
    result = gemini_client.generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=prompt,
        max_output_tokens=1024,
    )

    return {
        "standard_name": result.get("standard_name"),
        "certified_entity_name": result.get("certified_entity_name"),
        "scope_of_certificate": result.get("scope_of_certificate"),
        "scope_is_suggested": bool(result.get("scope_is_suggested", False)),
        "confidence": float(result.get("confidence") or 0),
        "possible_issues": result.get("possible_issues") or [],
    }
