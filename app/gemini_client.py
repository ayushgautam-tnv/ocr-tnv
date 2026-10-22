import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from app.config import settings

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_MESSAGE_PATTERN = re.compile(
    r"high demand|try again later|temporarily unavailable|overloaded|rate limit|resource exhausted|quota",
    re.IGNORECASE,
)
MAX_TRANSIENT_ATTEMPTS_PER_MODEL = 2


class GeminiError(Exception):
    def __init__(self, model: str, status_code: int, message: str, transient: bool):
        super().__init__(message)
        self.model = model
        self.status_code = status_code
        self.message = message
        self.transient = transient


@dataclass
class GenerationMetadata:
    model: str
    primary_model: str
    used_fallback_model: bool
    models_tried: list[str]


def _model_candidates() -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for model in [settings.gemini_model, *settings.gemini_fallback_models]:
        model = model.strip()
        if model and model not in seen:
            seen.add(model)
            candidates.append(model)
    return candidates


def _is_transient(status_code: int, message: str) -> bool:
    return status_code in TRANSIENT_STATUS_CODES or bool(TRANSIENT_MESSAGE_PATTERN.search(message))


def _retry_delay_seconds(attempt_index: int) -> float:
    base = min(5.0, 0.8 * (2**attempt_index))
    return base + random.uniform(0, 0.25)


def _response_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _parse_json_text(text: str) -> Any | None:
    stripped = _strip_json_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace == -1 or last_brace <= first_brace:
            return None
        try:
            return json.loads(stripped[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            return None


class GeminiClient:
    def _api_key(self) -> str:
        if not settings.gemini_api_key:
            raise GeminiError("", 503, "GEMINI_API_KEY is not configured", False)
        return settings.gemini_api_key

    def _request_text(
        self,
        model: str,
        system_instruction: str,
        prompt: str,
        response_mime_type: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        response = requests.post(
            GEMINI_ENDPOINT.format(model=model),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key(),
            },
            json={
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "candidateCount": 1,
                    "maxOutputTokens": max_output_tokens,
                    "responseMimeType": response_mime_type,
                },
            },
            timeout=60,
        )

        try:
            body = response.json()
        except ValueError:
            body = None

        if not response.ok:
            message = (
                (body or {}).get("error", {}).get("message")
                or f"Gemini request failed with status {response.status_code}"
            )
            raise GeminiError(model, response.status_code, message, _is_transient(response.status_code, message))

        text = _response_text(body) if body else ""
        if not text:
            raise GeminiError(model, 502, "Gemini returned an empty response", True)
        return text

    def generate_text_with_metadata(
        self,
        system_instruction: str,
        prompt: str,
        response_mime_type: str = "text/plain",
        max_output_tokens: int | None = None,
        temperature: float = 0.1,
    ) -> tuple[str, GenerationMetadata]:
        models = _model_candidates()
        tried_models: list[str] = []
        last_error: GeminiError | None = None

        for model_index, model in enumerate(models):
            if model not in tried_models:
                tried_models.append(model)

            for attempt_index in range(MAX_TRANSIENT_ATTEMPTS_PER_MODEL):
                try:
                    text = self._request_text(
                        model,
                        system_instruction,
                        prompt,
                        response_mime_type,
                        max_output_tokens or settings.gemini_max_output_tokens,
                        temperature,
                    )
                    metadata = GenerationMetadata(
                        model=model,
                        primary_model=settings.gemini_model,
                        used_fallback_model=model != settings.gemini_model,
                        models_tried=list(tried_models),
                    )
                    return text, metadata
                except GeminiError as error:
                    last_error = error
                    can_retry_same_model = error.transient and attempt_index < MAX_TRANSIENT_ATTEMPTS_PER_MODEL - 1
                    can_try_fallback_model = error.transient and model_index < len(models) - 1

                    if not can_retry_same_model and not can_try_fallback_model:
                        raise

                    if can_retry_same_model:
                        time.sleep(_retry_delay_seconds(attempt_index))
                        continue

                    break

        if last_error:
            raise last_error
        raise GeminiError("", 503, "Gemini request failed", True)

    def generate_text(self, **kwargs: Any) -> str:
        text, _ = self.generate_text_with_metadata(**kwargs)
        return text

    def generate_json_with_metadata(
        self,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int | None = None,
        temperature: float = 0.1,
        retry_on_invalid_json: bool = True,
    ) -> tuple[Any, GenerationMetadata]:
        first_text, first_metadata = self.generate_text_with_metadata(
            system_instruction=system_instruction,
            prompt=prompt,
            response_mime_type="application/json",
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        parsed = _parse_json_text(first_text)
        if parsed is not None:
            return parsed, first_metadata

        if retry_on_invalid_json and (max_output_tokens or 0) < settings.gemini_max_output_tokens:
            retry_text, retry_metadata = self.generate_text_with_metadata(
                system_instruction=system_instruction,
                prompt=prompt,
                response_mime_type="application/json",
                max_output_tokens=settings.gemini_max_output_tokens,
                temperature=temperature,
            )
            retry_parsed = _parse_json_text(retry_text)
            if retry_parsed is not None:
                return retry_parsed, retry_metadata
            raise GeminiError(retry_metadata.model, 502, "Gemini returned invalid JSON after retry", False)

        raise GeminiError(first_metadata.model, 502, "Gemini returned invalid JSON", False)

    def generate_json(self, **kwargs: Any) -> Any:
        value, _ = self.generate_json_with_metadata(**kwargs)
        return value


gemini_client = GeminiClient()
