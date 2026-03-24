import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from google import genai


class SQLGenerationError(Exception):
    """Raised when the LLM fails to generate SQL."""

    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


def _generation_error(subcode: str, message: str) -> SQLGenerationError:
    return SQLGenerationError(message=message, subcode=subcode)


@dataclass
class GenerationResult:
    sql: str
    explanation: Optional[str] = None


_client = None  # cached client


def _normalize_model_sql_text(text: str) -> str:
    raw = text.strip()

    fenced = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return raw


def _parse_json_response(text: str) -> Optional[GenerationResult]:
    """Try to parse the LLM response as JSON with sql + explanation fields."""
    raw = text.strip()

    # Strip markdown fences if present
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "sql" in parsed:
            sql = parsed["sql"].strip()
            explanation = parsed.get("explanation")
            if explanation and isinstance(explanation, str):
                explanation = explanation.strip()
            else:
                explanation = None
            if sql:
                return GenerationResult(sql=sql, explanation=explanation)
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


def _get_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise _generation_error("MISSING_API_KEY", "GEMINI_API_KEY is missing")

    try:
        _client = genai.Client(api_key=api_key)
    except Exception as e:
        raise _generation_error(
            "CLIENT_INIT_FAILED",
            f"Gemini client initialization failed: {type(e).__name__}",
        ) from e

    return _client


def generate_sql(prompt: str) -> GenerationResult:
    if not isinstance(prompt, str) or not prompt.strip():
        raise _generation_error("PROMPT_EMPTY", "Prompt is empty")

    client = _get_client()

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.1},
        )
    except Exception as e:
        raise _generation_error(
            "PROVIDER_FAILURE",
            f"Gemini generation failed: {type(e).__name__}",
        ) from e

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise _generation_error("INVALID_RESPONSE_TYPE", "Invalid response type from LLM")

    # Try JSON parsing first (new prompt format)
    json_result = _parse_json_response(text)
    if json_result:
        return json_result

    # Fallback: treat the whole response as raw SQL (backward compatibility)
    normalized = _normalize_model_sql_text(text)
    if not normalized:
        raise _generation_error("EMPTY_RESPONSE", "Empty response from LLM")

    return GenerationResult(sql=normalized, explanation=None)
