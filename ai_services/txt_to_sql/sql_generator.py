import os
import re
from google import genai


class SQLGenerationError(Exception):
    """Raised when the LLM fails to generate SQL."""
    pass


_client = None  # cached client


def _normalize_model_sql_text(text: str) -> str:
    raw = text.strip()

    # Handles:
    # ```sql
    # SELECT ...
    # ```
    # and
    # ```
    # SELECT ...
    # ```
    fenced = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", raw, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return raw


def _get_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SQLGenerationError("GEMINI_API_KEY is missing")

    try:
        _client = genai.Client(api_key=api_key)
    except Exception as e:
        raise SQLGenerationError(
            f"Gemini client initialization failed: {type(e).__name__}"
        ) from e

    return _client


def generate_sql(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise SQLGenerationError("Prompt is empty")

    client = _get_client()

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.1},
        )
    except Exception as e:
        raise SQLGenerationError(
            f"Gemini generation failed: {type(e).__name__}"
        ) from e

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise SQLGenerationError("Invalid response type from LLM")

    normalized = _normalize_model_sql_text(text)
    if not normalized:
        raise SQLGenerationError("Empty response from LLM")

    return normalized
