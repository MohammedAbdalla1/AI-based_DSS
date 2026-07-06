from __future__ import annotations

import os
from typing import Any

from config import load_environment

try:
    from google import genai
except Exception:  # pragma: no cover - import fallback for environments without the SDK
    genai = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - import fallback for environments without the SDK
    Groq = None


_gemini_client = None
_groq_client = None


class ProviderCallError(Exception):
    def __init__(self, provider: str, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.provider = provider
        self.subcode = subcode


def reset_client_cache() -> None:
    global _gemini_client, _groq_client
    _gemini_client = None
    _groq_client = None


def should_fallback_to_groq(subcode: str | None) -> bool:
    return subcode in {
        "MISSING_API_KEY",
        "CLIENT_INIT_FAILED",
        "PROVIDER_FAILURE",
        "EMPTY_RESPONSE",
        "INVALID_RESPONSE_TYPE",
        "RATE_LIMITED",
        "QUOTA_EXCEEDED",
        "MODEL_UNAVAILABLE",
        "SDK_MISSING",
    }


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                nested = item.get("content")
                if isinstance(nested, str):
                    parts.append(nested)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


def extract_total_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0

    for attribute_name in ("total_tokens", "total_token_count"):
        total = getattr(usage, attribute_name, None)
        if isinstance(total, int) and total >= 0:
            return total

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return max(0, prompt_tokens) + max(0, completion_tokens)

    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return max(0, prompt_tokens) + max(0, completion_tokens)

    return 0


def get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    load_environment()
    if genai is None:
        raise ProviderCallError("gemini", "Google GenAI SDK is unavailable", "SDK_MISSING")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ProviderCallError("gemini", "GEMINI_API_KEY is missing", "MISSING_API_KEY")

    try:
        _gemini_client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise ProviderCallError(
            "gemini",
            f"Gemini client initialization failed: {type(exc).__name__}",
            "CLIENT_INIT_FAILED",
        ) from exc

    return _gemini_client


def get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    load_environment()
    if Groq is None:
        raise ProviderCallError("groq", "Groq SDK is unavailable", "SDK_MISSING")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ProviderCallError("groq", "GROQ_API_KEY is missing", "MISSING_API_KEY")

    try:
        _groq_client = Groq(api_key=api_key)
    except Exception as exc:
        raise ProviderCallError(
            "groq",
            f"Groq client initialization failed: {type(exc).__name__}",
            "CLIENT_INIT_FAILED",
        ) from exc

    return _groq_client


def call_gemini_text(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProviderCallError("gemini", "Prompt is empty", "PROMPT_EMPTY")

    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
    except Exception as exc:
        raise ProviderCallError(
            "gemini",
            f"Gemini generation failed: {type(exc).__name__}",
            "PROVIDER_FAILURE",
        ) from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise ProviderCallError("gemini", "Invalid response type from Gemini.", "INVALID_RESPONSE_TYPE")

    cleaned = text.strip()
    if not cleaned:
        raise ProviderCallError("gemini", "Gemini returned an empty response.", "EMPTY_RESPONSE")
    return cleaned


def call_groq_completion(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
) -> tuple[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProviderCallError("groq", "Prompt is empty", "PROMPT_EMPTY")

    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as exc:
        raise ProviderCallError(
            "groq",
            f"Groq generation failed: {type(exc).__name__}",
            "PROVIDER_FAILURE",
        ) from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise ProviderCallError("groq", "Invalid response type from Groq.", "INVALID_RESPONSE_TYPE")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    text = _stringify_content(content).strip()
    if not text:
        raise ProviderCallError("groq", "Groq returned an empty response.", "EMPTY_RESPONSE")

    return text, response


def call_groq_text(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    text, _ = call_groq_completion(
        prompt,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return text
