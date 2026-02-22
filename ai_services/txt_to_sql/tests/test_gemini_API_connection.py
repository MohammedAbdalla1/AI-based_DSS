import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google import genai


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@pytest.mark.live
def test_gemini_api_connection():
    load_dotenv(dotenv_path=_ENV_PATH)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip(f"GEMINI_API_KEY not found in {_ENV_PATH}")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents="Say hello in one sentence",
        )
    except Exception as exc:
        name = type(exc).__name__
        text = str(exc)
        if name in {"ClientError", "ServerError"} or "RESOURCE_EXHAUSTED" in text:
            pytest.skip(f"Live Gemini unavailable (quota/service): {name}")
        raise

    assert isinstance(response.text, str)
    assert response.text.strip()
