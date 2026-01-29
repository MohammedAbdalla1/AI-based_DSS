import os
from google import genai
from dotenv import load_dotenv


load_dotenv()


class SQLGenerationError(Exception):
    """Raised when the LLM fails to generate SQL."""
    pass


def generate_sql(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns raw SQL text.
    This function does NOT validate or sanitize SQL.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SQLGenerationError("GEMINI_API_KEY is not set in environment variables")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
        )
    except Exception as e:
        raise SQLGenerationError(f"Gemini API call failed: {e}")

    if not response or not response.text:
        raise SQLGenerationError("Empty response from Gemini")

    # IMPORTANT: return output exactly as generated
    return response.text.strip()
