"""
Gemini embedding and answer-generation helpers for RAG.
"""

from __future__ import annotations

import os
from typing import Iterable

from config import get_settings, load_environment


FALLBACK_ANSWER = "I don't know based on the authorized documents."
_client = None


class GeminiRAGError(Exception):
    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


def _rag_error(subcode: str, message: str) -> GeminiRAGError:
    return GeminiRAGError(message=message, subcode=subcode)


def _get_client():
    global _client
    if _client is not None:
        return _client

    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise _rag_error("MISSING_API_KEY", "GEMINI_API_KEY is missing")

    try:
        from google import genai

        _client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise _rag_error(
            "CLIENT_INIT_FAILED",
            f"Gemini client initialization failed: {type(exc).__name__}",
        ) from exc

    return _client


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not items:
        raise _rag_error("EMPTY_INPUT", "Embedding input is empty.")

    settings = get_settings()
    client = _get_client()
    try:
        response = client.models.embed_content(
            model=settings.rag_embedding_model,
            contents=items,
        )
    except Exception as exc:
        raise _rag_error(
            "EMBEDDING_FAILURE",
            f"Gemini embedding failed: {type(exc).__name__}",
        ) from exc

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise _rag_error("EMPTY_RESPONSE", "Gemini embedding response was empty.")

    vectors: list[list[float]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if not isinstance(values, list) or not values:
            raise _rag_error("INVALID_RESPONSE_TYPE", "Gemini embedding response was invalid.")
        vectors.append([float(value) for value in values])

    return vectors


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]


def build_answer_prompt(question: str, contexts: list[str]) -> str:
    joined_context = "\n\n---\n\n".join(contexts)
    return (
        "You are answering questions for a document-restricted RAG system.\n"
        "Answer only from the retrieved context.\n"
        "If the answer is not contained in the retrieved context, reply exactly with:\n"
        f"{FALLBACK_ANSWER}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{joined_context}\n"
    )


def generate_answer(question: str, contexts: list[str]) -> str:
    answer, _ = generate_answer_with_usage(question, contexts)
    return answer


def generate_answer_with_usage(question: str, contexts: list[str]) -> tuple[str, int]:
    if not contexts:
        return FALLBACK_ANSWER, 0

    settings = get_settings()
    client = _get_client()
    prompt = build_answer_prompt(question, contexts)

    try:
        response = client.models.generate_content(
            model=settings.rag_generation_model,
            contents=prompt,
            config={"temperature": 0.1},
        )
    except Exception as exc:
        raise _rag_error(
            "PROVIDER_FAILURE",
            f"Gemini generation failed: {type(exc).__name__}",
        ) from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        raise _rag_error("INVALID_RESPONSE_TYPE", "Invalid response type from Gemini.")

    answer = text.strip()
    if not answer:
        raise _rag_error("EMPTY_RESPONSE", "Gemini returned an empty answer.")
    usage = getattr(response, "usage_metadata", None)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    return answer, total_tokens
