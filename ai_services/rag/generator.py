"""
Gemini embedding and answer-generation helpers for RAG.
"""

from __future__ import annotations

import os
from typing import Iterable

from config import get_settings
from llm_clients import (
    ProviderCallError,
    call_groq_completion,
    get_gemini_client,
    should_fallback_to_groq,
)


FALLBACK_ANSWER = "I don't know based on the authorized documents."


class GeminiRAGError(Exception):
    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


def _rag_error(subcode: str, message: str) -> GeminiRAGError:
    return GeminiRAGError(message=message, subcode=subcode)


def _groq_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


def _get_gemini_client():
    try:
        return get_gemini_client()
    except ProviderCallError as exc:
        raise _rag_error(exc.subcode, str(exc)) from exc


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    items = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not items:
        raise _rag_error("EMPTY_INPUT", "Embedding input is empty.")

    settings = get_settings()
    client = _get_gemini_client()
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


def _generate_answer_with_gemini(question: str, contexts: list[str]) -> tuple[str, int]:
    settings = get_settings()
    client = _get_gemini_client()
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


def _generate_answer_with_groq(question: str, contexts: list[str]) -> tuple[str, int]:
    prompt = build_answer_prompt(question, contexts)
    try:
        text, response = call_groq_completion(
            prompt,
            model=_groq_model(),
            temperature=0.1,
            max_completion_tokens=8192,
        )
    except ProviderCallError as exc:
        raise _rag_error(exc.subcode, str(exc)) from exc

    usage = getattr(response, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    return text, total_tokens


def generate_answer(question: str, contexts: list[str]) -> str:
    answer, _ = generate_answer_with_usage(question, contexts)
    return answer


def generate_answer_with_usage(question: str, contexts: list[str]) -> tuple[str, int]:
    if not contexts:
        return FALLBACK_ANSWER, 0

    try:
        return _generate_answer_with_gemini(question, contexts)
    except GeminiRAGError as gemini_exc:
        if not should_fallback_to_groq(gemini_exc.subcode):
            raise

        try:
            answer, tokens_used = _generate_answer_with_groq(question, contexts)
            return answer, tokens_used
        except GeminiRAGError as groq_exc:
            raise _rag_error(
                gemini_exc.subcode if gemini_exc.subcode != "UNKNOWN" else groq_exc.subcode,
                f"{gemini_exc} | Groq fallback failed: {groq_exc}",
            ) from groq_exc
