"""
Authorized retrieval orchestration for RAG.
"""

from __future__ import annotations

from config import get_settings

from . import generator, vector_store
from .vector_store import RetrievedChunk, TenantRetrievedChunk


def retrieve_authorized_chunks(question: str, allowed_document_ids: list[str]) -> list[RetrievedChunk]:
    settings = get_settings()
    query_vector = generator.embed_query(question)
    return vector_store.search_chunks(
        query_vector=query_vector,
        allowed_document_ids=allowed_document_ids,
        limit=settings.rag_top_k,
    )


def retrieve_tenant_chunks(
    *,
    tenant_id: str,
    query: str,
    top_k: int,
    file_ids: list[str] | None = None,
) -> list[TenantRetrievedChunk]:
    query_vector = generator.embed_query(query)
    return vector_store.search_tenant_chunks(
        tenant_id=tenant_id,
        query_vector=query_vector,
        limit=top_k,
        file_ids=file_ids,
    )
