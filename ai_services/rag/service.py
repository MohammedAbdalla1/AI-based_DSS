"""
Orchestration layer for RAG ingestion and question-answering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from pydantic import ValidationError

from config import get_settings

from . import access_control, database, generator, ingestion, rate_limit, retriever, vector_store
from .models import (
    RAGAskRequest,
    RAGAskResponse,
    RAGDeleteResponse,
    RAGDeleteSuccess,
    RAGAskSuccess,
    RAGError,
    RAGIngestMetadata,
    RAGIngestResponse,
    RAGIngestSuccess,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGQuerySource,
    RAGQuerySuccess,
    RAGSource,
)

def _error(error_code: str, message: str, error_subcode: Optional[str] = None) -> RAGError:
    return RAGError(
        status="error",
        error_code=error_code,
        error_subcode=error_subcode,
        message=message,
    )


def _fallback_success() -> RAGAskSuccess:
    return RAGAskSuccess(
        status="success",
        answer=generator.FALLBACK_ANSWER,
        sources=[],
        retrieved_chunks_count=0,
    )


def _build_sources(chunks: list[vector_store.RetrievedChunk]) -> list[RAGSource]:
    return [
        RAGSource(
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            snippet=chunk.text[:280],
            score=chunk.score,
        )
        for chunk in chunks
    ]


def _build_query_sources(chunks: list[vector_store.TenantRetrievedChunk]) -> list[RAGQuerySource]:
    return [
        RAGQuerySource(
            file_id=chunk.file_id,
            file_name=chunk.file_name,
            chunk_id=chunk.chunk_id,
            relevance_score=round(chunk.score, 4),
            excerpt=(chunk.text_preview or chunk.text)[:300],
        )
        for chunk in chunks
    ]


def run_rag_ask(request_or_payload: RAGAskRequest | Mapping[str, Any]) -> RAGAskResponse:
    try:
        if isinstance(request_or_payload, RAGAskRequest):
            request = request_or_payload
        else:
            request = RAGAskRequest(**dict(request_or_payload))
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc), error_subcode="PAYLOAD_VALIDATION")
    except Exception as exc:
        return _error(
            "INVALID_REQUEST",
            f"Invalid request payload type: {type(exc).__name__}",
            error_subcode="PAYLOAD_TYPE",
        )

    try:
        allowed_document_ids = access_control.get_allowed_document_ids(request.user_id)
        if not allowed_document_ids:
            return _fallback_success()

        chunks = retriever.retrieve_authorized_chunks(request.question, allowed_document_ids)
        if not chunks:
            return _fallback_success()

        settings = get_settings()
        best_score = max(chunk.score for chunk in chunks)
        if best_score < settings.rag_relevance_threshold:
            return _fallback_success()

        answer = generator.generate_answer(
            request.question,
            [chunk.text for chunk in chunks],
        )

        if answer.strip() == generator.FALLBACK_ANSWER:
            return _fallback_success()

        return RAGAskSuccess(
            status="success",
            answer=answer.strip(),
            sources=_build_sources(chunks),
            retrieved_chunks_count=len(chunks),
        )
    except vector_store.VectorStoreError as exc:
        return _error("RETRIEVAL_FAILED", str(exc), error_subcode=exc.subcode)
    except generator.GeminiRAGError as exc:
        return _error("GENERATION_FAILED", str(exc), error_subcode=exc.subcode)
    except Exception as exc:
        return _error(
            "INTERNAL_ERROR",
            f"Unexpected error: {type(exc).__name__}",
            error_subcode="UNEXPECTED_EXCEPTION",
        )


def run_rag_query(request_or_payload: RAGQueryRequest | Mapping[str, Any]) -> RAGQueryResponse:
    try:
        if isinstance(request_or_payload, RAGQueryRequest):
            request = request_or_payload
        else:
            request = RAGQueryRequest(**dict(request_or_payload))
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc), error_subcode="PAYLOAD_VALIDATION")
    except Exception as exc:
        return _error(
            "INVALID_REQUEST",
            f"Invalid request payload type: {type(exc).__name__}",
            error_subcode="PAYLOAD_TYPE",
        )

    try:
        rate_limit.enforce_tenant_rate_limit(request.tenant_id)

        raw_results = retriever.retrieve_tenant_chunks(
            tenant_id=request.tenant_id,
            query=request.query,
            top_k=request.top_k,
            file_ids=request.filters.file_ids or None,
        )
        if not raw_results:
            return _error("NOT_FOUND", "No relevant documents found.", error_subcode="NO_RESULTS")

        authorized_chunks = [
            chunk for chunk in raw_results if request.user_role in chunk.allowed_roles
        ]
        if not authorized_chunks:
            return _error(
                "ACCESS_DENIED",
                "You do not have permission to access the relevant documents.",
                error_subcode="ROLE_FILTER_EMPTY",
            )

        answer, tokens_used = generator.generate_answer_with_usage(
            request.query,
            [chunk.text for chunk in authorized_chunks],
        )
        return RAGQuerySuccess(
            answer=answer.strip(),
            sources=_build_query_sources(authorized_chunks),
            tokens_used=tokens_used,
        )
    except rate_limit.RateLimitError:
        return _error("RATE_LIMITED", "Rate limit exceeded.", error_subcode="TENANT_RATE_LIMIT")
    except vector_store.VectorStoreError as exc:
        return _error("RETRIEVAL_FAILED", "Failed to query documents.", error_subcode=exc.subcode)
    except generator.GeminiRAGError as exc:
        return _error("GENERATION_FAILED", "Failed to generate answer.", error_subcode=exc.subcode)
    except Exception:
        return _error(
            "INTERNAL_ERROR",
            "Failed to process query.",
            error_subcode="UNEXPECTED_EXCEPTION",
        )


async def run_rag_ingest(uploaded_file, metadata_or_payload: RAGIngestMetadata | Mapping[str, Any]) -> RAGIngestResponse:
    if uploaded_file is None:
        return _error("INVALID_REQUEST", "file: Field required", error_subcode="PAYLOAD_VALIDATION")

    try:
        if isinstance(metadata_or_payload, RAGIngestMetadata):
            metadata = metadata_or_payload
        else:
            metadata = RAGIngestMetadata(**dict(metadata_or_payload))
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc), error_subcode="PAYLOAD_VALIDATION")
    except Exception as exc:
        return _error(
            "INVALID_REQUEST",
            f"Invalid ingest payload type: {type(exc).__name__}",
            error_subcode="PAYLOAD_TYPE",
        )

    file_id = str(uuid4())
    persisted_storage_path = None
    storage_reference = None
    try:
        contents, file_name, ext, mime_type = await ingestion.read_and_validate_upload(uploaded_file)
        text = ingestion.extract_text(contents, ext)

        # Hash normalized extracted text (not raw bytes) so semantically identical
        # content deduplicates even when binary file metadata differs.
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        existing_file = database.get_file_by_content_hash(
            tenant_id=metadata.tenant_id,
            content_hash=content_hash,
        )
        if existing_file is not None:
            return RAGIngestSuccess(
                file_id=str(existing_file["id"]),
                file_name=str(existing_file["file_name"]),
                chunks_created=int(existing_file["chunks_count"]),
                status="already_indexed",
                already_exists=True,
            )

        settings = get_settings()
        if settings.rag_persist_source_files:
            persisted_storage_path = ingestion.save_to_storage(
                tenant_id=metadata.tenant_id,
                file_id=file_id,
                file_name=file_name,
                contents=contents,
            )
        storage_reference = ingestion.build_storage_reference(
            tenant_id=metadata.tenant_id,
            file_id=file_id,
            file_name=file_name,
            persisted_path=persisted_storage_path,
        )
        chunks = ingestion.chunk_text(text)

        embeddings = generator.embed_texts(chunk.text for chunk in chunks)
        vector_records = [
            vector_store.TenantChunkVectorRecord(
                tenant_id=metadata.tenant_id,
                file_id=file_id,
                file_name=file_name,
                chunk_id=f"{file_id}_chunk_{chunk.chunk_index}",
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=embeddings[index],
                allowed_roles=metadata.allowed_roles,
                source_type=ext.lstrip("."),
            )
            for index, chunk in enumerate(chunks)
        ]
        # Insert the file record into SQLite first.
        # If the subsequent Qdrant upsert fails, we delete this record
        # so we don't end up with a SQLite entry that has no vectors.
        database.insert_file_record(
            file_id=file_id,
            tenant_id=metadata.tenant_id,
            uploaded_by=metadata.uploaded_by,
            file_name=file_name,
            storage_path=storage_reference,
            allowed_roles=metadata.allowed_roles,
            metadata=metadata.metadata,
            content_hash=content_hash,
            chunks_count=len(chunks),
        )
        try:
            vector_store.upsert_tenant_chunks(vector_records)
        except Exception:
            database.delete_file_record(file_id)
            raise
        return RAGIngestSuccess(
            file_id=file_id,
            file_name=file_name,
            chunks_created=len(chunks),
            status="indexed",
        )
    except ingestion.DocumentIngestionError as exc:
        return _error(_map_ingest_error_code(exc.subcode), str(exc), error_subcode=exc.subcode)
    except vector_store.VectorStoreError as exc:
        if persisted_storage_path is not None:
            ingestion.delete_saved_file(persisted_storage_path)
        return _error("INGEST_FAILED", str(exc), error_subcode=exc.subcode)
    except generator.GeminiRAGError as exc:
        if persisted_storage_path is not None:
            ingestion.delete_saved_file(persisted_storage_path)
        return _error("INGEST_FAILED", "Failed to index file.", error_subcode=exc.subcode)
    except Exception as exc:
        if persisted_storage_path is not None:
            vector_store.delete_file_chunks(metadata.tenant_id, file_id)
            ingestion.delete_saved_file(persisted_storage_path)
        return _error(
            "INTERNAL_ERROR",
            "Failed to index file.",
            error_subcode="UNEXPECTED_EXCEPTION",
        )


def run_rag_delete_file(*, tenant_id: str, file_id: str) -> RAGDeleteResponse:
    tenant_id = tenant_id.strip()
    file_id = file_id.strip()

    if not tenant_id:
        return _error("INVALID_REQUEST", "tenant_id cannot be empty.", error_subcode="PAYLOAD_VALIDATION")
    if not file_id:
        return _error("INVALID_REQUEST", "file_id cannot be empty.", error_subcode="PAYLOAD_VALIDATION")

    try:
        file_record = database.get_file_record(file_id=file_id, tenant_id=tenant_id)
        if file_record is None:
            return _error("NOT_FOUND", "File not found for this tenant.", error_subcode="FILE_NOT_FOUND")

        vector_store.delete_file_chunks(tenant_id=tenant_id, file_id=file_id)

        storage_path = str(file_record["storage_path"] or "")
        if storage_path and not storage_path.startswith("transient://"):
            ingestion.delete_saved_file(Path(storage_path))

        database.delete_file_record(file_id, tenant_id=tenant_id)

        return RAGDeleteSuccess(file_id=file_id, status="deleted")
    except Exception:
        return _error(
            "INTERNAL_ERROR",
            "Failed to delete file.",
            error_subcode="UNEXPECTED_EXCEPTION",
        )


def _map_ingest_error_code(subcode: str) -> str:
    if subcode in {"UNSUPPORTED_FILE_TYPE", "EMPTY_FILE", "EMPTY_DOCUMENT", "INVALID_CHUNK_CONFIG", "TEXT_EXTRACTION_FAILED"}:
        return "INVALID_REQUEST"
    if subcode == "FILE_TOO_LARGE":
        return "PAYLOAD_TOO_LARGE"
    return "INGEST_FAILED"
