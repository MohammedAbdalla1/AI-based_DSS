"""
Qdrant integration for RAG chunk storage and filtered retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from config import get_settings


class VectorStoreError(Exception):
    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


@dataclass(frozen=True)
class ChunkVectorRecord:
    document_id: str
    file_name: str
    chunk_id: str
    chunk_index: int
    text: str
    vector: list[float]
    source_type: str = "pdf"


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    file_name: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class TenantChunkVectorRecord:
    tenant_id: str
    file_id: str
    file_name: str
    chunk_id: str
    chunk_index: int
    text: str
    vector: list[float]
    allowed_roles: list[str]
    source_type: str


@dataclass(frozen=True)
class TenantRetrievedChunk:
    file_id: str
    file_name: str
    chunk_id: str
    chunk_index: int
    text: str
    text_preview: str
    allowed_roles: list[str]
    score: float


_client = None
_models = None


def _vector_error(subcode: str, message: str) -> VectorStoreError:
    return VectorStoreError(message=message, subcode=subcode)


def _get_client_and_models():
    global _client, _models
    if _client is not None and _models is not None:
        return _client, _models

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ModuleNotFoundError as exc:
        raise _vector_error(
            "CLIENT_NOT_INSTALLED",
            "qdrant-client is not installed.",
        ) from exc

    settings = get_settings()

    qdrant_url = getattr(settings, "qdrant_url", "").strip()
    qdrant_api_key = getattr(settings, "qdrant_api_key", "").strip()

    if qdrant_url:
        # Qdrant Cloud or self-hosted Docker server
        _client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
        )
    else:
        # Local embedded file mode (dev / testing)
        settings.ensure_runtime_dirs()
        _client = QdrantClient(path=str(settings.rag_qdrant_path))

    _models = models
    return _client, _models


def ensure_collection(vector_size: int) -> None:
    settings = get_settings()
    client, models = _get_client_and_models()
    try:
        client.get_collection(settings.rag_qdrant_collection)
    except Exception:
        try:
            client.create_collection(
                collection_name=settings.rag_qdrant_collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        except Exception as exc:
            raise _vector_error(
                "COLLECTION_INIT_FAILED",
                f"Failed to initialize Qdrant collection: {type(exc).__name__}",
            ) from exc


def _tenant_collection_name(tenant_id: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in tenant_id)
    return f"tenant_{sanitized}_chunks"


def ensure_tenant_collection(tenant_id: str, vector_size: int) -> str:
    collection_name = _tenant_collection_name(tenant_id)
    client, models = _get_client_and_models()
    try:
        client.get_collection(collection_name)
    except Exception:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        except Exception as exc:
            raise _vector_error(
                "COLLECTION_INIT_FAILED",
                f"Failed to initialize tenant Qdrant collection: {type(exc).__name__}: {exc}",
            ) from exc
    return collection_name


def upsert_chunks(records: list[ChunkVectorRecord]) -> None:
    if not records:
        return

    settings = get_settings()
    client, models = _get_client_and_models()
    ensure_collection(len(records[0].vector))

    points = [
        models.PointStruct(
            id=record.chunk_id,
            vector=record.vector,
            payload={
                "document_id": record.document_id,
                "file_name": record.file_name,
                "chunk_id": record.chunk_id,
                "chunk_index": record.chunk_index,
                "source_type": record.source_type,
                "text": record.text,
            },
        )
        for record in records
    ]

    try:
        client.upsert(collection_name=settings.rag_qdrant_collection, points=points)
    except Exception as exc:
        raise _vector_error(
            "UPSERT_FAILED",
            f"Failed to upsert chunks into Qdrant: {type(exc).__name__}",
        ) from exc


def upsert_tenant_chunks(records: list[TenantChunkVectorRecord]) -> None:
    if not records:
        return

    client, models = _get_client_and_models()
    collection_name = ensure_tenant_collection(records[0].tenant_id, len(records[0].vector))

    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{record.tenant_id}:{record.chunk_id}")),
            vector=record.vector,
            payload={
                "tenant_id": record.tenant_id,
                "file_id": record.file_id,
                "file_name": record.file_name,
                "chunk_id": record.chunk_id,
                "chunk_index": record.chunk_index,
                "allowed_roles": record.allowed_roles,
                "text_preview": record.text[:300],
                "text": record.text,
                "source_type": record.source_type,
            },
        )
        for record in records
    ]

    try:
        client.upsert(collection_name=collection_name, points=points)
    except Exception as exc:
        raise _vector_error(
            "UPSERT_FAILED",
            f"Failed to upsert chunks into tenant Qdrant collection: {type(exc).__name__}: {exc}",
        ) from exc


def delete_file_chunks(tenant_id: str, file_id: str) -> None:
    client, models = _get_client_and_models()
    collection_name = _tenant_collection_name(tenant_id)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )
    except Exception:
        return


def search_tenant_chunks(
    *,
    tenant_id: str,
    query_vector: list[float],
    limit: int,
    file_ids: list[str] | None = None,
) -> list[TenantRetrievedChunk]:
    client, models = _get_client_and_models()
    collection_name = _tenant_collection_name(tenant_id)

    # If this tenant has never ingested a file, the collection won't exist yet.
    # Return empty results instead of raising an error.
    try:
        client.get_collection(collection_name)
    except Exception:
        return []

    must_conditions = []
    if file_ids:
        must_conditions.append(
            models.FieldCondition(
                key="file_id",
                match=models.MatchAny(any=file_ids),
            )
        )
    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        raise _vector_error(
            "SEARCH_FAILED",
            f"Failed to search tenant Qdrant collection: {type(exc).__name__}: {exc}",
        ) from exc

    chunks: list[TenantRetrievedChunk] = []
    for result in response.points:
        payload = getattr(result, "payload", {}) or {}
        allowed_roles = payload.get("allowed_roles", []) or []
        if not isinstance(allowed_roles, list):
            allowed_roles = []
        chunks.append(
            TenantRetrievedChunk(
                file_id=str(payload.get("file_id", "")),
                file_name=str(payload.get("file_name", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                text_preview=str(payload.get("text_preview", "")),
                allowed_roles=[str(role) for role in allowed_roles if str(role).strip()],
                score=float(getattr(result, "score", 0.0)),
            )
        )

    return chunks


def search_chunks(query_vector: list[float], allowed_document_ids: list[str], limit: int) -> list[RetrievedChunk]:
    if not allowed_document_ids:
        return []

    settings = get_settings()
    client, models = _get_client_and_models()
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=allowed_document_ids),
            )
        ]
    )

    try:
        response = client.query_points(
            collection_name=settings.rag_qdrant_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        raise _vector_error(
            "SEARCH_FAILED",
            f"Failed to search Qdrant: {type(exc).__name__}",
        ) from exc

    chunks: list[RetrievedChunk] = []
    for result in response.points:
        payload = getattr(result, "payload", {}) or {}
        chunks.append(
            RetrievedChunk(
                document_id=str(payload.get("document_id", "")),
                file_name=str(payload.get("file_name", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                score=float(getattr(result, "score", 0.0)),
            )
        )

    return chunks
