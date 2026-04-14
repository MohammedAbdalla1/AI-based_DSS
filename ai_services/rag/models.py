"""
Request and response models for the RAG API.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class RAGAskRequest(BaseModel):
    question: str = Field(..., description="Natural language question from the user.")
    user_id: str = Field(..., description="Trusted user identifier supplied by the main backend.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty or whitespace.")
        if len(value.split()) < 3:
            raise ValueError("Question must contain at least 3 words.")
        return value

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id cannot be empty or whitespace.")
        return value


class RAGQueryFilters(BaseModel):
    file_ids: list[str] = Field(default_factory=list)

    @field_validator("file_ids")
    @classmethod
    def validate_file_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if item and item not in seen:
                normalized.append(item)
                seen.add(item)
        return normalized


class RAGQueryRequest(BaseModel):
    tenant_id: str
    user_role: str
    query: str
    top_k: int = 5
    filters: RAGQueryFilters = Field(default_factory=RAGQueryFilters)

    @field_validator("tenant_id", "user_role", "query")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty or whitespace.")
        return value

    @field_validator("top_k", mode="before")
    @classmethod
    def clamp_top_k(cls, value: int | None) -> int:
        if value is None:
            return 5
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer.") from exc
        return min(max(numeric, 1), 20)


class RAGIngestMetadata(BaseModel):
    tenant_id: str = Field(..., description="Tenant identifier supplied by the trusted .NET backend.")
    uploaded_by: str = Field(..., description="User identifier of the uploader, supplied by the .NET backend.")
    uploader_role: str = Field(..., description="Role of the uploading user. Stored for metadata/auditing.")
    allowed_roles: list[str] = Field(..., description="Roles allowed to query this file.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional custom file metadata.")

    @field_validator("tenant_id", "uploaded_by", "uploader_role")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty or whitespace.")
        return value

    @field_validator("allowed_roles")
    @classmethod
    def validate_allowed_roles(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise ValueError("allowed_roles must contain strings.")
            item = value.strip()
            if not item:
                continue
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        if not normalized:
            raise ValueError("allowed_roles cannot be empty.")
        return normalized

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("metadata must be a JSON object.")
        return value


class RAGSource(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    chunk_index: int
    snippet: str
    score: Optional[float] = None


class RAGQuerySource(BaseModel):
    file_id: str
    file_name: str
    chunk_id: str
    relevance_score: float
    excerpt: str


class RAGAskSuccess(BaseModel):
    status: Literal["success"]
    answer: str
    sources: list[RAGSource]
    retrieved_chunks_count: int


class RAGQuerySuccess(BaseModel):
    answer: str
    sources: list[RAGQuerySource]
    tokens_used: int


class RAGIngestSuccess(BaseModel):
    file_id: str
    file_name: str
    chunks_created: int
    status: Literal["indexed", "already_indexed"]
    already_exists: bool = False


class RAGDeleteSuccess(BaseModel):
    file_id: str
    status: Literal["deleted"]


class RAGError(BaseModel):
    status: Literal["error"]
    error_code: str
    error_subcode: Optional[str] = None
    message: str


RAGAskResponse = Union[RAGAskSuccess, RAGError]
RAGQueryResponse = Union[RAGQuerySuccess, RAGError]
RAGIngestResponse = Union[RAGIngestSuccess, RAGError]
RAGDeleteResponse = Union[RAGDeleteSuccess, RAGError]
