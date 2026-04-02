"""
FastAPI router layer for Retrieval-Augmented Generation (RAG).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import get_settings, load_environment

from .models import RAGError, RAGIngestResponse, RAGQueryRequest, RAGQueryResponse
from .service import run_rag_ingest, run_rag_query


load_environment()

router = APIRouter(tags=["rag"])


ERROR_STATUS_MAP = {
    "INVALID_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "ACCESS_DENIED": 403,
    "PAYLOAD_TOO_LARGE": 413,
    "NOT_FOUND": 404,
    "RATE_LIMITED": 429,
    "INGEST_FAILED": 500,
    "RETRIEVAL_FAILED": 502,
    "GENERATION_FAILED": 502,
    "INTERNAL_ERROR": 500,
}


def _status_for_error(error: RAGError) -> int:
    return ERROR_STATUS_MAP.get(error.error_code, 500)


def request_validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request payload."

    first = errors[0]
    loc = [str(part) for part in first.get("loc", []) if part != "body"]
    path = ".".join(loc)
    msg = first.get("msg", "Invalid request payload.")
    if path:
        return f"{path}: {msg}"
    return str(msg)


def request_validation_error_response(exc: RequestValidationError) -> JSONResponse:
    error = RAGError(
        status="error",
        error_code="INVALID_REQUEST",
        error_subcode="PAYLOAD_VALIDATION",
        message=request_validation_message(exc),
    )
    return JSONResponse(status_code=400, content=error.model_dump())


def verify_internal_key(x_internal_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    if not settings.enforce_internal_api_key:
        return
    expected = settings.internal_api_key
    if not expected or x_internal_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _parse_allowed_roles(raw: str) -> list[str]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("allowed_roles must be a JSON array of strings.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        parsed = None

    if isinstance(parsed, list):
        roles = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        roles = [item.strip() for item in stripped.split(",") if item.strip()]

    if not roles:
        raise ValueError("allowed_roles must be a non-empty JSON array of strings.")
    return roles


def _parse_metadata(raw: str | None) -> dict:
    if raw is None or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("metadata must be a JSON object.")
    return parsed


@router.post(
    "/api/rag/query",
    response_model=RAGQueryResponse,
    responses={
        400: {"model": RAGError, "description": "Invalid query payload"},
        401: {"description": "Unauthorized"},
        403: {"model": RAGError, "description": "Forbidden"},
        404: {"model": RAGError, "description": "No relevant documents found"},
        429: {"model": RAGError, "description": "Rate limit exceeded"},
        500: {"model": RAGError, "description": "Query failure"},
    },
)
def query_rag(request: RAGQueryRequest, _: None = Depends(verify_internal_key)):
    result = run_rag_query(request)
    if isinstance(result, RAGError):
        return JSONResponse(status_code=_status_for_error(result), content=result.model_dump())
    return result


@router.post("/rag/ask", include_in_schema=False)
def ask_rag_legacy_alias(request: RAGQueryRequest, _: None = Depends(verify_internal_key)):
    return query_rag(request, _)


@router.post(
    "/api/rag/ingest",
    response_model=RAGIngestResponse,
    responses={
        401: {"model": RAGError, "description": "Unauthorized"},
        403: {"model": RAGError, "description": "Forbidden"},
        413: {"model": RAGError, "description": "File too large"},
        400: {"model": RAGError, "description": "Invalid ingest payload"},
        500: {"model": RAGError, "description": "Document ingestion failure"},
        502: {"model": RAGError, "description": "Upstream generation or retrieval failure"},
    },
)
async def ingest_rag(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    uploaded_by: str = Form(...),
    uploader_role: str = Form(...),
    allowed_roles: str = Form(..., description='JSON array string like ["admin","hr"].'),
    metadata: str | None = Form(default=None, description='Optional JSON object string like {"department":"HR"}.'),
    _: None = Depends(verify_internal_key),
):
    try:
        payload = {
            "tenant_id": tenant_id,
            "uploaded_by": uploaded_by,
            "uploader_role": uploader_role,
            "allowed_roles": _parse_allowed_roles(allowed_roles),
            "metadata": _parse_metadata(metadata),
        }
    except ValueError as exc:
        error = RAGError(
            status="error",
            error_code="INVALID_REQUEST",
            error_subcode="PAYLOAD_VALIDATION",
            message=str(exc),
        )
        return JSONResponse(status_code=400, content=error.model_dump())

    result = await run_rag_ingest(file, payload)
    if isinstance(result, RAGError):
        return JSONResponse(status_code=_status_for_error(result), content=result.model_dump())
    return result


@router.post("/rag/ingest", include_in_schema=False)
async def ingest_rag_legacy_alias(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    uploaded_by: str = Form(...),
    uploader_role: str = Form(...),
    allowed_roles: str = Form(...),
    metadata: str | None = Form(default=None),
    _: None = Depends(verify_internal_key),
):
    return await ingest_rag(file, tenant_id, uploaded_by, uploader_role, allowed_roles, metadata, _)
