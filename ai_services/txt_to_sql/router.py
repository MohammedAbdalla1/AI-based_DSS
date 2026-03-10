"""
FastAPI router layer for Text-to-SQL.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from .models import TextToSQLError, TextToSQLRequest, TextToSQLResponse
from .service import run_txt_to_sql


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


ERROR_STATUS_MAP = {
    "INVALID_REQUEST": 400,
    "SQL_REJECTED": 422,
    "GENERATION_FAILED": 502,
    "INTERNAL_ERROR": 500,
}


def _status_for_error(error: TextToSQLError) -> int:
    return ERROR_STATUS_MAP.get(error.error_code, 500)


router = APIRouter(prefix="/txt-to-sql", tags=["txt_to_sql"])


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
    error = TextToSQLError(
        status="error",
        error_code="INVALID_REQUEST",
        error_subcode="PAYLOAD_VALIDATION",
        message=request_validation_message(exc),
    )
    return JSONResponse(status_code=400, content=error.model_dump())


@router.post(
    "",
    response_model=TextToSQLResponse,
    responses={
        400: {"model": TextToSQLError, "description": "Invalid request payload"},
        422: {"model": TextToSQLError, "description": "SQL generated but rejected by validator policy"},
        502: {"model": TextToSQLError, "description": "Upstream model generation failure"},
        500: {"model": TextToSQLError, "description": "Unexpected internal failure"},
    },
)
def txt_to_sql(request: TextToSQLRequest):
    result = run_txt_to_sql(request)
    if result.status == "error":
        return JSONResponse(status_code=_status_for_error(result), content=result.model_dump())
    return result
