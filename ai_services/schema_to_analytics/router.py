"""
FastAPI router layer for Schema-to-Analytics.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .models import (
    AnalyticsSuggestionError,
    AnalyticsSuggestionRequest,
    AnalyticsSuggestionSuccess,
)
from .service import run_schema_to_analytics


ERROR_STATUS_MAP = {
    "INVALID_REQUEST": 400,
    "SQL_REJECTED": 422,
    "GENERATION_FAILED": 502,
    "INTERNAL_ERROR": 500,
}


def _status_for_error(error: AnalyticsSuggestionError) -> int:
    return ERROR_STATUS_MAP.get(error.error_code, 500)


router = APIRouter(prefix="/schema-to-analytics", tags=["analytics"])


@router.post(
    "",
    response_model=AnalyticsSuggestionSuccess,
    responses={
        400: {"model": AnalyticsSuggestionError, "description": "Invalid request payload"},
        422: {"model": AnalyticsSuggestionError, "description": "All suggestions rejected by validator"},
        502: {"model": AnalyticsSuggestionError, "description": "Upstream model generation failure"},
        500: {"model": AnalyticsSuggestionError, "description": "Unexpected internal failure"},
    },
)
def suggest_analytics(request: AnalyticsSuggestionRequest):
    result = run_schema_to_analytics(request)
    if result.status == "error":
        return JSONResponse(status_code=_status_for_error(result), content=result.model_dump())
    return result
