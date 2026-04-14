"""
router.py

FastAPI router for the Reports AI endpoints:
  POST /plan-report      — plan sections for a full report
  POST /narrate-section  — write narrative text for one section
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .models import (
    NarrateReportRequest,
    NarrateReportSuccess,
    NarrateSectionRequest,
    NarrateSectionSuccess,
    PlanReportRequest,
    PlanReportSuccess,
    ReportAiError,
)
from .service import run_narrate_report, run_narrate_section, run_plan_report


router = APIRouter(tags=["reports"])

_ERROR_STATUS_MAP = {
    "INVALID_REQUEST": 400,
    "RATE_LIMITED": 429,
    "PROVIDER_UNAVAILABLE": 503,
    "GENERATION_FAILED": 502,
    "INTERNAL_ERROR": 500,
}


def _status_for(error: ReportAiError) -> int:
    return _ERROR_STATUS_MAP.get(error.error_code, 500)


@router.post(
    "/plan-report",
    response_model=PlanReportSuccess,
    responses={
        400: {"model": ReportAiError, "description": "Invalid request payload"},
        429: {"model": ReportAiError, "description": "LLM provider rate-limited or quota-exhausted"},
        503: {"model": ReportAiError, "description": "LLM provider temporarily unavailable"},
        502: {"model": ReportAiError, "description": "LLM generation failure"},
        500: {"model": ReportAiError, "description": "Unexpected internal error"},
    },
    summary="Plan a multi-section report from schema + user prompt",
)
def plan_report(request: PlanReportRequest):
    result = run_plan_report(request)
    if result.status == "error":
        return JSONResponse(status_code=_status_for(result), content=result.model_dump())
    return result


@router.post(
    "/narrate-section",
    response_model=NarrateSectionSuccess,
    responses={
        429: {"model": ReportAiError, "description": "LLM provider rate-limited or quota-exhausted"},
        502: {"model": ReportAiError, "description": "LLM generation failure"},
        500: {"model": ReportAiError, "description": "Unexpected internal error"},
    },
    summary="Write a narrative paragraph for one report section given its SQL results",
)
def narrate_section(request: NarrateSectionRequest):
    result = run_narrate_section(request)
    if result.status == "error":
        return JSONResponse(status_code=_status_for(result), content=result.model_dump())
    return result


@router.post(
    "/narrate-report",
    response_model=NarrateReportSuccess,
    responses={
        429: {"model": ReportAiError, "description": "LLM provider rate-limited or quota-exhausted"},
        502: {"model": ReportAiError, "description": "LLM generation failure"},
        500: {"model": ReportAiError, "description": "Unexpected internal error"},
    },
    summary="Write narratives for all report sections in a single batched LLM call",
)
def narrate_report(request: NarrateReportRequest):
    result = run_narrate_report(request)
    if result.status == "error":
        return JSONResponse(status_code=_status_for(result), content=result.model_dump())
    return result
