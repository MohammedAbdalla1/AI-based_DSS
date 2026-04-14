"""
Shared FastAPI entrypoint for AI services.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from config import load_environment
from rag.router import request_validation_error_response
from rag.router import router as rag_router
from reports.router import router as reports_router
from schema_to_analytics.router import router as schema_to_analytics_router
from txt_to_sql.router import (
    request_validation_error_response,
    router as txt_to_sql_router,
)


app = FastAPI(
    title="AI Services API",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
def handle_request_validation_error(_request: Request, exc: RequestValidationError):
    return request_validation_error_response(exc)


app.include_router(txt_to_sql_router)
app.include_router(rag_router)
app.include_router(schema_to_analytics_router)
app.include_router(reports_router)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}
