"""
FastAPI router layer for Retrieval-Augmented Generation (RAG).
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask")
def ask_rag():
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={
            "status": "error",
            "error_code": "NOT_IMPLEMENTED",
            "message": "RAG answer pipeline is not implemented yet.",
        },
    )
