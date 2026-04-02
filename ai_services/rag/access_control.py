"""
Access-control helpers for document grants.
"""

from __future__ import annotations

from . import database


def normalize_allowed_user_ids(user_ids: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in user_ids:
        item = value.strip()
        if item and item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


def replace_document_access(document_id: str, allowed_user_ids: list[str], granted_by: str) -> list[str]:
    normalized = normalize_allowed_user_ids(allowed_user_ids)
    if not normalized:
        raise ValueError("allowed_user_ids cannot be empty.")
    database.replace_document_access(
        document_id=document_id,
        user_ids=normalized,
        granted_by=granted_by,
    )
    return normalized


def get_allowed_document_ids(user_id: str) -> list[str]:
    return database.get_allowed_document_ids(user_id.strip())
