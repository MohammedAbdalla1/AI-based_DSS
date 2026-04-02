"""
SQLite-backed persistence for RAG documents and access grants.
"""

from __future__ import annotations

import sqlite3
from json import dumps
from datetime import datetime, timezone
from config import get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    settings.ensure_runtime_dirs()
    connection = sqlite3.connect(settings.rag_sqlite_path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_schema() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                mime_type TEXT,
                status TEXT NOT NULL,
                chunks_indexed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_access (
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                granted_at TEXT NOT NULL,
                PRIMARY KEY (document_id, user_id),
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );

            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                file_name TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                allowed_roles TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                chunks_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def upsert_document(
    *,
    document_id: str,
    file_name: str,
    storage_path: str,
    uploaded_by: str,
    mime_type: str,
    status: str,
    chunks_indexed: int,
) -> None:
    ensure_schema()
    now = _utc_now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                file_name,
                storage_path,
                uploaded_by,
                mime_type,
                status,
                chunks_indexed,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                file_name=excluded.file_name,
                storage_path=excluded.storage_path,
                uploaded_by=excluded.uploaded_by,
                mime_type=excluded.mime_type,
                status=excluded.status,
                chunks_indexed=excluded.chunks_indexed,
                updated_at=excluded.updated_at
            """,
            (
                document_id,
                file_name,
                storage_path,
                uploaded_by,
                mime_type,
                status,
                chunks_indexed,
                now,
                now,
            ),
        )


def replace_document_access(*, document_id: str, user_ids: list[str], granted_by: str) -> None:
    ensure_schema()
    granted_at = _utc_now()
    with _connect() as connection:
        connection.execute("DELETE FROM document_access WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO document_access (document_id, user_id, granted_by, granted_at)
            VALUES (?, ?, ?, ?)
            """,
            [(document_id, user_id, granted_by, granted_at) for user_id in user_ids],
        )


def get_allowed_document_ids(user_id: str) -> list[str]:
    ensure_schema()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT document_id
            FROM document_access
            WHERE user_id = ?
            ORDER BY granted_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [str(row["document_id"]) for row in rows]


def insert_file_record(
    *,
    file_id: str,
    tenant_id: str,
    uploaded_by: str,
    file_name: str,
    storage_path: str,
    allowed_roles: list[str],
    metadata: dict,
    chunks_count: int,
) -> None:
    ensure_schema()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO files (
                id,
                tenant_id,
                uploaded_by,
                file_name,
                storage_path,
                allowed_roles,
                metadata,
                chunks_count,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                tenant_id,
                uploaded_by,
                file_name,
                storage_path,
                dumps(allowed_roles),
                dumps(metadata or {}),
                chunks_count,
                _utc_now(),
            ),
        )


def delete_file_record(file_id: str) -> None:
    """Remove a file record from SQLite. Used for compensating rollback."""
    ensure_schema()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM files WHERE id = ?",
            (file_id,),
        )
