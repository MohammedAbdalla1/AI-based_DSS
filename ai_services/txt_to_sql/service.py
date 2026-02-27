"""
service.py

Orchestration layer for Text-to-SQL.

Responsibilities:
- Accept request payloads
- Run prompt building, SQL generation, and SQL validation
- Return a stable API-facing response model with mapped error codes
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from .models import (
    TextToSQLError,
    TextToSQLRequest,
    TextToSQLResponse,
    TextToSQLSuccess,
)
from .prompt_builder import build_sql_prompt
from .sql_generator import SQLGenerationError, generate_sql
from .sql_validator import SQLValidationError, validate_sql


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    local_cleanup: bool = False
    regenerate: bool = False


GENERATION_RETRY_MATRIX = {
    "EMPTY_RESPONSE": RetryPolicy(max_attempts=2),
    "INVALID_RESPONSE_TYPE": RetryPolicy(max_attempts=2),
    "PROVIDER_FAILURE": RetryPolicy(max_attempts=2),
}

VALIDATION_RETRY_MATRIX = {
    "MARKDOWN_OR_COMMENTS": RetryPolicy(max_attempts=2, local_cleanup=True),
    "EMPTY_SQL": RetryPolicy(max_attempts=2, regenerate=True),
    "PARSE_ERROR": RetryPolicy(max_attempts=2, regenerate=True),
}

SQL_FENCE_PATTERN = re.compile(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", re.IGNORECASE)
SQL_BLOCK_COMMENT_PATTERN = re.compile(r"/\*[\s\S]*?\*/")
SQL_LINE_COMMENT_PATTERN = re.compile(r"--.*?$", re.MULTILINE)


def _error(error_code: str, message: str, error_subcode: Optional[str] = None) -> TextToSQLError:
    return TextToSQLError(
        status="error",
        error_code=error_code,
        error_subcode=error_subcode,
        message=message,
    )


def _classify_generation_error(exc: SQLGenerationError) -> str:
    subcode = getattr(exc, "subcode", None)
    if isinstance(subcode, str) and subcode.strip() and subcode != "UNKNOWN":
        return subcode

    message = str(exc).lower()

    if "prompt is empty" in message:
        return "PROMPT_EMPTY"
    if "gemini_api_key is missing" in message:
        return "MISSING_API_KEY"
    if "gemini client initialization failed" in message:
        return "CLIENT_INIT_FAILED"
    if "invalid response type from llm" in message:
        return "INVALID_RESPONSE_TYPE"
    if "empty response from llm" in message:
        return "EMPTY_RESPONSE"
    if "gemini generation failed" in message:
        return "PROVIDER_FAILURE"
    if isinstance(subcode, str) and subcode.strip():
        return subcode
    return "UNKNOWN"


def _classify_validation_error(exc: SQLValidationError) -> str:
    subcode = getattr(exc, "subcode", None)
    if isinstance(subcode, str) and subcode.strip() and subcode != "UNKNOWN":
        return subcode

    message = str(exc).lower()

    if "markdown or code formatting" in message or "sql comments are not allowed" in message:
        return "MARKDOWN_OR_COMMENTS"
    if "sql is empty" in message:
        return "EMPTY_SQL"
    if "sql parse error" in message:
        return "PARSE_ERROR"
    if "multiple sql statements" in message:
        return "MULTI_STATEMENT"
    if "only select queries are allowed" in message:
        return "NON_SELECT_QUERY"
    if "select into is not allowed" in message:
        return "SELECT_INTO"
    if "wildcard" in message:
        return "WILDCARD_SELECT"
    if "select without from" in message:
        return "SELECT_WITHOUT_FROM"
    if "recursive ctes" in message:
        return "RECURSIVE_CTE"
    if "union and union all" in message:
        return "UNION_NOT_SUPPORTED"
    if "schema-qualified functions are not allowed" in message:
        return "SCHEMA_QUALIFIED_FUNCTION"
    if "function not allowed" in message:
        return "FUNCTION_NOT_ALLOWED"
    if "access to system schema is not allowed" in message:
        return "SYSTEM_SCHEMA_ACCESS"
    if "table not allowed for this role" in message:
        return "TABLE_NOT_ALLOWED"
    if "unauthorized table access" in message:
        return "UNAUTHORIZED_TABLE_ACCESS"
    if "unauthorized access to" in message:
        return "UNAUTHORIZED_COLUMN_ACCESS"
    if "schema resolution error" in message:
        return "SCHEMA_RESOLUTION_ERROR"
    if "unresolved column detected" in message:
        return "UNRESOLVED_COLUMN"
    if isinstance(subcode, str) and subcode.strip():
        return subcode
    return "UNKNOWN"


def _sanitize_sql_locally(sql: str) -> str:
    cleaned = sql.strip()

    fenced = SQL_FENCE_PATTERN.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    cleaned = SQL_BLOCK_COMMENT_PATTERN.sub(" ", cleaned)
    cleaned = SQL_LINE_COMMENT_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip()


def _build_validation_retry_prompt(base_prompt: str, reason: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Previous SQL was rejected by validator.\n"
        f"Rejection reason: {reason}\n"
        "Regenerate one valid SQL query only.\n"
        "Do not include markdown, comments, or any explanation."
    )


def _generate_sql_with_retry(prompt: str) -> tuple[Optional[str], Optional[TextToSQLError]]:
    attempt = 0
    prompt_to_use = prompt

    while True:
        attempt += 1
        try:
            return generate_sql(prompt_to_use), None
        except SQLGenerationError as exc:
            subcode = _classify_generation_error(exc)
            policy = GENERATION_RETRY_MATRIX.get(subcode, RetryPolicy(max_attempts=1))
            if attempt < policy.max_attempts:
                prompt_to_use = (
                    f"{prompt}\n\n"
                    "Output format reminder:\n"
                    "- Return only SQL text.\n"
                    "- No markdown fences.\n"
                    "- No comments.\n"
                )
                continue
            return None, _error("GENERATION_FAILED", str(exc), error_subcode=subcode)


def run_txt_to_sql(request_or_payload: TextToSQLRequest | Mapping[str, Any]) -> TextToSQLResponse:
    """
    Execute the Text-to-SQL pipeline and return a typed response.

    Accepts either:
    - TextToSQLRequest instance
    - raw mapping payload (validated into TextToSQLRequest)
    """

    try:
        if isinstance(request_or_payload, TextToSQLRequest):
            request = request_or_payload
        else:
            request = TextToSQLRequest(**dict(request_or_payload))
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc), error_subcode="PAYLOAD_VALIDATION")
    except Exception as exc:
        return _error(
            "INVALID_REQUEST",
            f"Invalid request payload type: {type(exc).__name__}",
            error_subcode="PAYLOAD_TYPE",
        )

    try:
        prompt = build_sql_prompt(
            question=request.question,
            schema=request.role_schema,
            db_type=request.db_type,
        )
        generated_sql, generation_error = _generate_sql_with_retry(prompt)
        if generation_error:
            return generation_error
        assert generated_sql is not None

        validation_attempts_by_subcode: dict[str, int] = {}
        sql_candidate = generated_sql
        validation_prompt = prompt
        while True:
            try:
                validated_sql = validate_sql(
                    sql_candidate,
                    role_schema=request.role_schema,
                    dialect=request.db_type,
                )
                return TextToSQLSuccess(status="success", sql=validated_sql)
            except SQLValidationError as exc:
                subcode = _classify_validation_error(exc)
                policy = VALIDATION_RETRY_MATRIX.get(subcode, RetryPolicy(max_attempts=1))
                subcode_attempts = validation_attempts_by_subcode.get(subcode, 0) + 1
                validation_attempts_by_subcode[subcode] = subcode_attempts

                if policy.local_cleanup and subcode_attempts < policy.max_attempts:
                    sanitized = _sanitize_sql_locally(sql_candidate)
                    if sanitized and sanitized != sql_candidate:
                        sql_candidate = sanitized
                        continue

                if policy.regenerate and subcode_attempts < policy.max_attempts:
                    validation_prompt = _build_validation_retry_prompt(validation_prompt, str(exc))
                    sql_candidate, generation_error = _generate_sql_with_retry(validation_prompt)
                    if generation_error:
                        return generation_error
                    assert sql_candidate is not None
                    continue

                return _error("SQL_REJECTED", str(exc), error_subcode=subcode)
    except Exception as exc:
        return _error(
            "INTERNAL_ERROR",
            f"Unexpected error: {type(exc).__name__}",
            error_subcode="UNEXPECTED_EXCEPTION",
        )
