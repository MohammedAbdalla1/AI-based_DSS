"""
models.py

Defines request and response models for the Text-to-SQL API.

This file is responsible ONLY for:
- API input validation
- API output structure
- Boundary enforcement between services

It must NOT contain:
- LLM logic
- SQL validation logic
- Business logic
"""

from typing import Dict, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator


# =========================
# Request Model
# =========================

class TextToSQLRequest(BaseModel):
    """
    Input model for the Text-to-SQL API.

    Expected from the general backend:
    - Natural language question
    - Target database dialect
    - Role-sliced schema (nested table -> column -> type)
    """

    question: str = Field(
        ...,
        description="Natural language question from the user."
    )

    db_type: Literal["postgres", "mysql", "sqlserver"] = Field(
        ...,
        description="Target database dialect."
    )

    role_schema: Dict[str, Dict[str, str]] = Field(
        ...,
        description="RBAC-sliced schema in the form: {table: {column: type}}"
    )

    # -------------------------
    # Custom Validators
    # -------------------------

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty or whitespace.")

        if len(value.split()) < 3:
            raise ValueError("Question must contain at least 3 words.")

        return value

    @field_validator("role_schema")
    @classmethod
    def validate_role_schema(cls, schema: Dict[str, Dict[str, str]]):
        if not schema:
            raise ValueError("role_schema cannot be empty.")

        for table_name, columns in schema.items():

            if not table_name or not isinstance(table_name, str):
                raise ValueError("Invalid table name in role_schema.")

            if not isinstance(columns, dict) or not columns:
                raise ValueError(
                    f"Table '{table_name}' must contain at least one column."
                )

            for column_name, column_type in columns.items():

                if not column_name or not isinstance(column_name, str):
                    raise ValueError(
                        f"Invalid column name in table '{table_name}'."
                    )

                if not column_type or not isinstance(column_type, str):
                    raise ValueError(
                        f"Invalid type for column '{column_name}' in table '{table_name}'."
                    )

        return schema


# =========================
# Response Model
# =========================

class TextToSQLSuccess(BaseModel):
    status: Literal["success"]
    sql: str


class TextToSQLError(BaseModel):
    status: Literal["error"]
    error_code: str
    error_subcode: Optional[str] = None
    message: str


TextToSQLResponse = Union[TextToSQLSuccess, TextToSQLError]
