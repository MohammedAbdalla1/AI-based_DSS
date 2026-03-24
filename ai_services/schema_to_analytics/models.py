"""
models.py

Defines request and response models for the Schema-to-Analytics API.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# =========================
# Request Model
# =========================

class AnalyticsSuggestionRequest(BaseModel):
    """
    Input model for the Schema-to-Analytics API.

    Expected from the backend:
    - Target database dialect
    - Role-sliced schema (nested table -> column -> type)
    - Optional natural language description of desired analytics
    - Optional RAG context for business-aware suggestions
    """

    db_type: Literal["postgres", "mysql", "sqlserver"] = Field(
        ...,
        description="Target database dialect.",
    )

    role_schema: Dict[str, Dict[str, str]] = Field(
        ...,
        description="RBAC-sliced schema in the form: {table: {column: type}}",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        description="Optional natural language description of what the user wants to analyze.",
    )

    rag_context: Optional[str] = Field(
        default=None,
        description="Optional business context from RAG for more relevant suggestions.",
    )

    max_suggestions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of chart suggestions to return.",
    )

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
# Response Models
# =========================

class ChartSuggestion(BaseModel):
    title: str
    description: str
    sql: str
    chart_type: Literal[
        # Bar family
        "bar", "horizontal_bar", "stacked_bar", "grouped_bar", "floating_bar",
        # Line family
        "line", "area", "stepped_line", "multi_axis_line",
        # Circular family
        "pie", "doughnut", "polar_area",
        # Radial
        "radar",
        # Point-based
        "scatter", "bubble",
        # Mixed
        "composed",
        # Plugin-based
        "treemap", "funnel",
    ]
    chart_config: Dict[str, Any]
    insight: Optional[str] = None


class AnalyticsSuggestionSuccess(BaseModel):
    status: Literal["success"]
    suggestions: List[ChartSuggestion]


class AnalyticsSuggestionError(BaseModel):
    status: Literal["error"]
    error_code: str
    message: str
