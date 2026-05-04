"""
models.py

Pydantic request/response models for the Reports AI endpoints:
  POST /plan-report     — plan sections for a full report
  POST /narrate-section — write narrative text for one section given SQL results
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class ReportAiError(BaseModel):
    status: Literal["error"]
    error_code: str
    message: str


# ── Plan Report ───────────────────────────────────────────────────────────────

class PlanReportRequest(BaseModel):
    db_type: Literal["postgres", "mysql", "sqlserver"] = Field(
        ..., description="Target database dialect."
    )
    role_schema: Dict[str, Dict[str, str]] = Field(
        ..., description="RBAC-sliced schema: {table: {column: type}}"
    )
    user_prompt: str = Field(
        ..., min_length=3, description="What the user wants the report to cover."
    )
    max_sections: int = Field(
        default=5, ge=1, le=9, description="Maximum number of sections to plan."
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
                raise ValueError(f"Table '{table_name}' must contain at least one column.")
            for col_name, col_type in columns.items():
                if not col_name or not isinstance(col_name, str):
                    raise ValueError(f"Invalid column name in table '{table_name}'.")
                if not col_type or not isinstance(col_type, str):
                    raise ValueError(f"Invalid type for column '{col_name}' in table '{table_name}'.")
        return schema


class PlannedSection(BaseModel):
    heading: str
    sql: Optional[str] = None
    chart_type: Optional[str] = None
    chart_config: Optional[Dict[str, Any]] = None


class PlanReportSuccess(BaseModel):
    status: Literal["success"]
    title: str
    summary: Optional[str] = None
    sections: List[PlannedSection]


# ── Narrate Section ───────────────────────────────────────────────────────────

class NarrateSectionRequest(BaseModel):
    heading: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1, description="Original report prompt for context.")
    sql: str = Field(default="", description="SQL that was executed for this section.")
    sample_rows_json: str = Field(
        default="[]", description="JSON array of up to 20 result rows."
    )


class NarrateSectionSuccess(BaseModel):
    status: Literal["success"]
    narrative: str


# ── Narrate Report (batched) ──────────────────────────────────────────────────

class NarrateReportSectionInput(BaseModel):
    heading: str = Field(..., min_length=1)
    sql: Optional[str] = None
    chart_type: Optional[str] = None
    sample_rows_json: str = Field(default="[]")


class NarrateReportRequest(BaseModel):
    user_prompt: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: Optional[str] = None
    sections: List[NarrateReportSectionInput] = Field(..., min_length=1, max_length=8)


class NarratedSection(BaseModel):
    heading: str
    narrative: str


class NarrateReportSuccess(BaseModel):
    status: Literal["success"]
    sections: List[NarratedSection]
