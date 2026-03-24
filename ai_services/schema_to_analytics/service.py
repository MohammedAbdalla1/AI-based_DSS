"""
service.py

Orchestration layer for Schema-to-Analytics.

Responsibilities:
- Accept request payloads
- Build analytics prompt and call LLM
- Parse JSON response into chart suggestions
- Validate each suggestion's SQL
- Return stable API-facing response
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from pydantic import ValidationError

from .models import (
    AnalyticsSuggestionError,
    AnalyticsSuggestionRequest,
    AnalyticsSuggestionSuccess,
    ChartSuggestion,
)
from .prompt_builder import build_analytics_prompt
from txt_to_sql.sql_generator import SQLGenerationError, generate_sql
from txt_to_sql.sql_validator import SQLValidationError, validate_sql


def _error(error_code: str, message: str) -> AnalyticsSuggestionError:
    return AnalyticsSuggestionError(
        status="error",
        error_code=error_code,
        message=message,
    )


def _extract_json_array(text: str) -> Optional[str]:
    """Extract a JSON array from the LLM response, handling markdown fences."""
    cleaned = text.strip()

    # Strip markdown fences
    fenced = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    # Find the JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return None


def _parse_suggestions(raw_text: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of suggestion dicts."""
    json_str = _extract_json_array(raw_text)
    if not json_str:
        return []

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return parsed


def _validate_suggestion(
    suggestion_dict: Dict[str, Any],
    role_schema: Dict[str, Dict[str, str]],
    dialect: str,
) -> Optional[ChartSuggestion]:
    """Validate a single suggestion dict. Returns None if invalid."""
    try:
        sql = suggestion_dict.get("sql", "")
        if not sql or not isinstance(sql, str):
            return None

        # Validate SQL against the schema
        validated_sql = validate_sql(
            sql=sql,
            role_schema=role_schema,
            dialect=dialect,
        )

        # Validate chart_type
        VALID_CHART_TYPES = {
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
        }

        chart_type = suggestion_dict.get("chart_type", "bar")
        if chart_type not in VALID_CHART_TYPES:
            chart_type = "bar"

        # Build the suggestion with validated SQL
        raw_insight = suggestion_dict.get("insight")
        insight = str(raw_insight).strip() if raw_insight else None

        return ChartSuggestion(
            title=str(suggestion_dict.get("title", "Untitled Chart")),
            description=str(suggestion_dict.get("description", "")),
            sql=validated_sql,
            chart_type=chart_type,
            chart_config=suggestion_dict.get("chart_config", {}),
            insight=insight if insight else None,
        )
    except (SQLValidationError, Exception) as exc:
        logger.warning("Suggestion validation failed: %s | SQL: %s", exc, suggestion_dict.get("sql", "")[:200])
        return None


def run_schema_to_analytics(
    request_or_payload: AnalyticsSuggestionRequest | dict,
) -> AnalyticsSuggestionSuccess | AnalyticsSuggestionError:
    """
    Execute the Schema-to-Analytics pipeline and return a typed response.
    """

    # Validate request
    try:
        if isinstance(request_or_payload, AnalyticsSuggestionRequest):
            request = request_or_payload
        else:
            request = AnalyticsSuggestionRequest(**request_or_payload)
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc))
    except Exception as exc:
        return _error("INVALID_REQUEST", f"Invalid request payload: {type(exc).__name__}")

    try:
        # Build the analytics prompt
        prompt = build_analytics_prompt(
            schema=request.role_schema,
            db_type=request.db_type,
            max_suggestions=request.max_suggestions,
            user_prompt=request.user_prompt,
            rag_context=request.rag_context,
        )

        # Call the LLM (reuse the same Gemini generator)
        raw_response = generate_sql(prompt)

        # Parse JSON suggestions from LLM response
        suggestion_dicts = _parse_suggestions(raw_response)

        if not suggestion_dicts:
            return _error(
                "GENERATION_FAILED",
                "The AI did not return valid chart suggestions. Please try again.",
            )

        # Validate each suggestion's SQL against the schema
        valid_suggestions: List[ChartSuggestion] = []
        for suggestion_dict in suggestion_dicts:
            validated = _validate_suggestion(
                suggestion_dict=suggestion_dict,
                role_schema=request.role_schema,
                dialect=request.db_type,
            )
            if validated:
                valid_suggestions.append(validated)

            if len(valid_suggestions) >= request.max_suggestions:
                break

        if not valid_suggestions:
            return _error(
                "SQL_REJECTED",
                "All suggested queries failed validation. The schema may be too restrictive.",
            )

        return AnalyticsSuggestionSuccess(
            status="success",
            suggestions=valid_suggestions,
        )

    except SQLGenerationError as exc:
        return _error("GENERATION_FAILED", str(exc))
    except Exception as exc:
        return _error("INTERNAL_ERROR", f"Unexpected error: {type(exc).__name__}")
