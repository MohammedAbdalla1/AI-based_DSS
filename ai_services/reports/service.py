"""
service.py

Orchestration for the two report AI endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from google import genai

from .models import (
    NarrateSectionRequest,
    NarrateSectionSuccess,
    PlanReportRequest,
    PlanReportSuccess,
    PlannedSection,
    ReportAiError,
)
from .prompt_builder import build_narrate_section_prompt, build_plan_report_prompt
from txt_to_sql.sql_generator import SQLGenerationError

logger = logging.getLogger(__name__)


# ── Gemini client (cached, with retry) ────────────────────────────────────────

_gemini_client: Optional[genai.Client] = None
_provider_limit_until_monotonic: float = 0.0


def _provider_limit_cooldown_seconds() -> int:
    raw = os.getenv("REPORTS_PROVIDER_LIMIT_COOLDOWN_SECONDS", "300").strip()
    try:
        value = int(raw)
        return max(30, value)
    except ValueError:
        return 300


def _is_provider_limited_now() -> bool:
    return time.monotonic() < _provider_limit_until_monotonic


def _mark_provider_limited() -> None:
    global _provider_limit_until_monotonic
    _provider_limit_until_monotonic = time.monotonic() + _provider_limit_cooldown_seconds()


def _classify_provider_error(exc: Exception) -> Optional[str]:
    text = f"{type(exc).__name__}: {exc}".lower()

    has_429 = "429" in text
    has_quota = "resource_exhausted" in text or "quota" in text or "billing" in text
    has_rate_limit = (
        "rate limit" in text
        or "rate_limited" in text
        or "too many requests" in text
    )

    if has_429 and has_quota:
        return "QUOTA_EXCEEDED"
    if has_429 or has_rate_limit or has_quota:
        return "RATE_LIMITED"
    return None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SQLGenerationError("GEMINI_API_KEY is missing", subcode="MISSING_API_KEY")
    try:
        _gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        raise SQLGenerationError(
            f"Gemini client init failed: {type(e).__name__}",
            subcode="CLIENT_INIT_FAILED",
        ) from e
    return _gemini_client


def _generate_text_with_retry(
    prompt: str,
    *,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
    attempts: int = 3,
) -> str:
    """
    Call Gemini with exponential-backoff retry on transient server errors.
    Reports responses are larger than single-SQL responses, so we use a much
    higher max_output_tokens than the shared `generate_sql` helper.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise SQLGenerationError("Prompt is empty", subcode="PROMPT_EMPTY")

    if _is_provider_limited_now():
        raise SQLGenerationError(
            "Gemini provider is currently rate limited. Retry later.",
            subcode="RATE_LIMITED",
        )

    client = _get_gemini_client()
    last_exc: Optional[Exception] = None
    failure_subcode = "PROVIDER_FAILURE"

    for i in range(attempts):
        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Empty response from Gemini")
            return text
        except Exception as e:
            last_exc = e
            name = type(e).__name__
            classified = _classify_provider_error(e)

            if classified is not None:
                failure_subcode = classified
                _mark_provider_limited()
                logger.warning(
                    "Gemini provider limited (%s). Cooldown enabled for %ds.",
                    classified,
                    _provider_limit_cooldown_seconds(),
                )
                break

            # Retry transient server errors, not config errors
            is_transient = name in {"ServerError", "ServiceUnavailable", "InternalServerError", "RuntimeError"} or "5" in str(e)[:5]
            if i < attempts - 1 and is_transient:
                wait = 2 ** i  # 1s, 2s
                logger.warning(
                    "Gemini call failed (%s), retrying in %ds (attempt %d/%d)",
                    name, wait, i + 1, attempts,
                )
                time.sleep(wait)
                continue
            logger.error("Gemini call failed after %d attempt(s): %s — %s", i + 1, name, str(e)[:200])
            break

    raise SQLGenerationError(
        f"Gemini generation failed: {type(last_exc).__name__ if last_exc else 'Unknown'}",
        subcode=failure_subcode,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error(error_code: str, message: str) -> ReportAiError:
    return ReportAiError(status="error", error_code=error_code, message=message)


def _extract_json_object(text: str) -> Optional[Dict]:
    """Extract the first JSON object from LLM output, stripping markdown fences."""
    cleaned = text.strip()

    # Strip markdown code fences
    fenced = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    # Find outermost { }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _sample_rows_is_empty(sample_rows_json: str) -> bool:
    raw = sample_rows_json.strip()
    if not raw or raw.lower() == "null":
        return True

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False

    if parsed is None:
        return True
    if isinstance(parsed, list):
        return len(parsed) == 0
    return False


def _parse_planned_sections(raw_sections: Any) -> List[PlannedSection]:
    """Convert the raw LLM sections list into PlannedSection objects."""
    if not isinstance(raw_sections, list):
        return []

    sections: List[PlannedSection] = []
    for item in raw_sections:
        if not isinstance(item, dict):
            continue

        heading = item.get("heading")
        if not heading or not isinstance(heading, str):
            continue

        sql = item.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            sql = None

        chart_type = item.get("chart_type")
        if not isinstance(chart_type, str) or not chart_type.strip():
            chart_type = None

        chart_config = item.get("chart_config")
        if not isinstance(chart_config, dict):
            chart_config = None

        sections.append(PlannedSection(
            heading=heading.strip(),
            sql=sql,
            chart_type=chart_type,
            chart_config=chart_config,
        ))

    return sections


# ── Plan Report ───────────────────────────────────────────────────────────────

def run_plan_report(request: PlanReportRequest) -> PlanReportSuccess | ReportAiError:
    try:
        prompt = build_plan_report_prompt(
            schema=request.role_schema,
            db_type=request.db_type,
            user_prompt=request.user_prompt,
            max_sections=request.max_sections,
        )

        text = _generate_text_with_retry(prompt)

        data = _extract_json_object(text)
        if not data:
            logger.warning("plan-report: LLM returned no parseable JSON. Raw (truncated): %s", text[:400])
            return _error("GENERATION_FAILED", "The AI did not return a valid report plan. Please try again.")

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            title = "Report"

        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = None

        sections = _parse_planned_sections(data.get("sections", []))
        if not sections:
            logger.warning("plan-report: LLM returned 0 parseable sections. Data: %s", str(data)[:400])
            return _error("GENERATION_FAILED", "The AI did not plan any sections. Please try a more specific prompt.")

        logger.info("plan-report: planned %d sections. Title: %s", len(sections), title)
        return PlanReportSuccess(
            status="success",
            title=title.strip(),
            summary=summary,
            sections=sections,
        )

    except SQLGenerationError as exc:
        if getattr(exc, "subcode", "") in {"RATE_LIMITED", "QUOTA_EXCEEDED"}:
            logger.warning("plan-report: provider limit reached: %s", exc)
            return _error(
                "RATE_LIMITED",
                "AI provider rate limit/quota reached. Please retry later.",
            )
        logger.error("plan-report: SQLGenerationError: %s", exc)
        return _error("GENERATION_FAILED", str(exc))
    except Exception as exc:
        logger.exception("plan-report: unexpected error")
        return _error("INTERNAL_ERROR", f"Unexpected error: {type(exc).__name__}")


# ── Narrate Section ───────────────────────────────────────────────────────────

def run_narrate_section(request: NarrateSectionRequest) -> NarrateSectionSuccess | ReportAiError:
    try:
        is_sql_backed_section = bool(request.sql and request.sql.strip())
        if is_sql_backed_section and _sample_rows_is_empty(request.sample_rows_json):
            logger.info("narrate-section: skipping Gemini for '%s' due to empty SQL result", request.heading)
            return NarrateSectionSuccess(status="success", narrative="")

        prompt = build_narrate_section_prompt(
            heading=request.heading,
            user_prompt=request.user_prompt,
            sql=request.sql,
            sample_rows_json=request.sample_rows_json,
        )

        text = _generate_text_with_retry(prompt)

        # Try to parse as JSON {"narrative": "..."}
        data = _extract_json_object(text)
        if data and isinstance(data.get("narrative"), str) and data["narrative"].strip():
            narrative = data["narrative"].strip()
        else:
            # Fallback: treat the whole response as plain text
            narrative = text.strip()

        if not narrative:
            return NarrateSectionSuccess(status="success", narrative="")

        logger.info("narrate-section: generated narrative for '%s' (%d chars)", request.heading, len(narrative))
        return NarrateSectionSuccess(status="success", narrative=narrative)

    except SQLGenerationError as exc:
        if getattr(exc, "subcode", "") in {"RATE_LIMITED", "QUOTA_EXCEEDED"}:
            logger.warning("narrate-section: provider limit reached, returning empty narrative")
            return NarrateSectionSuccess(status="success", narrative="")
        logger.error("narrate-section: SQLGenerationError: %s", exc)
        return _error("GENERATION_FAILED", str(exc))
    except Exception as exc:
        logger.exception("narrate-section: unexpected error")
        return _error("INTERNAL_ERROR", f"Unexpected error: {type(exc).__name__}")
