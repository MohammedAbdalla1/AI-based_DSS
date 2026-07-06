"""
service.py

Orchestration for the two report AI endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

from llm_clients import ProviderCallError, call_groq_text, get_gemini_client, should_fallback_to_groq

from .models import (
    NarratedSection,
    NarrateReportRequest,
    NarrateReportSuccess,
    NarrateSectionRequest,
    NarrateSectionSuccess,
    PlanReportRequest,
    PlanReportSuccess,
    PlannedSection,
    ReportAiError,
)
from .prompt_builder import (
    build_narrate_report_prompt,
    build_narrate_section_prompt,
    build_plan_report_prompt,
)
from txt_to_sql.sql_generator import SQLGenerationError

logger = logging.getLogger(__name__)


# ── Gemini retry state ────────────────────────────────────────────────────────

_provider_limit_until_monotonic: float = 0.0
_model_unavailable_until_monotonic: Dict[str, float] = {}


def _provider_limit_cooldown_seconds() -> int:
    raw = os.getenv("REPORTS_PROVIDER_LIMIT_COOLDOWN_SECONDS", "300").strip()
    try:
        value = int(raw)
        return max(30, value)
    except ValueError:
        return 300


def _model_unavailable_cooldown_seconds() -> int:
    raw = os.getenv("REPORTS_MODEL_UNAVAILABLE_COOLDOWN_SECONDS", "60").strip()
    try:
        value = int(raw)
        return max(15, value)
    except ValueError:
        return 60


def _attempts_per_model(env_name: str, default_value: int = 2) -> int:
    raw = os.getenv(env_name, str(default_value)).strip()
    try:
        value = int(raw)
        return max(1, min(value, 5))
    except ValueError:
        return default_value


def _parse_model_list(raw: str, default_models: List[str]) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        parts = default_models

    # De-duplicate while preserving order.
    seen: set[str] = set()
    result: List[str] = []
    for model in parts:
        if model in seen:
            continue
        seen.add(model)
        result.append(model)
    return result


def _plan_models() -> List[str]:
    single = os.getenv("REPORTS_PLAN_MODEL", "").strip()
    listed = os.getenv("REPORTS_PLAN_MODELS", "").strip()
    combined = ",".join([x for x in [single, listed] if x])
    return _parse_model_list(
        combined,
        ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"],
    )


def _narrate_models() -> List[str]:
    single = os.getenv("REPORTS_NARRATE_MODEL", "").strip()
    listed = os.getenv("REPORTS_NARRATE_MODELS", "").strip()
    combined = ",".join([x for x in [single, listed] if x])
    return _parse_model_list(
        combined,
        ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"],
    )


def _groq_model() -> str:
    raw = os.getenv("REPORTS_GROQ_MODEL", "").strip()
    if raw:
        return raw
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"


def _is_provider_limited_now() -> bool:
    return time.monotonic() < _provider_limit_until_monotonic


def _mark_provider_limited() -> None:
    global _provider_limit_until_monotonic
    _provider_limit_until_monotonic = time.monotonic() + _provider_limit_cooldown_seconds()


def _is_model_temporarily_unavailable(model: str) -> bool:
    return time.monotonic() < _model_unavailable_until_monotonic.get(model, 0.0)


def _mark_model_temporarily_unavailable(model: str) -> None:
    _model_unavailable_until_monotonic[model] = (
        time.monotonic() + _model_unavailable_cooldown_seconds()
    )


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


def _is_transient_provider_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "503" in text or "unavailable" in text or "service unavailable" in text:
        return True

    name = type(exc).__name__
    return name in {
        "ServerError",
        "ServiceUnavailable",
        "InternalServerError",
        "RuntimeError",
        "GatewayTimeout",
    }


def _get_gemini_client() -> object:
    try:
        return get_gemini_client()
    except ProviderCallError as exc:
        raise SQLGenerationError(str(exc), subcode=exc.subcode) from exc


def _generate_text_with_retry(
    prompt: str,
    *,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
    attempts: int = 2,
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

    model_candidates = models if models is not None else [model or "models/gemini-2.5-flash-lite"]
    model_candidates = [m.strip() for m in model_candidates if isinstance(m, str) and m.strip()]
    if not model_candidates:
        model_candidates = ["models/gemini-2.5-flash-lite"]

    eligible_models = [m for m in model_candidates if not _is_model_temporarily_unavailable(m)]
    if not eligible_models:
        raise SQLGenerationError(
            "Gemini report models are temporarily unavailable. Retry shortly.",
            subcode="MODEL_UNAVAILABLE",
        )

    client = _get_gemini_client()
    last_exc: Optional[Exception] = None
    failure_subcode = "PROVIDER_FAILURE"
    saw_transient_failure = False

    for current_model in eligible_models:
        for i in range(attempts):
            try:
                response = client.models.generate_content(
                    model=current_model,
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

                is_transient = _is_transient_provider_error(e)
                if is_transient:
                    saw_transient_failure = True
                    if i < attempts - 1:
                        wait = (2 ** i) + random.uniform(0.05, 0.35)
                        logger.warning(
                            "Gemini call failed (%s, model=%s), retrying in %.2fs (attempt %d/%d)",
                            name,
                            current_model,
                            wait,
                            i + 1,
                            attempts,
                        )
                        time.sleep(wait)
                        continue

                    _mark_model_temporarily_unavailable(current_model)
                    failure_subcode = "MODEL_UNAVAILABLE"
                    logger.warning(
                        "Gemini model %s marked unavailable for %ds after transient errors.",
                        current_model,
                        _model_unavailable_cooldown_seconds(),
                    )
                    break

                logger.error(
                    "Gemini call failed after %d attempt(s): %s (model=%s) — %s",
                    i + 1,
                    name,
                    current_model,
                    str(e)[:200],
                )
                break

        if failure_subcode in {"RATE_LIMITED", "QUOTA_EXCEEDED"}:
            break

    if saw_transient_failure and failure_subcode == "PROVIDER_FAILURE":
        failure_subcode = "MODEL_UNAVAILABLE"

    raise SQLGenerationError(
        f"Gemini generation failed: {type(last_exc).__name__ if last_exc else 'Unknown'}",
        subcode=failure_subcode,
    )


def _generate_text_with_groq(
    prompt: str,
    *,
    model: str,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
) -> str:
    try:
        return call_groq_text(
            prompt,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
        )
    except ProviderCallError as exc:
        raise SQLGenerationError(
            f"Groq generation failed: {type(exc).__name__}",
            subcode=exc.subcode,
        ) from exc


def _generate_text_with_fallback(
    prompt: str,
    *,
    model: Optional[str] = None,
    models: Optional[List[str]] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
    attempts: int = 2,
) -> str:
    try:
        return _generate_text_with_retry(
            prompt,
            model=model,
            models=models,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            attempts=attempts,
        )
    except SQLGenerationError as gemini_exc:
        if not should_fallback_to_groq(gemini_exc.subcode):
            raise

        groq_error: Optional[SQLGenerationError] = None
        try:
            return _generate_text_with_groq(
                prompt,
                model=_groq_model(),
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except SQLGenerationError as exc:
            groq_error = exc

        assert groq_error is not None
        subcode = gemini_exc.subcode if gemini_exc.subcode != "UNKNOWN" else groq_error.subcode
        raise SQLGenerationError(
            f"{gemini_exc} | Groq fallback failed: {groq_error}",
            subcode=subcode,
        ) from groq_error


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

        text = _generate_text_with_fallback(
            prompt,
            models=_plan_models(),
            attempts=_attempts_per_model("REPORTS_PLAN_ATTEMPTS_PER_MODEL", 2),
            max_output_tokens=8192,
            temperature=0.2,
        )

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
        subcode = getattr(exc, "subcode", "")
        if subcode in {"RATE_LIMITED", "QUOTA_EXCEEDED"}:
            logger.warning("plan-report: provider limit reached: %s", exc)
            return _error(
                "RATE_LIMITED",
                "AI provider rate limit/quota reached. Please retry later.",
            )
        if subcode == "MODEL_UNAVAILABLE":
            logger.warning("plan-report: provider/model unavailable: %s", exc)
            return _error(
                "PROVIDER_UNAVAILABLE",
                "AI provider is temporarily overloaded right now. Please retry in a few seconds.",
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

        text = _generate_text_with_fallback(
            prompt,
            models=_narrate_models(),
            attempts=_attempts_per_model("REPORTS_NARRATE_ATTEMPTS_PER_MODEL", 2),
            max_output_tokens=8192,
            temperature=0.25,
        )

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
        if getattr(exc, "subcode", "") in {"RATE_LIMITED", "QUOTA_EXCEEDED", "MODEL_UNAVAILABLE"}:
            logger.warning("narrate-section: provider limit reached, returning empty narrative")
            return NarrateSectionSuccess(status="success", narrative="")
        logger.error("narrate-section: SQLGenerationError: %s", exc)
        return _error("GENERATION_FAILED", str(exc))
    except Exception as exc:
        logger.exception("narrate-section: unexpected error")
        return _error("INTERNAL_ERROR", f"Unexpected error: {type(exc).__name__}")


# ── Narrate Report (batched) ──────────────────────────────────────────────────

def _all_empty_narratives(headings: List[str]) -> NarrateReportSuccess:
    return NarrateReportSuccess(
        status="success",
        sections=[NarratedSection(heading=h, narrative="") for h in headings],
    )


def run_narrate_report(request: NarrateReportRequest) -> NarrateReportSuccess | ReportAiError:
    """
    Single batched LLM call that produces narratives for ALL sections of a report at once.
    Sections with empty SQL results are skipped and returned with an empty narrative.
    """
    try:
        original_headings = [s.heading for s in request.sections]

        # Partition into narratable vs skip (empty data).
        narratable_indices: List[int] = []
        narratable_payload: List[Dict[str, Any]] = []
        for idx, section in enumerate(request.sections):
            is_sql_backed = bool(section.sql and section.sql.strip())
            has_data = not _sample_rows_is_empty(section.sample_rows_json or "[]")

            # Rules:
            # - SQL-backed section with no data → skip (empty narrative).
            # - SQL-backed section with data → narrate.
            # - Text-only section (no SQL) → narrate (LLM writes from the overall topic).
            if is_sql_backed and not has_data:
                continue

            narratable_indices.append(idx)
            narratable_payload.append({
                "heading": section.heading,
                "sql": section.sql,
                "chart_type": section.chart_type,
                "sample_rows_json": section.sample_rows_json,
            })

        if not narratable_payload:
            logger.info("narrate-report: all sections empty, returning empty narratives for %d sections",
                        len(original_headings))
            return _all_empty_narratives(original_headings)

        prompt = build_narrate_report_prompt(
            user_prompt=request.user_prompt,
            title=request.title,
            summary=request.summary,
            sections=narratable_payload,
        )

        text = _generate_text_with_fallback(
            prompt,
            models=_narrate_models(),
            attempts=_attempts_per_model("REPORTS_NARRATE_ATTEMPTS_PER_MODEL", 2),
            max_output_tokens=8192,
            temperature=0.3,
        )

        data = _extract_json_object(text)
        if not data or not isinstance(data.get("sections"), list):
            logger.warning("narrate-report: LLM returned no parseable sections. Raw (truncated): %s", text[:400])
            return _all_empty_narratives(original_headings)

        raw_sections = data["sections"]

        # Build map heading -> narrative from the LLM response (heading-based merge
        # is more robust than position-based if the LLM reorders).
        by_heading: Dict[str, str] = {}
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            h = item.get("heading")
            n = item.get("narrative")
            if isinstance(h, str) and isinstance(n, str):
                by_heading[h.strip()] = n.strip()

        # Assemble in original order; empty-data sections get "".
        result_sections: List[NarratedSection] = []
        narratable_position = 0
        for idx, heading in enumerate(original_headings):
            if idx in narratable_indices:
                narrative = by_heading.get(heading.strip(), "")
                # Fallback to positional match if heading didn't match (LLM drift).
                if not narrative and narratable_position < len(raw_sections):
                    pos_item = raw_sections[narratable_position]
                    if isinstance(pos_item, dict) and isinstance(pos_item.get("narrative"), str):
                        narrative = pos_item["narrative"].strip()
                narratable_position += 1
                result_sections.append(NarratedSection(heading=heading, narrative=narrative))
            else:
                result_sections.append(NarratedSection(heading=heading, narrative=""))

        logger.info("narrate-report: generated %d narratives (%d skipped) for '%s'",
                    len(narratable_payload),
                    len(original_headings) - len(narratable_payload),
                    request.title)
        return NarrateReportSuccess(status="success", sections=result_sections)

    except SQLGenerationError as exc:
        if getattr(exc, "subcode", "") in {"RATE_LIMITED", "QUOTA_EXCEEDED", "MODEL_UNAVAILABLE"}:
            logger.warning("narrate-report: provider limit reached, returning empty narratives")
            return _all_empty_narratives([s.heading for s in request.sections])
        logger.error("narrate-report: SQLGenerationError: %s", exc)
        return _error("GENERATION_FAILED", str(exc))
    except Exception as exc:
        logger.exception("narrate-report: unexpected error")
        return _error("INTERNAL_ERROR", f"Unexpected error: {type(exc).__name__}")
