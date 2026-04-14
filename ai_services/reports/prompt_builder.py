"""
prompt_builder.py

Builds LLM prompts for the two report AI endpoints.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from txt_to_sql.prompt_builder import _format_schema, FUNCTION_HINTS


# ── Plan Report ───────────────────────────────────────────────────────────────

def build_plan_report_prompt(
    schema: Dict[str, Dict[str, str]],
    db_type: str,
    user_prompt: str,
    max_sections: int = 5,
) -> str:
    """
    Prompt that instructs the LLM to plan a full multi-section report.
    Returns a JSON object: {title, summary, sections: [{heading, sql, chart_type, chart_config}]}
    """
    schema_text = _format_schema(schema)
    fn_hints = FUNCTION_HINTS.get(db_type.lower(), "COUNT, SUM, AVG, MIN, MAX")

    return f"""You are a senior business intelligence analyst creating a structured data report.

The user wants a report on: {user_prompt}

Your task is to plan {max_sections} cohesive report sections that together give a complete picture.
Each section covers a distinct angle: totals, trends, breakdowns, comparisons, rankings, etc.

DATABASE DIALECT: {db_type}
AVAILABLE FUNCTIONS: {fn_hints}

DATABASE SCHEMA:
{schema_text}

SQL RULES (strictly enforced by the execution engine):
- SELECT queries only — no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE
- Every query must include a FROM clause
- Use only tables and columns listed in the schema above — do NOT guess or hallucinate names
- Do NOT use SELECT * — list explicit columns with aliases
- Do NOT use UNION, UNION ALL, or WITH RECURSIVE
- Non-recursive CTEs (WITH ...) are allowed
- Use {db_type} dialect and only the listed functions

CHART TYPE OPTIONS (pick the most meaningful for each section):
Bar family:    bar, horizontal_bar, stacked_bar, grouped_bar
Line family:   line, area
Circular:      pie, doughnut  (only when ≤8 categories)
Other:         scatter, radar, treemap, funnel, composed
Use null for a text-only intro or summary section (no SQL needed).

CHART CONFIG FORMAT per chart type:
- bar / horizontal_bar / line / area:  {{"xAxis": "col", "yAxis": ["col"], "colors": ["#hex"]}}
- stacked_bar / grouped_bar:           {{"xAxis": "col", "yAxis": ["col1","col2"], "stacked": true/false}}
- pie / doughnut:                      {{"labelKey": "col", "valueKey": "col"}}
- scatter:                             {{"xAxis": "col", "yAxis": "col"}}
- radar:                               {{"xAxis": "col", "yAxis": ["col1","col2"]}}
- treemap / funnel:                    {{"labelKey": "col", "valueKey": "col"}}
- composed:                            {{"xAxis": "col", "barKeys": ["col"], "lineKeys": ["col"]}}
- Column names in chart_config MUST exactly match the aliases used in the SQL SELECT clause.

Return ONLY a JSON object — no markdown, no explanation, no extra text:
{{
  "title": "Short descriptive report title (5-10 words)",
  "summary": "1-2 sentence executive summary of the whole report",
  "sections": [
    {{
      "heading": "Section heading",
      "sql": "SELECT ... FROM ... (or null for a text-only section)",
      "chart_type": "bar|line|pie|...|null",
      "chart_config": {{ ... }}
    }}
  ]
}}

Respond with ONLY the JSON object:""".strip()


# ── Narrate Section ───────────────────────────────────────────────────────────

def build_narrate_section_prompt(
    heading: str,
    user_prompt: str,
    sql: str,
    sample_rows_json: str,
    chart_type: Optional[str] = None,
) -> str:
    """
    Prompt that instructs the LLM to write a detailed narrative
    for one report section, grounded in the actual query results.
    """
    sql_is_present = bool(sql.strip())
    sample_rows_clean = sample_rows_json.strip()
    has_no_data = sample_rows_clean in ("", "[]", "null")
    chart_type_clean = (chart_type or "").strip() or "unspecified"

    sql_block = f"SQL executed:\n{sql}" if sql_is_present else "No SQL query for this section."
    data_block = sample_rows_clean if not has_no_data else "No data returned."
    chart_block = f"CHART TYPE: {chart_type_clean}"

    if sql_is_present and has_no_data:
        narrative_instruction = (
            "No rows were returned for this SQL query. "
            "Return an empty narrative in JSON format."
        )
    else:
        narrative_instruction = (
            "Write a detailed narrative of 240-340 words (roughly 9-13 sentences). "
            "Explain what the numbers mean, why they matter, notable patterns, "
            "possible business drivers, risks, practical next actions, and a short forward-looking outlook."
        )

    return f"""You are an experienced small-business analyst writing one section of a report.
  Your audience is non-technical small-business owners and operators.

OVERALL REPORT TOPIC: {user_prompt}
SECTION: {heading}

{sql_block}
{chart_block}

DATA SAMPLE (up to 20 rows):
{data_block}

{narrative_instruction}
Guidelines:
- Open with a clear executive takeaway in 1-2 sentences
- Deep-dive into key trends, comparisons, and outliers using concrete values from the data
- Mention at least 3 concrete numbers, ratios, or percentages from the rows when available
- Use the chart type as context for interpretation (trend, composition, ranking, correlation)
- For ranking/comparison sections, name at least one top performer and one lagging segment with values when available
- Add context and interpretation: explain likely reasons behind movements or differences
- Include 2-3 practical recommendations the owner can apply this week
- Include a brief 30-90 day forward look or prediction in 1-2 sentences; frame it as an estimate, not certainty
- Use evidence-based reasoning (for example: seasonality effects, pricing elasticity, inventory turnover, customer cohort behavior)
- Use simple language and short sentences; avoid jargon where possible
- Include caveats where uncertainty exists and avoid overclaiming
- End with next-step actions and one optional follow-up question
- If helpful, suggest one beginner-friendly learning resource (book, article, or course topic) by title only, without links
- Keep language rich and descriptive, but still clear for non-technical business readers
- Do not invent facts that are not present in the supplied rows
- Do not fabricate citations, URLs, or research statistics
- Do not say "this chart shows", "the data shows", or "as we can see" — state insights directly
- Use readable formatting: exactly 3 short paragraphs separated by blank lines
- Paragraph structure:
    1) performance snapshot with key figures and comparisons
    2) interpretation of likely drivers, risks, and opportunities
    3) concrete actions for this week plus a short future outlook
- Avoid one giant paragraph
- Optional: use lightweight labels such as "Key takeaway:", "Why this matters:", "Recommended action:" when helpful
- Keep format natural and varied; do not force the exact same template every time
- Plain text only — no markdown tables, no code blocks

Return ONLY a JSON object — no extra text:
{{"narrative": "your detailed narrative here"}}""".strip()


# ── Narrate Report (batched, single LLM call for all sections) ────────────────

def _truncate_rows(rows_json: str, max_rows: int = 15) -> str:
    """Trim a JSON rows array to at most `max_rows` entries to control input tokens."""
    raw = (rows_json or "").strip()
    if not raw or raw.lower() == "null":
        return "[]"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "[]"
    if not isinstance(parsed, list):
        return "[]"
    return json.dumps(parsed[:max_rows], ensure_ascii=False)


def build_narrate_report_prompt(
    user_prompt: str,
    title: str,
    summary: Optional[str],
    sections: List[Dict[str, Any]],
) -> str:
    """
    Single-prompt batched narration for an entire report.
    `sections` is a list of dicts with keys: heading, sql, chart_type, sample_rows_json.
    Callers must pass ONLY sections that have non-empty data.
    Returns a prompt that instructs the LLM to return all narratives in one JSON response.
    """
    summary_block = f"REPORT SUMMARY: {summary.strip()}\n" if summary and summary.strip() else ""

    section_blocks: List[str] = []
    for i, s in enumerate(sections, start=1):
        heading = s.get("heading", f"Section {i}")
        sql = (s.get("sql") or "").strip()
        chart_type = (s.get("chart_type") or "").strip() or "unspecified"
        rows = _truncate_rows(s.get("sample_rows_json") or "[]", max_rows=18)
        sql_line = f"SQL: {sql}" if sql else "SQL: (text-only section, no query)"
        section_blocks.append(
            f"--- SECTION {i}: {heading} ---\n"
            f"{sql_line}\n"
            f"CHART TYPE: {chart_type}\n"
            f"DATA (up to 18 rows): {rows}"
        )

    sections_text = "\n\n".join(section_blocks)

    return f"""You are an experienced small-business analyst writing a full multi-section report.
Your audience is non-technical small-business owners and operators.

OVERALL REPORT TOPIC: {user_prompt}
REPORT TITLE: {title}
{summary_block}
You will receive {len(sections)} sections below. For EACH section, write a clear, grounded narrative
of 220-320 words (about 8-12 sentences). Base every claim on the numbers in that section's data only.

For every section's narrative:
- Open with a 1-2 sentence executive takeaway using concrete values from the data
- Call out the key trend, comparison, or outlier with specific numbers
- Include at least 3 concrete numbers, ratios, or percentages when available
- Use the chart type to guide interpretation (trend, composition, ranking, correlation)
- For ranking/comparison sections, name at least one top performer and one lagging segment with values when available
- Offer 2-3 practical next-step actions the owner can apply this week
- Add a short future outlook (30-90 days) in 1-2 sentences as an estimate, with a caveat
- Use simple language, short sentences, no jargon
- Do not say "this chart shows", "the data shows", or "as we can see" — state insights directly
- Do not invent facts, citations, URLs, or numbers that are not in the rows
- Plain text only — no markdown, no code blocks, no tables
- Use exactly 3 short paragraphs separated by blank lines; avoid a single wall of text
- Paragraph structure:
    1) performance snapshot with key figures and comparisons
    2) interpretation of likely drivers, risks, and opportunities
    3) concrete actions for this week plus a short future outlook
- Optional lightweight labels like "Key takeaway:" or "Recommended action:" when helpful

SECTIONS:

{sections_text}

Return ONLY a JSON object in this exact shape — no markdown, no explanation, no extra text.
The `sections` array MUST contain exactly {len(sections)} entries in the SAME ORDER as above,
and each entry's `heading` MUST match the section heading verbatim:

{{
  "sections": [
    {{"heading": "<heading 1 verbatim>", "narrative": "<narrative for section 1>"}},
    {{"heading": "<heading 2 verbatim>", "narrative": "<narrative for section 2>"}}
  ]
}}

Respond with ONLY the JSON object:""".strip()
