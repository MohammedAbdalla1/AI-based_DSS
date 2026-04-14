"""
prompt_builder.py

Builds LLM prompts for the two report AI endpoints.
"""

from __future__ import annotations

from typing import Dict

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
) -> str:
    """
    Prompt that instructs the LLM to write a detailed narrative
    for one report section, grounded in the actual query results.
    """
    sql_is_present = bool(sql.strip())
    sample_rows_clean = sample_rows_json.strip()
    has_no_data = sample_rows_clean in ("", "[]", "null")

    sql_block = f"SQL executed:\n{sql}" if sql_is_present else "No SQL query for this section."
    data_block = sample_rows_clean if not has_no_data else "No data returned."

    if sql_is_present and has_no_data:
        narrative_instruction = (
            "No rows were returned for this SQL query. "
            "Return an empty narrative in JSON format."
        )
    else:
        narrative_instruction = (
            "Write a detailed narrative of 180-320 words (roughly 8-12 sentences). "
            "Explain what the numbers mean, why they matter, notable patterns, "
        "possible business drivers, risks, and practical next actions for a small-business owner."
        )

    return f"""You are an experienced small-business analyst writing one section of a report.
  Your audience is non-technical small-business owners and operators.

OVERALL REPORT TOPIC: {user_prompt}
SECTION: {heading}

{sql_block}

DATA SAMPLE (up to 20 rows):
{data_block}

{narrative_instruction}
Guidelines:
- Open with a clear executive takeaway in 1-2 sentences
- Deep-dive into key trends, comparisons, and outliers using concrete values from the data
- Add context and interpretation: explain likely reasons behind movements or differences
- Include 2-4 practical recommendations the owner can apply this week
- Use evidence-based reasoning (for example: seasonality effects, pricing elasticity, inventory turnover, customer cohort behavior)
- Use simple language and short sentences; avoid jargon where possible
- Include caveats where uncertainty exists and avoid overclaiming
- End with next-step actions and one optional follow-up question
- If helpful, suggest one beginner-friendly learning resource (book, article, or course topic) by title only, without links
- Keep language rich and descriptive, but still clear for non-technical business readers
- Do not invent facts that are not present in the supplied rows
- Do not fabricate citations, URLs, or research statistics
- Do not say "this chart shows", "the data shows", or "as we can see" — state insights directly
- Use readable formatting: 3-5 short paragraphs separated by blank lines
- Avoid one giant paragraph
- Optional: use lightweight labels such as "Key takeaway:", "Why this matters:", "Recommended action:" when helpful
- Keep format natural and varied; do not force the exact same template every time
- Plain text only — no markdown tables, no code blocks

Return ONLY a JSON object — no extra text:
{{"narrative": "your detailed narrative here"}}""".strip()
