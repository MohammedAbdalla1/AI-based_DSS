"""
prompt_builder.py

Builds the LLM prompt for schema-to-analytics chart suggestions.
Reuses schema formatting and function hints from txt_to_sql.
"""

from typing import Dict, Optional

from txt_to_sql.prompt_builder import _format_schema, FUNCTION_HINTS


def build_analytics_prompt(
    schema: Dict[str, Dict[str, str]],
    db_type: str,
    max_suggestions: int = 5,
    user_prompt: Optional[str] = None,
    rag_context: Optional[str] = None,
) -> str:
    """
    Build a prompt that instructs the LLM to analyze a database schema
    and return JSON chart suggestions.
    """

    schema_text = _format_schema(schema)

    function_examples = FUNCTION_HINTS.get(
        db_type.lower(),
        "COUNT, SUM, AVG, MIN, MAX",
    )

    user_intent_section = ""
    if user_prompt:
        user_intent_section = f"""
The user specifically wants to analyze:
{user_prompt}

Prioritize suggestions that match this intent. You may also include other useful analytics.
"""

    rag_context_section = ""
    if rag_context:
        rag_context_section = f"""
Business context (use this to make suggestions more relevant):
{rag_context}
"""

    prompt = f"""You are a senior data analytics engineer. Analyze the database schema below and suggest {max_suggestions} meaningful chart visualizations.

Rules you MUST follow:
- Generate ONLY SELECT queries
- Every SELECT query MUST include a FROM clause
- DO NOT use SELECT *; always list explicit columns
- Use ONLY tables and columns from the provided schema
- DO NOT guess table or column names
- Use ONLY standard functions supported in {db_type} (e.g. {function_examples})
- DO NOT use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE
- DO NOT use WITH RECURSIVE or UNION/UNION ALL
- Use {db_type} SQL dialect

For each suggestion, choose the MOST appropriate chart type — use VARIETY across suggestions, do NOT default everything to "bar":

BAR FAMILY:
- "bar": vertical bars for comparing categories or discrete values
- "horizontal_bar": horizontal bars for ranked comparisons, especially with long category labels
- "stacked_bar": stacked vertical bars showing composition breakdown across categories
- "grouped_bar": side-by-side bars for comparing multiple metrics across categories
- "floating_bar": bars with custom min/max ranges, great for showing ranges (e.g. min-max price per category)

LINE FAMILY:
- "line": smooth lines for time series or continuous trends
- "area": filled area under lines for cumulative trends or volume over time
- "stepped_line": step-style lines for discrete state changes over time (e.g. status changes, pricing tiers)
- "multi_axis_line": multiple lines on separate Y-axes for comparing metrics with different scales

CIRCULAR FAMILY:
- "pie": proportional slices for part-to-whole breakdowns (use only when <= 8 categories)
- "doughnut": ring chart like pie with center cutout, highlights dominant segment (use when <= 8 categories)
- "polar_area": equal-angle segments with varying radius, good for comparing magnitudes across categories

RADIAL:
- "radar": spider/web chart for multi-metric comparison across 3+ dimensions

POINT-BASED:
- "scatter": X/Y scatter plot for correlations between two numeric values
- "bubble": like scatter but with a third dimension shown as bubble size (requires 3 numeric columns: x, y, size)

MIXED:
- "composed": overlay bars + lines on same chart for comparing related metrics on different scales

PLUGIN-BASED:
- "treemap": nested rectangles showing hierarchical category breakdown by size
- "funnel": horizontal bars sorted largest-to-smallest for conversion/pipeline stages

Return your response as a JSON array with exactly this structure (no markdown, no explanation, just the JSON):
[
  {{
    "title": "Short descriptive title",
    "description": "One sentence explaining what this chart shows and why it's useful",
    "sql": "SELECT ... FROM ...",
    "chart_type": "bar|horizontal_bar|stacked_bar|grouped_bar|floating_bar|line|area|stepped_line|multi_axis_line|pie|doughnut|polar_area|radar|scatter|bubble|composed|treemap|funnel",
    "chart_config": {{
      "xAxis": "column_name_for_x_axis",
      "yAxis": ["column_name_for_y_axis"],
      "colors": ["#f97316", "#0ea5e9", "#22c55e", "#f43f5e"],
      "labelKey": "column_for_pie_labels",
      "valueKey": "column_for_pie_values"
    }},
    "insight": "A detailed 6-10 line insight in simple business language, including key numbers, trends, likely causes, risks/opportunities, and practical next actions."
  }}
]

chart_config rules:
- For bar/horizontal_bar/grouped_bar/line/area/stepped_line: include "xAxis" and "yAxis" (array of column names)
- For stacked_bar: same as bar, add "stacked": true
- For floating_bar: include "xAxis", "yAxis" with one column. The SQL must return two numeric columns: a min and max value. Include "minKey" and "maxKey" in chart_config
- For multi_axis_line: include "xAxis" and "yAxis" (array of 2+ columns on different scales). Each gets its own Y-axis
- For pie/doughnut/polar_area: include "labelKey" and "valueKey" instead of xAxis/yAxis
- For scatter: use "xAxis" for one numeric column and "yAxis" for another (single value, not array)
- For bubble: include "xAxis" (x column), "yAxis" (single y column), and "sizeKey" (radius column) — all 3 must be numeric
- For radar: include "xAxis" (the spoke/category label column) and "yAxis" (array of metric columns)
- For composed: include "xAxis", "barKeys" (array of columns for bars), "lineKeys" (array of columns for lines)
- For treemap/funnel: include "labelKey" (category/stage name) and "valueKey" (size/count metric)
- "colors" is optional, but when you provide it use a diverse palette with distinct hues
- Avoid using the same single light-blue color for every chart
- Prefer semantic coloring when meaningful: higher values toward green, lower values toward red
- For multi-series charts: provide one distinct color per series
- For single-series category charts: provide 4-8 colors so categories can be visually distinguished
- Column names in chart_config MUST match the aliases used in the SQL SELECT clause

insight rules:
- The "insight" field is REQUIRED for every suggestion
- Write 6-10 lines (each line should be a short sentence separated by a newline)
- Include concrete numbers from the query output when possible
- Explain why the pattern might be happening (business drivers)
- Mention at least one risk and one opportunity
- End with 2-3 practical recommendations the user can act on
- Keep wording simple for non-technical small-business users
- Do not use markdown or bullet points inside insight text

Database schema:
{schema_text}
{user_intent_section}{rag_context_section}
Suggest {max_suggestions} useful charts as a JSON array:
""".strip()

    return prompt
