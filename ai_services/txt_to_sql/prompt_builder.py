from typing import Dict, List


def _format_schema(schema: Dict[str, List[str]]) -> str:
    """
    Convert schema dict into a readable text format for the LLM.
    """
    lines = []
    for table, columns in schema.items():
        lines.append(f"Table: {table}")
        for col in columns:
            lines.append(f"  - {col}")
        lines.append("")
    return "\n".join(lines).strip()


def build_sql_prompt(question: str, role: str, schema: Dict[str, List[str]],) -> str:
    """
    Build a strict, role-aware Text-to-SQL prompt.
    """

    schema_text = _format_schema(schema)

    prompt = f"""
You are a senior data analyst generating SQL queries.

Rules you MUST follow:
- Use ONLY the tables and columns listed below
- Generate ONLY a SELECT query
- DO NOT use INSERT, UPDATE, DELETE, DROP, or ALTER
- DO NOT guess table or column names
- DO NOT add explanations, comments, or markdown
- Output SQL only
- Use PostgreSQL syntax

User role: {role}

Database schema:
{schema_text}

User question:
{question}

SQL query:
""".strip()

    return prompt
