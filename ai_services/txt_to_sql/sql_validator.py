"""
sql_validator.py

AST-based SQL validation for LLM-generated queries using sqlglot.
The validator is strict and fails closed to preserve security.

SaaS-ready:
- Multi-dialect support
- Nested schema support (table -> {column -> type})
- Strict column qualification
- Dialect-aware function validation
- Returns normalized qualified SQL
"""

from __future__ import annotations

from typing import Dict, Set

from sqlglot import exp, parse
from sqlglot.errors import ParseError
from sqlglot.optimizer.qualify import qualify


class SQLValidationError(Exception):
    """Raised when SQL fails validation rules."""

    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


def _validation_error(subcode: str, message: str) -> SQLValidationError:
    return SQLValidationError(message=message, subcode=subcode)


# =========================================================
# Dialect-Specific Allowed Functions
# =========================================================

ALLOWED_FUNCTIONS = {
    "postgres": {
        "sum", "count", "avg", "min", "max",
        "coalesce", "nullif", "round", "cast",
        "lower", "upper", "trim", "length", "substring",
        "abs", "floor", "ceil",
        "date_trunc", "timestamp_trunc", "date_part", "extract",
        "now", "current_timestamp", "current_date",
        "case", "lag", "row_number", "rank", "dense_rank",
    },
    "mysql": {
        "sum", "count", "avg", "min", "max",
        "coalesce", "nullif", "round", "cast",
        "lower", "upper", "trim", "length", "substring",
        "abs", "floor", "ceil",
        "now", "curdate", "current_date", "date_format", "time_to_str",
        "year", "month", "day", "ts_or_ds_to_date",
        "case", "lag", "row_number", "rank", "dense_rank",
    },
    "sqlserver": {
        "sum", "count", "avg", "min", "max",
        "coalesce", "nullif", "round", "cast",
        "lower", "upper", "trim", "length", "len", "substring",
        "abs", "floor", "ceil", "ceiling",
        "getdate", "current_timestamp",
        "dateadd", "date_add", "datediff", "datepart", "extract",
        "time_str_to_time",
        "case", "lag", "row_number", "rank", "dense_rank",
    },
}


# =========================================================
# Identifier Normalization
# =========================================================

def _normalize_role_schema(
    role_schema: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    Normalize table/column identifiers from the API role schema to lowercase.

    This aligns matching with dialect behavior where unquoted identifiers are
    case-insensitive (e.g., Postgres folds to lowercase).
    """

    normalized_schema: Dict[str, Dict[str, str]] = {}

    for table_name, columns in role_schema.items():
        normalized_table = table_name.lower()

        if normalized_table in normalized_schema and table_name != normalized_table:
            raise _validation_error(
                "SCHEMA_RESOLUTION_ERROR",
                f"Schema normalization conflict on table: '{table_name}'",
            )

        normalized_columns: Dict[str, str] = {}
        for column_name, column_type in columns.items():
            normalized_column = column_name.lower()
            if normalized_column in normalized_columns and column_name != normalized_column:
                raise _validation_error(
                    "SCHEMA_RESOLUTION_ERROR",
                    (
                        "Schema normalization conflict on column "
                        f"'{column_name}' in table '{table_name}'"
                    ),
                )
            normalized_columns[normalized_column] = column_type

        normalized_schema[normalized_table] = normalized_columns

    return normalized_schema


# =========================================================
# Main Validation Entry
# =========================================================

def validate_sql(
    sql: str,
    role_schema: Dict[str, Dict[str, str]],
    dialect: str = "postgres",
) -> str:
    """
    Validate a SQL query against strict safety and schema rules.

    Args:
        sql: LLM-generated SQL string.
        role_schema: allowed tables/columns (nested schema).
        dialect: SQL dialect.

    Returns:
        Qualified and normalized SQL string.

    Raises:
        SQLValidationError
    """

    if not sql or not sql.strip():
        raise _validation_error("EMPTY_SQL", "SQL is empty")

    raw = sql.strip()
    normalized_role_schema = _normalize_role_schema(role_schema)

    _reject_markdown_or_comments(raw)

    sqlglot_dialect = _to_sqlglot_dialect(dialect)

    try:
        statements = parse(raw, read=sqlglot_dialect)
    except ParseError as exc:
        raise _validation_error("PARSE_ERROR", f"SQL parse error: {exc}")

    if len(statements) != 1:
        raise _validation_error("MULTI_STATEMENT", "Multiple SQL statements are not allowed")

    statement = statements[0]

    # Enforce wildcard policy before qualify() can expand SELECT * to explicit columns.
    _reject_wildcards(statement)

    # STRICT qualification (fail on unresolved or ambiguous columns)
    try:
        qualified = qualify(
            statement,
            schema=normalized_role_schema,
            validate_qualify_columns=True,
        )
    except Exception as exc:
        raise _validation_error("SCHEMA_RESOLUTION_ERROR", f"Schema resolution error: {exc}")

    _validate_statement(qualified, normalized_role_schema, dialect)

    # Return normalized SQL
    return qualified.sql(dialect=sqlglot_dialect)


# =========================================================
# Structural Validations
# =========================================================

def _reject_markdown_or_comments(sql: str) -> None:
    if "```" in sql or "`" in sql:
        raise _validation_error("MARKDOWN_OR_COMMENTS", "Markdown or code formatting is not allowed in SQL")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise _validation_error("MARKDOWN_OR_COMMENTS", "SQL comments are not allowed")


def _require_select_only(statement: exp.Expression) -> None:
    if not isinstance(statement, exp.Select):
        raise _validation_error("NON_SELECT_QUERY", "Only SELECT queries are allowed")

    if statement.find(exp.Insert) or statement.find(exp.Update) or statement.find(exp.Delete):
        raise _validation_error("NON_SELECT_QUERY", "Only SELECT queries are allowed")


def _reject_select_into(statement: exp.Expression) -> None:
    if statement.find(exp.Into):
        raise _validation_error("SELECT_INTO", "SELECT INTO is not allowed")


def _reject_wildcards(statement: exp.Expression) -> None:
    for star in statement.find_all(exp.Star):
        # Allow COUNT(*)
        if isinstance(star.parent, exp.Count):
            continue
        raise _validation_error(
            "WILDCARD_SELECT",
            "Wildcard '*' is not allowed; select explicit columns"
        )


def _select_has_no_from(statement: exp.Expression) -> bool:
    if not isinstance(statement, exp.Select):
        return False
    # sqlglot stores FROM as "from_" in parsed/qualified Select nodes.
    return statement.args.get("from") is None and statement.args.get("from_") is None


# =========================================================
# Core Validation
# =========================================================

def _validate_statement(
    statement: exp.Expression,
    role_schema: Dict[str, Dict[str, str]],
    dialect: str,
) -> None:
    _validate_ctes(statement)

    if statement.find(exp.Union):
        raise _validation_error("UNION_NOT_SUPPORTED", "UNION and UNION ALL are not supported")

    _require_select_only(statement)
    _reject_select_into(statement)
    _reject_wildcards(statement)

    if _select_has_no_from(statement):
        raise _validation_error("SELECT_WITHOUT_FROM", "SELECT without FROM is not allowed")

    _validate_functions(statement, dialect)
    _validate_schema_access(statement, role_schema)


# =========================================================
# Function Validation
# =========================================================

def _validate_ctes(statement: exp.Expression) -> None:
    with_clause = statement.args.get("with") or statement.args.get("with_")
    if not with_clause:
        return

    if with_clause.args.get("recursive"):
        raise _validation_error("RECURSIVE_CTE", "Recursive CTEs (WITH RECURSIVE) are not supported")


def _to_sqlglot_dialect(dialect: str) -> str:
    dialect_normalized = dialect.lower()
    if dialect_normalized == "sqlserver":
        return "tsql"
    return dialect_normalized


def _validate_functions(statement: exp.Expression, dialect: str) -> None:
    dialect_key = dialect.lower()
    if dialect_key == "tsql":
        dialect_key = "sqlserver"

    allowed = ALLOWED_FUNCTIONS.get(
        dialect_key,
        {"sum", "count", "avg", "min", "max"},
    )

    # Block schema-qualified functions (e.g., pg_catalog.now())
    for prop in statement.find_all(exp.Property):
        if isinstance(prop.this, exp.Func):
            raise _validation_error(
                "SCHEMA_QUALIFIED_FUNCTION",
                f"Schema-qualified functions are not allowed: '{prop.sql()}'"
            )

    # Handle parser outputs like pg_catalog.now() represented as Dot(..., Func)
    for dot in statement.find_all(exp.Dot):
        if isinstance(dot.expression, exp.Func):
            raise _validation_error(
                "SCHEMA_QUALIFIED_FUNCTION",
                f"Schema-qualified functions are not allowed: '{dot.sql()}'"
            )

    for func in statement.find_all(exp.Func):
        # Logical connectors are not function calls and should not be checked
        # against dialect allowlists.
        if isinstance(func, exp.Connector):
            continue

        # CASE expressions are represented with internal IF nodes in sqlglot.
        if isinstance(func, exp.If) and isinstance(func.parent, exp.Case):
            continue

        canonical_name = func.sql_name()
        raw_name = getattr(func, "name", None)

        candidates = {
            name.lower()
            for name in (canonical_name, raw_name)
            if isinstance(name, str) and name.strip()
        }

        if not candidates:
            raise _validation_error("UNKNOWN_FUNCTION", "Unknown SQL function detected")

        for name in candidates:
            if "." in name:
                raise _validation_error("SCHEMA_QUALIFIED_FUNCTION", "Schema-qualified functions are not allowed")

        if not any(name in allowed for name in candidates):
            # Prefer canonical sqlglot name in the error for consistency.
            rejected = (canonical_name or raw_name or "unknown").lower()
            raise _validation_error("FUNCTION_NOT_ALLOWED", f"Function not allowed: '{rejected}'")


# =========================================================
# Schema Validation
# =========================================================

def _validate_schema_access(
    statement: exp.Expression,
    role_schema: Dict[str, Dict[str, str]],
) -> None:

    allowed_tables: Set[str] = set(role_schema.keys())
    alias_to_table: Dict[str, str] = {}
    cte_names: Set[str] = set()
    derived_aliases: Set[str] = set()
    select_aliases: Set[str] = set()

    if isinstance(statement, exp.Select):
        for expression in statement.expressions:
            if isinstance(expression, exp.Alias) and expression.alias:
                select_aliases.add(expression.alias)

    for cte in statement.find_all(exp.CTE):
        if cte.alias:
            cte_names.add(cte.alias)

    for table in statement.find_all(exp.Table):
        name = table.name

        if not name:
            raise _validation_error("TABLE_NAME_MISSING", "Table name is missing")

        if table.db and table.db.lower() in {"pg_catalog", "information_schema"}:
            raise _validation_error(
                "SYSTEM_SCHEMA_ACCESS",
                f"Access to system schema is not allowed: '{table.db}'"
            )

        if name in cte_names:
            alias = table.alias
            if alias:
                alias_to_table[alias] = name
            continue

        if name not in allowed_tables:
            raise _validation_error(
                "TABLE_NOT_ALLOWED",
                f"Table not allowed for this role: '{name}'"
            )

        alias = table.alias
        if alias:
            alias_to_table[alias] = name

    # Allow columns that come from derived-table aliases (subqueries in FROM/JOIN).
    for subquery in statement.find_all(exp.Subquery):
        if subquery.alias:
            derived_aliases.add(subquery.alias)

    virtual_sources = cte_names | derived_aliases

    for column in statement.find_all(exp.Column):
        table = column.table

        # After qualify(), every column should have table
        if not table:
            # SELECT aliases are valid in ORDER BY / GROUP BY clauses.
            if column.name in select_aliases:
                continue
            raise _validation_error(
                "UNRESOLVED_COLUMN",
                f"Unresolved column detected: '{column.name}'"
            )

        resolved_table = alias_to_table.get(table, table)

        if resolved_table in virtual_sources:
            continue

        if resolved_table not in role_schema:
            raise _validation_error(
                "UNAUTHORIZED_TABLE_ACCESS",
                f"Unauthorized table access: '{table}'"
            )

        if column.name not in role_schema[resolved_table]:
            raise _validation_error(
                "UNAUTHORIZED_COLUMN_ACCESS",
                f"Unauthorized access to {resolved_table}.{column.name}"
            )
