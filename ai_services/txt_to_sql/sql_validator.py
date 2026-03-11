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

from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

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


BLOCKED_SYSTEM_SCHEMAS = {"pg_catalog", "information_schema"}


@dataclass(frozen=True)
class NormalizedRoleSchema:
    access_schema: Dict[str, Dict[str, str]]
    qualify_schema: Dict[str, Any]
    table_path_to_key: Dict[Tuple[str, ...], str]
    unqualified_to_key: Dict[str, str]


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
) -> NormalizedRoleSchema:
    """
    Normalize table/column identifiers from the API role schema to lowercase.

    This aligns matching with dialect behavior where unquoted identifiers are
    case-insensitive (e.g., Postgres folds to lowercase).
    """

    access_schema: Dict[str, Dict[str, str]] = {}
    qualify_schema: Dict[str, Any] = {}
    table_path_to_key: Dict[Tuple[str, ...], str] = {}
    table_name_to_keys: Dict[str, Set[str]] = {}

    for table_name, columns in role_schema.items():
        normalized_parts = _normalize_table_identifier_parts(table_name)
        normalized_table = ".".join(normalized_parts)

        if normalized_table in access_schema and table_name != normalized_table:
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

        access_schema[normalized_table] = normalized_columns
        table_path_to_key[normalized_parts] = normalized_table
        table_name_to_keys.setdefault(normalized_parts[-1], set()).add(normalized_table)

        _insert_qualify_schema_entry(
            qualify_schema=qualify_schema,
            table_parts=normalized_parts,
            columns=normalized_columns,
            source_table_name=table_name,
        )

    unqualified_to_key: Dict[str, str] = {}
    for table_name, candidates in table_name_to_keys.items():
        if len(candidates) == 1:
            unqualified_to_key[table_name] = next(iter(candidates))

    return NormalizedRoleSchema(
        access_schema=access_schema,
        qualify_schema=qualify_schema,
        table_path_to_key=table_path_to_key,
        unqualified_to_key=unqualified_to_key,
    )


def _normalize_table_identifier_parts(table_name: str) -> Tuple[str, ...]:
    parts = [part.strip().lower() for part in table_name.split(".")]
    if not parts or any(not part for part in parts):
        raise _validation_error(
            "SCHEMA_RESOLUTION_ERROR",
            f"Invalid table identifier in role_schema: '{table_name}'",
        )
    if len(parts) > 3:
        raise _validation_error(
            "SCHEMA_RESOLUTION_ERROR",
            f"Unsupported table identifier depth in role_schema: '{table_name}'",
        )
    return tuple(parts)


def _is_column_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(v, str) for v in value.values())


def _insert_qualify_schema_entry(
    qualify_schema: Dict[str, Any],
    table_parts: Tuple[str, ...],
    columns: Dict[str, str],
    source_table_name: str,
) -> None:
    node: Dict[str, Any] = qualify_schema

    for index, part in enumerate(table_parts):
        is_leaf = index == len(table_parts) - 1
        existing = node.get(part)

        if is_leaf:
            if existing is None:
                node[part] = dict(columns)
                return

            if _is_column_mapping(existing):
                if existing != columns:
                    raise _validation_error(
                        "SCHEMA_RESOLUTION_ERROR",
                        f"Schema normalization conflict on table: '{source_table_name}'",
                    )
                return

            raise _validation_error(
                "SCHEMA_RESOLUTION_ERROR",
                f"Schema normalization conflict on table: '{source_table_name}'",
            )

        if existing is None:
            node[part] = {}
            node = node[part]
            continue

        if _is_column_mapping(existing):
            raise _validation_error(
                "SCHEMA_RESOLUTION_ERROR",
                f"Schema normalization conflict on table: '{source_table_name}'",
            )

        node = existing


def _statement_uses_unqualified_tables(statement: exp.Expression) -> bool:
    found_table = False
    for table in statement.find_all(exp.Table):
        if table.name:
            found_table = True
        if table.db or getattr(table, "catalog", None):
            return False
    return found_table


def _build_unqualified_qualify_schema(
    schema_info: NormalizedRoleSchema,
) -> Dict[str, Dict[str, str]]:
    fallback: Dict[str, Dict[str, str]] = {}
    for table_name, resolved_key in schema_info.unqualified_to_key.items():
        fallback[table_name] = schema_info.access_schema[resolved_key]
    return fallback


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
            statement.copy(),
            schema=normalized_role_schema.qualify_schema,
            validate_qualify_columns=True,
        )
    except Exception as exc:
        should_try_unqualified = _statement_uses_unqualified_tables(statement)
        fallback_schema = (
            _build_unqualified_qualify_schema(normalized_role_schema)
            if should_try_unqualified
            else {}
        )

        if fallback_schema:
            try:
                qualified = qualify(
                    statement.copy(),
                    schema=fallback_schema,
                    validate_qualify_columns=True,
                )
            except Exception:
                raise _validation_error("SCHEMA_RESOLUTION_ERROR", f"Schema resolution error: {exc}")
        else:
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
    role_schema: Dict[str, Dict[str, str]] | NormalizedRoleSchema,
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
    role_schema: Dict[str, Dict[str, str]] | NormalizedRoleSchema,
) -> None:
    schema_info = _ensure_normalized_role_schema(role_schema)
    normalized_schema = schema_info.access_schema

    alias_to_table: Dict[str, str] = {}
    cte_names: Set[str] = set()
    derived_aliases: Set[str] = set()
    select_aliases: Set[str] = set()

    if isinstance(statement, exp.Select):
        for expression in statement.expressions:
            if isinstance(expression, exp.Alias) and expression.alias:
                select_aliases.add(expression.alias.lower())

    for cte in statement.find_all(exp.CTE):
        if cte.alias:
            cte_names.add(cte.alias.lower())

    for table in statement.find_all(exp.Table):
        table_parts = _table_identifier_parts(table)
        if not table_parts:
            raise _validation_error("TABLE_NAME_MISSING", "Table name is missing")

        name = table_parts[-1]
        schema_part = table_parts[-2] if len(table_parts) >= 2 else None

        if schema_part and schema_part in BLOCKED_SYSTEM_SCHEMAS:
            raise _validation_error(
                "SYSTEM_SCHEMA_ACCESS",
                f"Access to system schema is not allowed: '{schema_part}'"
            )

        if name in cte_names:
            alias = table.alias
            if alias:
                alias_to_table[alias.lower()] = name
            continue

        resolved_table_key = _resolve_table_key(
            table_parts=table_parts,
            schema_info=schema_info,
            allow_unqualified_fallback=True,
        )
        if not resolved_table_key:
            raise _validation_error(
                "TABLE_NOT_ALLOWED",
                f"Table not allowed for this role: '{'.'.join(table_parts)}'"
            )

        alias = table.alias
        if alias:
            alias_to_table[alias.lower()] = resolved_table_key

    # Allow columns that come from derived-table aliases (subqueries in FROM/JOIN).
    for subquery in statement.find_all(exp.Subquery):
        if subquery.alias:
            derived_aliases.add(subquery.alias.lower())

    virtual_sources = cte_names | derived_aliases

    for column in statement.find_all(exp.Column):
        table = column.table

        # After qualify(), every column should have table
        if not table:
            # SELECT aliases are valid in ORDER BY / GROUP BY clauses.
            if column.name and column.name.lower() in select_aliases:
                continue
            raise _validation_error(
                "UNRESOLVED_COLUMN",
                f"Unresolved column detected: '{column.name}'"
            )

        table_lookup_key = table.lower()
        resolved_table = alias_to_table.get(table_lookup_key)

        if resolved_table and resolved_table in virtual_sources:
            continue
        if not resolved_table and table_lookup_key in virtual_sources:
            continue

        if not resolved_table:
            resolved_table = _resolve_table_key(
                table_parts=(table_lookup_key,),
                schema_info=schema_info,
                allow_unqualified_fallback=True,
            )

        if not resolved_table:
            raise _validation_error(
                "UNAUTHORIZED_TABLE_ACCESS",
                f"Unauthorized table access: '{table}'"
            )

        column_name = (column.name or "").lower()
        if column_name not in normalized_schema[resolved_table]:
            raise _validation_error(
                "UNAUTHORIZED_COLUMN_ACCESS",
                f"Unauthorized access to {resolved_table}.{column_name}"
            )


def _ensure_normalized_role_schema(
    role_schema: Dict[str, Dict[str, str]] | NormalizedRoleSchema,
) -> NormalizedRoleSchema:
    if isinstance(role_schema, NormalizedRoleSchema):
        return role_schema
    return _normalize_role_schema(role_schema)


def _resolve_table_key(
    table_parts: Tuple[str, ...],
    schema_info: NormalizedRoleSchema,
    allow_unqualified_fallback: bool,
) -> str | None:
    direct_match = schema_info.table_path_to_key.get(table_parts)
    if direct_match:
        return direct_match

    if allow_unqualified_fallback and len(table_parts) == 1:
        return schema_info.unqualified_to_key.get(table_parts[0])

    return None


def _table_identifier_parts(table: exp.Table) -> Tuple[str, ...]:
    name = table.name.lower() if table.name else ""
    if not name:
        return tuple()

    db = table.db.lower() if table.db else ""
    catalog_value = getattr(table, "catalog", None)
    catalog = catalog_value.lower() if isinstance(catalog_value, str) else ""

    parts = []
    if catalog:
        parts.append(catalog)
    if db:
        parts.append(db)
    parts.append(name)
    return tuple(parts)
