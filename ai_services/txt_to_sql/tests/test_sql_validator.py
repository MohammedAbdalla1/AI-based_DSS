from sqlglot import exp, parse_one
import pytest

from txt_to_sql.sql_validator import (
    SQLValidationError,
    _reject_markdown_or_comments,
    _reject_select_into,
    _reject_wildcards,
    _require_select_only,
    _select_has_no_from,
    _validate_functions,
    _validate_schema_access,
    validate_sql,
)


ROLE_SCHEMA = {
    "products": {
        "id": "INT",
        "name": "TEXT",
        "price": "DECIMAL",
        "category_id": "INT",
        "brand_id": "INT",
    },
    "orders": {
        "id": "INT",
        "total_amount": "DECIMAL",
        "order_date": "DATE",
    },
}


QUALIFIED_ROLE_SCHEMA = {
    "dbo.employee": {
        "id": "INT",
        "name": "TEXT",
    }
}


def assert_invalid(sql: str, match: str) -> None:
    with pytest.raises(SQLValidationError, match=match):
        validate_sql(sql, ROLE_SCHEMA)


def assert_invalid_with_subcode(sql: str, match: str, subcode: str) -> None:
    with pytest.raises(SQLValidationError, match=match) as exc:
        validate_sql(sql, ROLE_SCHEMA)
    assert exc.value.subcode == subcode


def test_valid_simple_select_returns_normalized_sql():
    sql = "SELECT id, name FROM products"
    out = validate_sql(sql, ROLE_SCHEMA)

    assert isinstance(out, str)
    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()


def test_valid_uppercase_role_schema_with_lowercase_sql():
    uppercase_schema = {
        "PRODUCTS": {
            "ID": "INT",
            "NAME": "TEXT",
            "PRICE": "DECIMAL",
            "CATEGORY_ID": "INT",
            "BRAND_ID": "INT",
        },
    }
    out = validate_sql("SELECT id, name FROM products", uppercase_schema)
    out_upper_sql = validate_sql("SELECT ID, NAME FROM PRODUCTS", uppercase_schema)

    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()
    assert "SELECT" in out_upper_sql.upper()
    assert "FROM" in out_upper_sql.upper()


@pytest.mark.parametrize(
    "dialect, table_key, sql",
    [
        ("postgres", "public.products", "SELECT public.products.id FROM public.products"),
        ("sqlserver", "dbo.employee", "SELECT dbo.employee.id FROM dbo.employee"),
        ("mysql", "mydatabase.employee", "SELECT mydatabase.employee.id FROM mydatabase.employee"),
    ],
)
def test_valid_schema_qualified_table_identifiers(dialect: str, table_key: str, sql: str):
    qualified_schema = {
        table_key: {
            "id": "INT",
        }
    }

    out = validate_sql(sql, qualified_schema, dialect=dialect)

    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()


def test_valid_unqualified_query_with_single_qualified_role_table():
    out = validate_sql(
        "SELECT id FROM employee",
        QUALIFIED_ROLE_SCHEMA,
        dialect="sqlserver",
    )
    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()


def test_reject_role_schema_normalization_conflict():
    conflicting_schema = {
        "products": {"id": "INT"},
        "PRODUCTS": {"ID": "INT"},
    }

    with pytest.raises(SQLValidationError, match="Schema normalization conflict on table"):
        validate_sql("SELECT id FROM products", conflicting_schema)


def test_reject_role_schema_column_normalization_conflict():
    conflicting_schema = {
        "products": {"id": "INT", "ID": "INT"},
    }

    with pytest.raises(SQLValidationError, match="Schema normalization conflict on column"):
        validate_sql("SELECT id FROM products", conflicting_schema)


def test_valid_count_star():
    sql = "SELECT COUNT(*) FROM products"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "COUNT(*)" in out.upper()


def test_valid_allowed_function():
    sql = "SELECT SUM(price) FROM products"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "SUM" in out.upper()


def test_valid_case_expression():
    sql = "SELECT CASE WHEN price > 100 THEN 'high' ELSE 'low' END AS price_bucket FROM products"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "CASE" in out.upper()


def test_valid_case_expression_with_or_condition():
    sql = (
        "SELECT CASE "
        "WHEN price IS NULL OR price = 0 THEN NULL "
        "ELSE ROUND(price / NULLIF(price, 0), 2) "
        "END AS ratio "
        "FROM products"
    )
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "CASE" in out.upper()
    assert "NULLIF" in out.upper()


def test_valid_lag_window_function():
    sql = "SELECT LAG(total_amount, 1, 0) OVER (ORDER BY order_date) AS prev_total FROM orders"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "LAG" in out.upper()


@pytest.mark.parametrize("dialect", ["postgres", "mysql", "sqlserver"])
def test_valid_nullif_function(dialect: str):
    sql = "SELECT ROUND(products.price / NULLIF(products.price, 0), 2) FROM products"
    out = validate_sql(sql, ROLE_SCHEMA, dialect=dialect)
    assert "NULLIF" in out.upper()


@pytest.mark.parametrize(
    "dialect, sql",
    [
        (
            "postgres",
            (
                "SELECT LOWER(products.name), ABS(products.price), "
                "ROW_NUMBER() OVER (ORDER BY orders.order_date), "
                "RANK() OVER (ORDER BY orders.order_date), "
                "DENSE_RANK() OVER (ORDER BY orders.order_date), "
                "DATE_PART('month', orders.order_date), CAST(products.price AS DECIMAL) "
                "FROM products JOIN orders ON products.id = orders.id"
            ),
        ),
        (
            "mysql",
            (
                "SELECT LOWER(products.name), ABS(products.price), "
                "ROW_NUMBER() OVER (ORDER BY orders.order_date), "
                "RANK() OVER (ORDER BY orders.order_date), "
                "DENSE_RANK() OVER (ORDER BY orders.order_date), "
                "YEAR(orders.order_date), CAST(products.price AS DECIMAL) "
                "FROM products JOIN orders ON products.id = orders.id"
            ),
        ),
        (
            "sqlserver",
            (
                "SELECT LOWER(products.name), ABS(products.price), "
                "ROW_NUMBER() OVER (ORDER BY orders.order_date), "
                "RANK() OVER (ORDER BY orders.order_date), "
                "DENSE_RANK() OVER (ORDER BY orders.order_date), "
                "DATEPART(month, orders.order_date), CAST(products.price AS DECIMAL) "
                "FROM products JOIN orders ON products.id = orders.id"
            ),
        ),
    ],
)
def test_valid_new_allowed_functions_by_dialect(dialect: str, sql: str):
    out = validate_sql(sql, ROLE_SCHEMA, dialect=dialect)
    assert "SELECT" in out.upper()


def test_valid_select_with_aliases():
    sql = "SELECT p.id, p.name FROM products AS p"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()


def test_valid_select_with_derived_table_alias():
    sql = "SELECT p2.id FROM (SELECT p.id FROM products AS p) AS p2"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "SELECT" in out.upper()
    assert "FROM" in out.upper()


def test_valid_order_by_select_alias():
    sql = "SELECT SUM(price) AS total_price FROM products ORDER BY total_price DESC"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "ORDER BY" in out.upper()


@pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
def test_reject_empty_sql(sql):
    assert_invalid(sql, "SQL is empty")


def test_reject_markdown_or_backticks():
    assert_invalid("```SELECT id FROM products```", "Markdown or code formatting")
    assert_invalid("SELECT `id` FROM products", "Markdown or code formatting")


def test_reject_comments():
    assert_invalid("SELECT id FROM products -- comment", "SQL comments are not allowed")
    assert_invalid("SELECT id FROM products /* comment */", "SQL comments are not allowed")


def test_reject_parse_error():
    assert_invalid("SELEC FROM", "SQL parse error")


def test_reject_multiple_statements():
    assert_invalid(
        "SELECT id FROM products; SELECT id FROM products",
        "Multiple SQL statements",
    )


def test_reject_non_select_statements():
    assert_invalid("DELETE FROM products", "Only SELECT queries are allowed")
    assert_invalid("UPDATE products SET price = 1", "Only SELECT queries are allowed")
    assert_invalid("INSERT INTO products (id) VALUES (1)", "Only SELECT queries are allowed")


def test_reject_select_into():
    assert_invalid("SELECT id INTO new_table FROM products", "SELECT INTO is not allowed")


def test_reject_wildcard_select():
    assert_invalid("SELECT * FROM products", r"Wildcard '\*' is not allowed")


def test_reject_select_without_from():
    assert_invalid("SELECT 1", "SELECT without FROM is not allowed")


def test_valid_non_recursive_cte():
    sql = "WITH t AS (SELECT id FROM products) SELECT t.id FROM t"
    out = validate_sql(sql, ROLE_SCHEMA)
    assert "WITH" in out.upper()


def test_reject_recursive_cte():
    assert_invalid(
        "WITH RECURSIVE t AS (SELECT id FROM products) SELECT id FROM t",
        r"Recursive CTEs \(WITH RECURSIVE\) are not supported",
    )


def test_reject_union():
    assert_invalid(
        "SELECT id FROM products UNION SELECT id FROM products",
        "UNION and UNION ALL are not supported",
    )


def test_reject_disallowed_function():
    assert_invalid("SELECT sqrt(price) FROM products", "Function not allowed")


def test_reject_schema_qualified_function():
    assert_invalid(
        "SELECT pg_catalog.now() FROM products",
        "Schema-qualified functions are not allowed",
    )


def test_reject_unknown_table_as_schema_resolution_error():
    assert_invalid("SELECT id FROM customers", "Schema resolution error")


def test_reject_unknown_table_inside_subquery_as_schema_resolution_error():
    assert_invalid(
        "SELECT x.id FROM (SELECT id FROM customers) AS x",
        "Schema resolution error",
    )


def test_reject_unknown_table_inside_cte_as_schema_resolution_error():
    assert_invalid(
        "WITH t AS (SELECT id FROM customers) SELECT id FROM t",
        "Schema resolution error",
    )


def test_reject_unknown_column_as_schema_resolution_error():
    assert_invalid("SELECT products.sku FROM products", "Schema resolution error")


def test_reject_schema_qualified_table_not_in_role_schema():
    with pytest.raises(SQLValidationError, match="Table not allowed for this role") as exc:
        validate_sql(
            "SELECT COUNT(*) FROM hr.employee",
            QUALIFIED_ROLE_SCHEMA,
            dialect="sqlserver",
        )

    assert exc.value.subcode == "TABLE_NOT_ALLOWED"


def test_reject_ambiguous_unqualified_reference_for_multi_schema_same_table():
    ambiguous_schema = {
        "dbo.employee": {"id": "INT"},
        "hr.employee": {"id": "INT"},
    }

    with pytest.raises(SQLValidationError, match="Schema resolution error") as exc:
        validate_sql("SELECT id FROM employee", ambiguous_schema, dialect="sqlserver")

    assert exc.value.subcode == "SCHEMA_RESOLUTION_ERROR"


def test_reject_system_schema_table():
    assert_invalid(
        "SELECT COUNT(*) FROM pg_catalog.pg_class",
        "Access to system schema is not allowed",
    )


def test_reject_parse_error_subcode():
    assert_invalid_with_subcode("SELEC FROM", "SQL parse error", "PARSE_ERROR")


def test_reject_wildcard_select_subcode():
    assert_invalid_with_subcode("SELECT * FROM products", r"Wildcard '\*' is not allowed", "WILDCARD_SELECT")


def test_reject_system_schema_subcode():
    assert_invalid_with_subcode(
        "SELECT COUNT(*) FROM pg_catalog.pg_class",
        "Access to system schema is not allowed",
        "SYSTEM_SCHEMA_ACCESS",
    )


def test_helper_reject_markdown_or_comments_direct():
    with pytest.raises(SQLValidationError, match="Markdown or code formatting"):
        _reject_markdown_or_comments("SELECT `id` FROM products")
    with pytest.raises(SQLValidationError, match="SQL comments are not allowed"):
        _reject_markdown_or_comments("SELECT id FROM products -- comment")


def test_helper_require_select_only_direct():
    with pytest.raises(SQLValidationError, match="Only SELECT queries are allowed"):
        _require_select_only(parse_one("DELETE FROM products", read="postgres"))


def test_helper_reject_select_into_direct():
    with pytest.raises(SQLValidationError, match="SELECT INTO is not allowed"):
        _reject_select_into(parse_one("SELECT id INTO t FROM products", read="postgres"))


def test_helper_reject_wildcards_direct():
    with pytest.raises(SQLValidationError, match=r"Wildcard '\*' is not allowed"):
        _reject_wildcards(parse_one("SELECT * FROM products", read="postgres"))


def test_helper_allows_count_star_direct():
    _reject_wildcards(parse_one("SELECT COUNT(*) FROM products", read="postgres"))


def test_helper_select_has_no_from():
    no_from_stmt = parse_one("SELECT 1", read="postgres")
    with_from_stmt = parse_one("SELECT id FROM products", read="postgres")

    assert _select_has_no_from(no_from_stmt) is True
    assert _select_has_no_from(with_from_stmt) is False
    assert _select_has_no_from(parse_one("DELETE FROM products", read="postgres")) is False


def test_helper_validate_functions_fallback_dialect():
    stmt = parse_one("SELECT SUM(price) FROM products", read="postgres")
    _validate_functions(stmt, "unknown_dialect")

    stmt_bad = parse_one("SELECT ROUND(price) FROM products", read="postgres")
    with pytest.raises(SQLValidationError, match="Function not allowed"):
        _validate_functions(stmt_bad, "unknown_dialect")


def test_helper_validate_functions_schema_qualified_via_property():
    stmt = parse_one("SELECT pg_catalog.now() FROM products", read="postgres")
    with pytest.raises(SQLValidationError, match="Schema-qualified functions are not allowed"):
        _validate_functions(stmt, "postgres")


def test_helper_validate_schema_access_table_name_missing():
    statement = parse_one("SELECT id FROM products", read="postgres")
    statement.set("from", exp.From(this=exp.Table(this=None)))

    with pytest.raises(SQLValidationError, match="Table name is missing"):
        _validate_schema_access(statement, ROLE_SCHEMA)


def test_helper_validate_schema_access_unauthorized_table_access_branch():
    statement = parse_one("SELECT products.id FROM products", read="postgres")
    statement.set("expressions", [exp.Column(this=exp.Identifier(this="id"), table="ghost_table")])

    with pytest.raises(SQLValidationError, match="Unauthorized table access: 'ghost_table'"):
        _validate_schema_access(statement, ROLE_SCHEMA)


def test_helper_validate_schema_access_unauthorized_column_branch():
    statement = parse_one("SELECT products.id FROM products", read="postgres")
    statement.set("expressions", [exp.Column(this=exp.Identifier(this="sku"), table="products")])

    with pytest.raises(SQLValidationError, match="Unauthorized access to products.sku"):
        _validate_schema_access(statement, ROLE_SCHEMA)


def test_helper_validate_schema_access_unresolved_column_branch():
    statement = parse_one("SELECT products.id FROM products", read="postgres")
    statement.set("expressions", [exp.Column(this=exp.Identifier(this="id"))])

    with pytest.raises(SQLValidationError, match="Unresolved column detected: 'id'"):
        _validate_schema_access(statement, ROLE_SCHEMA)
