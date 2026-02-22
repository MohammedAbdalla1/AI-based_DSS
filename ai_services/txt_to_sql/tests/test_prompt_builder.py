import pytest
from txt_to_sql.prompt_builder import build_sql_prompt


ROLE_SCHEMA = {
    "users": {
        "id": "INT",
        "name": "TEXT"
    },
    "orders": {
        "id": "INT",
        "user_id": "INT",
        "price": "DECIMAL"
    }
}


def test_prompt_contains_question():
    prompt = build_sql_prompt(
        question="List all users",
        schema=ROLE_SCHEMA,
        db_type="postgres"
    )

    assert "List all users" in prompt


def test_prompt_contains_schema_tables():
    prompt = build_sql_prompt(
        question="List all users",
        schema=ROLE_SCHEMA,
        db_type="postgres"
    )

    assert "users" in prompt
    assert "orders" in prompt
    assert "id (INT)" in prompt
    assert "price (DECIMAL)" in prompt


def test_prompt_contains_db_specific_functions():
    prompt = build_sql_prompt(
        question="Count users",
        schema=ROLE_SCHEMA,
        db_type="postgres"
    )

    assert "COUNT" in prompt
    assert "SUM" in prompt


def test_prompt_returns_string():
    prompt = build_sql_prompt(
        question="List users",
        schema=ROLE_SCHEMA,
        db_type="postgres"
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0