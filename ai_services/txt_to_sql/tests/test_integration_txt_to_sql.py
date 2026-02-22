import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

from txt_to_sql.models import TextToSQLRequest
from txt_to_sql.prompt_builder import build_sql_prompt
from txt_to_sql.sql_generator import SQLGenerationError, generate_sql
from txt_to_sql.sql_validator import SQLValidationError, validate_sql
from txt_to_sql import sql_generator as sg


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

ROLE_SCHEMA = {
    "products": {
        "id": "INT",
        "name": "TEXT",
        "price": "DECIMAL",
    }
}


@pytest.fixture(autouse=True)
def reset_client_cache():
    sg._client = None
    yield
    sg._client = None


def _run_pipeline(payload: dict) -> str:
    req = TextToSQLRequest(**payload)
    prompt = build_sql_prompt(req.question, req.role_schema, req.db_type)
    generated_sql = generate_sql(prompt)
    return validate_sql(generated_sql, req.role_schema, req.db_type)


def test_pipeline_success(monkeypatch):
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text="SELECT id, name FROM products")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    payload = {
        "question": "List all products with id and name",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    validated_sql = _run_pipeline(payload)

    assert isinstance(validated_sql, str)
    assert "SELECT" in validated_sql.upper()
    assert "FROM" in validated_sql.upper()
    assert "products" in validated_sql.lower()
    assert len(calls) == 1
    assert "Table: products" in calls[0]["contents"]
    assert "List all products with id and name" in calls[0]["contents"]


def test_pipeline_rejects_invalid_generated_sql(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="SELECT * FROM products")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    payload = {
        "question": "Show all products",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    with pytest.raises(SQLValidationError, match=r"Wildcard '\*' is not allowed"):
        _run_pipeline(payload)


def test_pipeline_rejects_invalid_request_before_generation():
    payload = {
        "question": "   ",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    with pytest.raises(ValidationError, match="Question cannot be empty or whitespace"):
        _run_pipeline(payload)


def test_pipeline_rejects_too_short_question_before_generation():
    payload = {
        "question": "list products",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    with pytest.raises(ValidationError, match="Question must contain at least 3 words"):
        _run_pipeline(payload)


def test_pipeline_surfaces_generation_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    payload = {
        "question": "List product ids",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    with pytest.raises(SQLGenerationError, match="GEMINI_API_KEY is missing"):
        _run_pipeline(payload)


def test_pipeline_normalizes_markdown_wrapped_sql_from_llm(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="```sql\nSELECT id, name FROM products\n```")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    payload = {
        "question": "List all products",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    validated_sql = _run_pipeline(payload)

    assert "SELECT" in validated_sql.upper()
    assert "FROM" in validated_sql.upper()


def test_pipeline_enforces_dialect_specific_functions(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="SELECT CURDATE() FROM products")

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(sg.genai, "Client", FakeClient)

    mysql_payload = {
        "question": "Get current date",
        "db_type": "mysql",
        "role_schema": ROLE_SCHEMA,
    }
    postgres_payload = {
        "question": "Get current date",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    mysql_sql = _run_pipeline(mysql_payload)
    assert "CURRENT_DATE" in mysql_sql.upper()

    with pytest.raises(SQLValidationError, match="Function not allowed"):
        _run_pipeline(postgres_payload)


@pytest.mark.live
def test_pipeline_live_with_real_gemini_key():
    load_dotenv(dotenv_path=_ENV_PATH)

    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip(f"GEMINI_API_KEY not found in {_ENV_PATH}")

    payload = {
        "question": "Return product id and name from products only.",
        "db_type": "postgres",
        "role_schema": ROLE_SCHEMA,
    }

    try:
        validated_sql = _run_pipeline(payload)
    except SQLGenerationError as exc:
        # Live tests should not fail the suite on external quota/connectivity issues.
        cause = exc.__cause__
        cause_name = type(cause).__name__ if cause else ""
        cause_text = str(cause) if cause else ""

        if cause_name in {"ClientError", "ServerError"} or "RESOURCE_EXHAUSTED" in cause_text:
            pytest.skip(f"Live Gemini unavailable (quota/service): {cause_name or exc}")
        raise

    assert isinstance(validated_sql, str)
    assert "SELECT" in validated_sql.upper()
    assert "FROM" in validated_sql.upper()
