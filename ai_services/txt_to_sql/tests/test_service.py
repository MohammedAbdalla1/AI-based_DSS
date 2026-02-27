from types import SimpleNamespace

from txt_to_sql.models import TextToSQLRequest
from txt_to_sql.service import run_txt_to_sql
from txt_to_sql import service as svc
from txt_to_sql.sql_generator import SQLGenerationError
from txt_to_sql.sql_validator import SQLValidationError


VALID_PAYLOAD = {
    "question": "List product id and name",
    "db_type": "postgres",
    "role_schema": {
        "products": {
            "id": "INT",
            "name": "TEXT",
        }
    },
}


def test_run_txt_to_sql_success_with_payload(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(svc, "generate_sql", lambda prompt: "SELECT id, name FROM products")
    monkeypatch.setattr(svc, "validate_sql", lambda sql, role_schema, dialect: "SELECT products.id, products.name FROM products")

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "success"
    assert "SELECT" in result.sql.upper()


def test_run_txt_to_sql_success_with_model_instance(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(svc, "generate_sql", lambda prompt: "SELECT id, name FROM products")
    monkeypatch.setattr(svc, "validate_sql", lambda sql, role_schema, dialect: "SELECT products.id, products.name FROM products")

    request = TextToSQLRequest(**VALID_PAYLOAD)
    result = run_txt_to_sql(request)

    assert result.status == "success"


def test_run_txt_to_sql_maps_invalid_request():
    result = run_txt_to_sql({"question": "  ", "db_type": "postgres", "role_schema": {"products": {"id": "INT"}}})

    assert result.status == "error"
    assert result.error_code == "INVALID_REQUEST"
    assert result.error_subcode == "PAYLOAD_VALIDATION"


def test_run_txt_to_sql_maps_generation_error(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")

    def _raise_generation(prompt):
        raise SQLGenerationError("provider unavailable")

    monkeypatch.setattr(svc, "generate_sql", _raise_generation)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "error"
    assert result.error_code == "GENERATION_FAILED"
    assert result.error_subcode == "UNKNOWN"


def test_run_txt_to_sql_maps_sql_rejected(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(svc, "generate_sql", lambda prompt: "SELECT * FROM products")

    def _raise_validation(sql, role_schema, dialect):
        raise SQLValidationError("Wildcard '*' is not allowed")

    monkeypatch.setattr(svc, "validate_sql", _raise_validation)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "error"
    assert result.error_code == "SQL_REJECTED"
    assert result.error_subcode == "WILDCARD_SELECT"


def test_run_txt_to_sql_maps_internal_error(monkeypatch):
    def _raise_unexpected(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "build_sql_prompt", _raise_unexpected)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "error"
    assert result.error_code == "INTERNAL_ERROR"
    assert result.error_subcode == "UNEXPECTED_EXCEPTION"


def test_run_txt_to_sql_retries_generation_empty_response(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    attempts = {"count": 0}

    def _generate(prompt):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise SQLGenerationError("Empty response from LLM")
        return "SELECT id, name FROM products"

    monkeypatch.setattr(svc, "generate_sql", _generate)
    monkeypatch.setattr(svc, "validate_sql", lambda sql, role_schema, dialect: "SELECT products.id, products.name FROM products")

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "success"
    assert attempts["count"] == 2


def test_run_txt_to_sql_no_retry_for_non_retryable_generation_error(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    attempts = {"count": 0}

    def _generate(prompt):
        attempts["count"] += 1
        raise SQLGenerationError("Prompt is empty")

    monkeypatch.setattr(svc, "generate_sql", _generate)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "error"
    assert result.error_code == "GENERATION_FAILED"
    assert result.error_subcode == "PROMPT_EMPTY"
    assert attempts["count"] == 1


def test_run_txt_to_sql_cleans_markdown_or_comments_before_rejecting(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(svc, "generate_sql", lambda prompt: "SELECT `id`, `name` FROM products -- comment")
    validated_inputs = []

    def _validate(sql, role_schema, dialect):
        validated_inputs.append(sql)
        if "--" in sql or "`" in sql:
            raise SQLValidationError("SQL comments are not allowed")
        return "SELECT products.id, products.name FROM products"

    monkeypatch.setattr(svc, "validate_sql", _validate)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "success"
    assert len(validated_inputs) == 2
    assert "--" in validated_inputs[0]
    assert "--" not in validated_inputs[1]
    assert "`" not in validated_inputs[1]


def test_run_txt_to_sql_regenerates_after_parse_error(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    generation_attempts = {"count": 0}

    def _generate(prompt):
        generation_attempts["count"] += 1
        if generation_attempts["count"] == 1:
            return "SELEC FROM"
        return "SELECT id, name FROM products"

    def _validate(sql, role_schema, dialect):
        if sql == "SELEC FROM":
            raise SQLValidationError("SQL parse error: invalid syntax")
        return "SELECT products.id, products.name FROM products"

    monkeypatch.setattr(svc, "generate_sql", _generate)
    monkeypatch.setattr(svc, "validate_sql", _validate)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "success"
    assert generation_attempts["count"] == 2


def test_run_txt_to_sql_does_not_retry_hard_sql_rejection(monkeypatch):
    monkeypatch.setattr(svc, "build_sql_prompt", lambda **kwargs: "prompt")
    generation_attempts = {"count": 0}
    validation_attempts = {"count": 0}

    def _generate(prompt):
        generation_attempts["count"] += 1
        return "SELECT id FROM customers"

    def _validate(sql, role_schema, dialect):
        validation_attempts["count"] += 1
        raise SQLValidationError("Table not allowed for this role: 'customers'")

    monkeypatch.setattr(svc, "generate_sql", _generate)
    monkeypatch.setattr(svc, "validate_sql", _validate)

    result = run_txt_to_sql(VALID_PAYLOAD)

    assert result.status == "error"
    assert result.error_code == "SQL_REJECTED"
    assert result.error_subcode == "TABLE_NOT_ALLOWED"
    assert generation_attempts["count"] == 1
    assert validation_attempts["count"] == 1
