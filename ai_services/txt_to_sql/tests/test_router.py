from fastapi.testclient import TestClient

import main
from txt_to_sql import router
from txt_to_sql.models import TextToSQLError, TextToSQLSuccess


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


client = TestClient(main.app)


def test_txt_to_sql_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        router,
        "run_txt_to_sql",
        lambda request: TextToSQLSuccess(
            status="success",
            sql="SELECT products.id, products.name FROM products",
        ),
    )

    response = client.post("/txt-to-sql", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "SELECT" in body["sql"].upper()


def test_txt_to_sql_endpoint_maps_invalid_request_to_400(monkeypatch):
    monkeypatch.setattr(
        router,
        "run_txt_to_sql",
        lambda request: TextToSQLError(
            status="error",
            error_code="INVALID_REQUEST",
            error_subcode="PAYLOAD_VALIDATION",
            message="Question must contain at least 3 words.",
        ),
    )

    response = client.post("/txt-to-sql", json=VALID_PAYLOAD)

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "INVALID_REQUEST"


def test_txt_to_sql_endpoint_maps_sql_rejected_to_422(monkeypatch):
    monkeypatch.setattr(
        router,
        "run_txt_to_sql",
        lambda request: TextToSQLError(
            status="error",
            error_code="SQL_REJECTED",
            error_subcode="WILDCARD_SELECT",
            message="Wildcard '*' is not allowed; select explicit columns",
        ),
    )

    response = client.post("/txt-to-sql", json=VALID_PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "SQL_REJECTED"


def test_txt_to_sql_endpoint_maps_generation_failed_to_502(monkeypatch):
    monkeypatch.setattr(
        router,
        "run_txt_to_sql",
        lambda request: TextToSQLError(
            status="error",
            error_code="GENERATION_FAILED",
            error_subcode="PROVIDER_FAILURE",
            message="Gemini generation failed: ServerError",
        ),
    )

    response = client.post("/txt-to-sql", json=VALID_PAYLOAD)

    assert response.status_code == 502
    body = response.json()
    assert body["error_code"] == "GENERATION_FAILED"


def test_txt_to_sql_endpoint_maps_internal_error_to_500(monkeypatch):
    monkeypatch.setattr(
        router,
        "run_txt_to_sql",
        lambda request: TextToSQLError(
            status="error",
            error_code="INTERNAL_ERROR",
            error_subcode="UNEXPECTED_EXCEPTION",
            message="Unexpected error: RuntimeError",
        ),
    )

    response = client.post("/txt-to-sql", json=VALID_PAYLOAD)

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_ERROR"


def test_txt_to_sql_endpoint_fastapi_validation_uses_uniform_error_shape():
    invalid_payload = {
        "question": "string",
        "db_type": "postgres",
        "role_schema": {"products": {"id": "INT"}},
    }

    response = client.post("/txt-to-sql", json=invalid_payload)

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "INVALID_REQUEST"
    assert body["error_subcode"] == "PAYLOAD_VALIDATION"
    assert "at least 3 words" in body["message"]


def test_txt_to_sql_endpoint_missing_field_uses_uniform_error_shape():
    invalid_payload = {
        "question": "List product id and name",
        "role_schema": {"products": {"id": "INT"}},
    }

    response = client.post("/txt-to-sql", json=invalid_payload)

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "INVALID_REQUEST"
    assert body["error_subcode"] == "PAYLOAD_VALIDATION"
    assert "db_type" in body["message"]


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
