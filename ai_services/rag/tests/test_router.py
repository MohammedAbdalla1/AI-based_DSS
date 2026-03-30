from fastapi.testclient import TestClient

import main
from rag import router
from rag.models import RAGError, RAGIngestSuccess, RAGQuerySuccess


client = TestClient(main.app)


def test_rag_query_endpoint_success(monkeypatch):
    main.app.dependency_overrides[router.verify_internal_key] = lambda: None
    monkeypatch.setattr(
        router,
        "run_rag_query",
        lambda request: RAGQuerySuccess(
            answer="Employees may work remotely two days per week.",
            sources=[
                {
                    "file_id": "file_1",
                    "file_name": "policy.pdf",
                    "chunk_id": "file_1_chunk_0",
                    "relevance_score": 0.91,
                    "excerpt": "Employees may work remotely two days per week.",
                }
            ],
            tokens_used=123,
        ),
    )

    response = client.post(
        "/api/rag/query",
        headers={"X-Internal-API-Key": "test-key"},
        json={"tenant_id": "tenant_1", "user_role": "hr", "query": "What is the remote work policy"},
    )
    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "remote" in response.json()["answer"].lower()


def test_rag_query_endpoint_maps_invalid_request_to_400(monkeypatch):
    main.app.dependency_overrides[router.verify_internal_key] = lambda: None
    monkeypatch.setattr(
        router,
        "run_rag_query",
        lambda request: RAGError(
            status="error",
            error_code="INVALID_REQUEST",
            error_subcode="PAYLOAD_VALIDATION",
            message="query is required",
        ),
    )

    response = client.post(
        "/api/rag/query",
        headers={"X-Internal-API-Key": "test-key"},
        json={"tenant_id": "tenant_1", "user_role": "hr", "query": "What is the remote work policy"},
    )
    main.app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_REQUEST"


def test_rag_query_endpoint_fastapi_validation_uses_uniform_error_shape():
    main.app.dependency_overrides[router.verify_internal_key] = lambda: None
    response = client.post(
        "/api/rag/query",
        headers={"X-Internal-API-Key": "test-key"},
        json={"tenant_id": "tenant_1", "user_role": "hr", "query": " "},
    )
    main.app.dependency_overrides.clear()

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error_code"] == "INVALID_REQUEST"
    assert body["error_subcode"] == "PAYLOAD_VALIDATION"


def test_rag_ingest_endpoint_requires_internal_key():
    original_get_settings = router.get_settings

    class _Settings:
        enforce_internal_api_key = True
        internal_api_key = "secret"

    router.get_settings = lambda: _Settings()
    response = client.post(
        "/api/rag/ingest",
        files={"file": ("policy.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        data={
            "tenant_id": "tenant_1",
            "uploaded_by": "user_1",
            "uploader_role": "admin",
            "allowed_roles": '["admin","hr"]',
        },
    )
    router.get_settings = original_get_settings

    assert response.status_code == 401


def test_rag_ingest_endpoint_success(monkeypatch):
    main.app.dependency_overrides[router.verify_internal_key] = lambda: None

    async def fake_run_rag_ingest(file, payload):
        return RAGIngestSuccess(
            file_id="file_1",
            file_name="policy.pdf",
            chunks_created=2,
            status="indexed",
        )

    monkeypatch.setattr(
        router,
        "run_rag_ingest",
        fake_run_rag_ingest,
    )

    response = client.post(
        "/api/rag/ingest",
        headers={"X-Internal-API-Key": "test-key"},
        files={"file": ("policy.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        data={
            "tenant_id": "tenant_1",
            "uploaded_by": "user_1",
            "uploader_role": "admin",
            "allowed_roles": '["admin","hr"]',
            "metadata": '{"department":"HR"}',
        },
    )

    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert response.json()["chunks_created"] == 2


def test_rag_ingest_endpoint_accepts_comma_separated_allowed_roles(monkeypatch):
    main.app.dependency_overrides[router.verify_internal_key] = lambda: None

    async def fake_run_rag_ingest(file, payload):
        return RAGIngestSuccess(
            file_id="file_1",
            file_name="policy.pdf",
            chunks_created=1,
            status="indexed",
        )

    monkeypatch.setattr(router, "run_rag_ingest", fake_run_rag_ingest)

    response = client.post(
        "/api/rag/ingest",
        headers={"X-Internal-API-Key": "test-key"},
        files={"file": ("policy.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        data={
            "tenant_id": "tenant_1",
            "uploaded_by": "user_1",
            "uploader_role": "admin",
            "allowed_roles": "admin,hr",
        },
    )

    main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
