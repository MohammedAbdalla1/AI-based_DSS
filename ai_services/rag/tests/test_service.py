import asyncio
from io import BytesIO
from types import SimpleNamespace

from rag import service as svc
from rag import rate_limit
from rag.vector_store import RetrievedChunk, TenantRetrievedChunk


def _fake_upload(filename: str = "policy.pdf", content_type: str = "application/pdf"):
    return SimpleNamespace(
        filename=filename,
        content_type=content_type,
        file=BytesIO(b"%PDF-1.4 fake pdf bytes"),
        read=lambda: b"%PDF-1.4 fake pdf bytes",
    )


def test_run_rag_query_success(monkeypatch):
    rate_limit.reset_rate_limits()
    monkeypatch.setattr(
        svc.retriever,
        "retrieve_tenant_chunks",
        lambda tenant_id, query, top_k, file_ids=None: [
            TenantRetrievedChunk(
                file_id="file_1",
                file_name="policy.pdf",
                chunk_id="file_1_chunk_0",
                chunk_index=0,
                text="Employees may work remotely two days per week.",
                text_preview="Employees may work remotely two days per week.",
                allowed_roles=["hr", "admin"],
                score=0.91,
            )
        ],
    )
    monkeypatch.setattr(
        svc.generator,
        "generate_answer_with_usage",
        lambda question, contexts: ("Employees may work remotely two days per week.", 120),
    )

    result = svc.run_rag_query({"tenant_id": "tenant_1", "user_role": "hr", "query": "What is the remote work policy"})

    assert "remote" in result.answer.lower()
    assert result.tokens_used == 120
    assert result.sources[0].file_id == "file_1"


def test_run_rag_query_returns_404_when_no_results(monkeypatch):
    rate_limit.reset_rate_limits()
    monkeypatch.setattr(
        svc.retriever,
        "retrieve_tenant_chunks",
        lambda tenant_id, query, top_k, file_ids=None: [],
    )

    result = svc.run_rag_query({"tenant_id": "tenant_1", "user_role": "hr", "query": "What is the remote work policy"})

    assert result.status == "error"
    assert result.error_code == "NOT_FOUND"


def test_run_rag_query_returns_403_when_role_cannot_access_results(monkeypatch):
    rate_limit.reset_rate_limits()
    monkeypatch.setattr(
        svc.retriever,
        "retrieve_tenant_chunks",
        lambda tenant_id, query, top_k, file_ids=None: [
            TenantRetrievedChunk(
                file_id="file_1",
                file_name="policy.pdf",
                chunk_id="file_1_chunk_0",
                chunk_index=0,
                text="Employees may work remotely two days per week.",
                text_preview="Employees may work remotely two days per week.",
                allowed_roles=["finance"],
                score=0.91,
            )
        ],
    )

    result = svc.run_rag_query({"tenant_id": "tenant_1", "user_role": "hr", "query": "What is the remote work policy"})

    assert result.status == "error"
    assert result.error_code == "ACCESS_DENIED"


def test_run_rag_query_maps_invalid_request():
    rate_limit.reset_rate_limits()
    result = svc.run_rag_query({"tenant_id": " ", "user_role": " ", "query": " "})

    assert result.status == "error"
    assert result.error_code == "INVALID_REQUEST"
    assert result.error_subcode == "PAYLOAD_VALIDATION"


async def _fake_read_and_validate_upload(uploaded_file):
    return (b"%PDF-1.4 fake pdf bytes", "policy.pdf", ".pdf", "application/pdf")


def test_run_rag_ingest_success(monkeypatch):
    upload = _fake_upload()
    monkeypatch.setattr(svc.ingestion, "read_and_validate_upload", _fake_read_and_validate_upload)
    monkeypatch.setattr(svc.ingestion, "extract_text", lambda contents, ext: "Employees may work remotely two days per week.")
    monkeypatch.setattr(
        svc.ingestion,
        "chunk_text",
        lambda text: [svc.ingestion.TextChunk(chunk_id="chunk_0", chunk_index=0, text=text)],
    )
    monkeypatch.setattr(svc.generator, "embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    calls = {"upsert": 0, "file_record": 0, "storage_path": None}
    monkeypatch.setattr(svc.vector_store, "upsert_tenant_chunks", lambda records: calls.__setitem__("upsert", len(records)))
    monkeypatch.setattr(
        svc.database,
        "insert_file_record",
        lambda **kwargs: (calls.__setitem__("file_record", 1), calls.__setitem__("storage_path", kwargs["storage_path"])),
    )

    result = asyncio.run(
        svc.run_rag_ingest(
            upload,
            {
                "tenant_id": "tenant_1",
                "uploaded_by": "user_1",
                "uploader_role": "admin",
                "allowed_roles": ["admin", "hr"],
                "metadata": {"department": "HR"},
            },
        )
    )

    assert result.status == "indexed"
    assert result.file_name == "policy.pdf"
    assert result.chunks_created == 1
    assert calls["upsert"] == 1
    assert calls["file_record"] == 1
    assert calls["storage_path"].startswith("transient://tenant_1/")


def test_run_rag_ingest_accepts_arbitrary_uploader_role(monkeypatch):
    upload = _fake_upload()
    monkeypatch.setattr(svc.ingestion, "read_and_validate_upload", _fake_read_and_validate_upload)
    monkeypatch.setattr(svc.ingestion, "extract_text", lambda contents, ext: "Employees may work remotely two days per week.")
    monkeypatch.setattr(
        svc.ingestion,
        "chunk_text",
        lambda text: [svc.ingestion.TextChunk(chunk_id="chunk_0", chunk_index=0, text=text)],
    )
    monkeypatch.setattr(svc.generator, "embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(svc.vector_store, "upsert_tenant_chunks", lambda records: None)
    monkeypatch.setattr(svc.database, "insert_file_record", lambda **kwargs: None)

    result = asyncio.run(
        svc.run_rag_ingest(
            upload,
            {
                "tenant_id": "tenant_1",
                "uploaded_by": "user_1",
                "uploader_role": "professor",
                "allowed_roles": ["admin"],
                "metadata": {},
            },
        )
    )

    assert result.status == "indexed"


def test_run_rag_ingest_requires_file():
    result = asyncio.run(
        svc.run_rag_ingest(
            None,
            {
                "tenant_id": "tenant_1",
                "uploaded_by": "user_1",
                "uploader_role": "admin",
                "allowed_roles": ["admin"],
                "metadata": {},
            },
        )
    )

    assert result.status == "error"
    assert result.error_code == "INVALID_REQUEST"
    assert result.error_subcode == "PAYLOAD_VALIDATION"
