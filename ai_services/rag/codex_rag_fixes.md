# Codex Instructions — RAG Fix List

These are targeted edits to existing files. Do not rewrite files from scratch.
Apply each fix exactly as described, in the file and location specified.

---

## Fix 1 — config.py: add two missing Qdrant fields

**File:** `config.py` (in the root ai_services folder, not inside rag/)

**What to do:** Add `qdrant_url` and `qdrant_api_key` fields in two places.

### Part A — add to the Settings dataclass

Find the `@dataclass(frozen=True)` class named `Settings`.
Add these two lines anywhere inside it alongside the other `rag_*` fields:

```python
qdrant_url: str
qdrant_api_key: str
```

### Part B — add to the get_settings() return statement

Find the `return Settings(` call inside `get_settings()`.
Add these two lines inside the return, alongside the other fields:

```python
qdrant_url=os.getenv("QDRANT_URL", ""),
qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
```

### Part C — add to .env file

In the `.env` file at the root of the project, add these two lines.
Leave them empty for now (local file mode). Fill them in later for Qdrant Cloud.

```env
QDRANT_URL=
QDRANT_API_KEY=
```

---

## Fix 2 — vector_store.py: support Qdrant Cloud and Docker, not just local file

**File:** `rag/vector_store.py`

**What to do:** Replace the entire `_get_client_and_models()` function.

Find this function:
```python
def _get_client_and_models():
    global _client, _models
    if _client is not None and _models is not None:
        return _client, _models

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ModuleNotFoundError as exc:
        raise _vector_error(
            "CLIENT_NOT_INSTALLED",
            "qdrant-client is not installed.",
        ) from exc

    settings = get_settings()
    settings.ensure_runtime_dirs()
    _client = QdrantClient(path=str(settings.rag_qdrant_path))
    _models = models
    return _client, _models
```

Replace it entirely with:
```python
def _get_client_and_models():
    global _client, _models
    if _client is not None and _models is not None:
        return _client, _models

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ModuleNotFoundError as exc:
        raise _vector_error(
            "CLIENT_NOT_INSTALLED",
            "qdrant-client is not installed.",
        ) from exc

    settings = get_settings()

    qdrant_url = getattr(settings, "qdrant_url", "").strip()
    qdrant_api_key = getattr(settings, "qdrant_api_key", "").strip()

    if qdrant_url:
        # Qdrant Cloud or self-hosted Docker server
        _client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key or None,
        )
    else:
        # Local embedded file mode (dev / testing)
        settings.ensure_runtime_dirs()
        _client = QdrantClient(path=str(settings.rag_qdrant_path))

    _models = models
    return _client, _models
```

---

## Fix 3 — vector_store.py: guard against missing collection in search

**File:** `rag/vector_store.py`

**What to do:** Add a collection-existence check at the top of `search_tenant_chunks()`.

Find the function `search_tenant_chunks` and locate this line near the top of the function body:

```python
    client, models = _get_client_and_models()
    collection_name = _tenant_collection_name(tenant_id)

    must_conditions = []
```

Insert these lines between `collection_name = _tenant_collection_name(tenant_id)` and `must_conditions = []`:

```python
    # If this tenant has never ingested a file, the collection won't exist yet.
    # Return empty results instead of raising an error.
    try:
        client.get_collection(collection_name)
    except Exception:
        return []
```

After the edit that section should look like:
```python
    client, models = _get_client_and_models()
    collection_name = _tenant_collection_name(tenant_id)

    try:
        client.get_collection(collection_name)
    except Exception:
        return []

    must_conditions = []
```

---

## Fix 4 — service.py: fix the order of SQLite insert and Qdrant upsert

**File:** `rag/service.py`

**What to do:** Inside `run_rag_ingest()`, swap the order so SQLite is written first,
then Qdrant. If Qdrant fails after SQLite succeeds, delete the SQLite record.

**First**, add a new import at the top of `service.py` if not already present.
Find the imports block and make sure this line exists:
```python
from . import access_control, database, generator, ingestion, rate_limit, retriever, vector_store
```
It already exists — no change needed there.

**Second**, find this block inside `run_rag_ingest()` (inside the `try:` block):

```python
        vector_store.upsert_tenant_chunks(vector_records)
        database.insert_file_record(
            file_id=file_id,
            tenant_id=metadata.tenant_id,
            uploaded_by=metadata.uploaded_by,
            file_name=file_name,
            storage_path=storage_path,
            allowed_roles=metadata.allowed_roles,
            metadata=metadata.metadata,
            chunks_count=len(chunks),
        )
        return RAGIngestSuccess(
            file_id=file_id,
            file_name=file_name,
            chunks_created=len(chunks),
            status="indexed",
        )
```

Replace it with:

```python
        # Insert the file record into SQLite first.
        # If the subsequent Qdrant upsert fails, we delete this record
        # so we don't end up with a SQLite entry that has no vectors.
        database.insert_file_record(
            file_id=file_id,
            tenant_id=metadata.tenant_id,
            uploaded_by=metadata.uploaded_by,
            file_name=file_name,
            storage_path=storage_path,
            allowed_roles=metadata.allowed_roles,
            metadata=metadata.metadata,
            chunks_count=len(chunks),
        )
        try:
            vector_store.upsert_tenant_chunks(vector_records)
        except Exception:
            database.delete_file_record(file_id)
            raise
        return RAGIngestSuccess(
            file_id=file_id,
            file_name=file_name,
            chunks_created=len(chunks),
            status="indexed",
        )
```

---

## Fix 5 — database.py: add the delete_file_record function

**File:** `rag/database.py`

**What to do:** Add a new function at the end of the file.

Append this function after the last function in the file (`insert_file_record`):

```python
def delete_file_record(file_id: str) -> None:
    """Remove a file record from SQLite. Used for compensating rollback."""
    ensure_schema()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM files WHERE id = ?",
            (file_id,),
        )
```

---

## Fix 6 — main.py: use the RAG router's own validation error handler

**File:** `main.py` (root folder)

**What to do:** Change which module `request_validation_error_response` is imported from.

Find this import block:
```python
from txt_to_sql.router import (
    request_validation_error_response,
    router as txt_to_sql_router,
)
```

Replace it with:
```python
from rag.router import request_validation_error_response
from txt_to_sql.router import router as txt_to_sql_router
```

The rest of `main.py` stays exactly the same.
Both routers produce the same error shape, so this change is safe.
It removes the coupling where the txt_to_sql module was accidentally owning
a shared concern.

---

## Verification checklist

After applying all fixes, confirm the following before running:

- [ ] `config.py` Settings dataclass has `qdrant_url: str` and `qdrant_api_key: str`
- [ ] `config.py` get_settings() return includes `qdrant_url=` and `qdrant_api_key=`
- [ ] `.env` has `QDRANT_URL=` and `QDRANT_API_KEY=` lines (can be empty for local mode)
- [ ] `vector_store.py` _get_client_and_models() checks `qdrant_url` before choosing mode
- [ ] `vector_store.py` search_tenant_chunks() has the `client.get_collection()` guard
- [ ] `service.py` inserts SQLite record before Qdrant upsert, with rollback on failure
- [ ] `database.py` has the new `delete_file_record()` function
- [ ] `main.py` imports `request_validation_error_response` from `rag.router`

## Install dependencies (run once)

```bash
pip install qdrant-client pypdf python-docx google-genai \
            fastapi uvicorn pydantic pydantic-settings \
            python-multipart python-dotenv
```

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

Test it is running:
```
GET http://localhost:8000/health
Expected: {"status": "ok"}
```
