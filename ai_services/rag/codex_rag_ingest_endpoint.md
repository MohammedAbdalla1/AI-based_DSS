# Codex Implementation Instructions — RAG Ingestion Endpoint

## Context

You are building a **SaaS RAG (Retrieval-Augmented Generation) API** where multiple
businesses (tenants) upload files (PDFs, DOCX, TXT) and later query them.

**IMPORTANT — caller identity:** This API is NOT called directly by end users or a
browser. It is called exclusively by the project's **.NET backend**, which is the only
client. The .NET backend has already authenticated the user with its own JWT, resolved
the tenant, and checked permissions. It forwards the relevant context (tenant_id, role,
user_id) in the request body. The RAG API must NOT re-implement user authentication —
it only needs to verify that the caller is the trusted .NET backend.

---

## Tech Stack Assumptions

- **Runtime**: Python (FastAPI) or Node.js (Express)
- **Auth**: Shared internal API key in a custom header `X-Internal-API-Key`
- **File parsing**: `pypdf` / `python-docx` / plain read (Python) or `pdf-parse` / `mammoth` (Node)
- **Chunking**: sliding window or `RecursiveCharacterTextSplitter`
- **Embedding model**: OpenAI `text-embedding-3-small` or any sentence-transformer
- **Vector DB**: Pinecone / Weaviate / Qdrant / pgvector
- **Relational DB**: PostgreSQL
- **Object storage**: S3 / MinIO / local disk

---

## Endpoint Specification

```
POST /api/rag/ingest
Content-Type: multipart/form-data
X-Internal-API-Key: <shared-secret>
```

### Request fields (multipart/form-data)

| Field           | Type        | Required | Description                                                      |
|----------------|-------------|----------|------------------------------------------------------------------|
| file            | binary      | yes      | The file to upload (PDF, DOCX, or TXT)                           |
| tenant_id       | string      | yes      | UUID of the business — forwarded by .NET backend from its own auth |
| uploaded_by     | string      | yes      | UUID of the user who triggered the upload                         |
| uploader_role   | string      | yes      | Role of the uploading user e.g. "admin" — used for permission check |
| allowed_roles   | JSON string | yes      | e.g. '["admin","hr"]' — roles that can query this file           |
| metadata        | JSON string | no       | Optional tags e.g. '{"department":"HR","year":2024}'             |

### Example call from .NET backend

```http
POST /api/rag/ingest
X-Internal-API-Key: super-secret-internal-key-here
Content-Type: multipart/form-data

file:           <binary PDF>
tenant_id:      "a1b2c3d4-..."
uploaded_by:    "user-uuid-..."
uploader_role:  "admin"
allowed_roles:  '["admin","hr"]'
metadata:       '{"department":"HR"}'
```

### Success response (200)

```json
{
  "file_id": "uuid-v4",
  "file_name": "HR_Policy_2024.pdf",
  "chunks_created": 42,
  "status": "indexed"
}
```

### Error responses

| Code | Reason                                                                 |
|------|------------------------------------------------------------------------|
| 400  | Missing required fields, unsupported file type, or malformed JSON      |
| 401  | X-Internal-API-Key header is missing or does not match the secret      |
| 403  | uploader_role is not in the allowed set (admin, uploader)              |
| 413  | File exceeds size limit (20MB)                                         |
| 500  | Parsing, embedding, or DB error                                        |

---

## Implementation Steps — Do These in Order

### Step 1 — Internal API key check (replaces JWT auth)

The RAG API trusts the .NET backend entirely. The only security check needed is
confirming the request comes from the .NET backend and not an outside caller.

Store the shared secret in an environment variable: INTERNAL_API_KEY.

```python
import os
from fastapi import Request, HTTPException

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]

def verify_internal_key(request: Request):
    key = request.headers.get("X-Internal-API-Key", "")
    if key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
```

Use this as a dependency on every route in this API. That is the entire auth layer.
Do NOT implement JWT decoding, user lookups, or session management here — the .NET
backend handles all of that.

---

### Step 2 — Role check (RBAC gate for upload)

Only users with the role admin or uploader are allowed to ingest files.
The role comes from the uploader_role field in the request body.

```python
ALLOWED_UPLOAD_ROLES = {"admin", "uploader"}

if uploader_role not in ALLOWED_UPLOAD_ROLES:
    raise HTTPException(status_code=403, detail="Insufficient permissions to upload")
```

---

### Step 3 — Receive and validate the file

1. Check the file extension. Supported: .pdf, .docx, .txt
2. Enforce a 20MB size limit. Reject with 413 if exceeded.
3. Generate a file_id as UUID v4.
4. Save the raw file to object storage using key: {tenant_id}/{file_id}/{original_filename}

```python
import uuid, os

SUPPORTED_TYPES = {".pdf", ".docx", ".txt"}
MAX_SIZE = 20 * 1024 * 1024

file_id = str(uuid.uuid4())
ext = os.path.splitext(file.filename)[1].lower()

if ext not in SUPPORTED_TYPES:
    raise HTTPException(400, f"Unsupported file type: {ext}")

contents = await file.read()
if len(contents) > MAX_SIZE:
    raise HTTPException(413, "File exceeds 20MB limit")

storage_path = f"{tenant_id}/{file_id}/{file.filename}"
save_to_storage(storage_path, contents)
```

---

### Step 4 — Parse the file to raw text

```python
def extract_text(file_bytes: bytes, ext: str) -> str:
    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        import io
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
```

If extraction returns empty string, return 400: "File appears empty or unreadable".

---

### Step 5 — Chunk the text

Chunk size: 2000 characters. Overlap: 200 characters.

```python
def chunk_text(text: str, chunk_size=2000, overlap=200) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]
```

---

### Step 6 — Embed each chunk

Use the same embedding model as the query endpoint. Changing it later requires
re-indexing all existing chunks.

```python
import openai

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",   # must match query endpoint
        input=chunks
    )
    return [item.embedding for item in response.data]
```

---

### Step 7 — Build chunk records with metadata

```python
records = []
for i, (text, vector) in enumerate(zip(chunks, vectors)):
    records.append({
        "id": f"{file_id}_chunk_{i}",
        "vector": vector,
        "metadata": {
            "tenant_id":     tenant_id,
            "file_id":       file_id,
            "file_name":     file.filename,
            "chunk_index":   i,
            "allowed_roles": allowed_roles,   # list of strings e.g. ["admin","hr"]
            "text_preview":  text[:300],
        }
    })
```

---

### Step 8 — Upsert to the vector DB (tenant-isolated collection)

The collection name is always derived from tenant_id. Never mix tenants in one collection.

```python
collection_name = f"tenant_{tenant_id}_chunks"

vector_db.create_collection_if_not_exists(
    name=collection_name,
    dimension=1536    # for text-embedding-3-small; adjust for your model
)

vector_db.upsert(collection=collection_name, records=records)
```

---

### Step 9 — Save file metadata to relational DB

```sql
CREATE TABLE files (
    id            UUID PRIMARY KEY,
    tenant_id     UUID NOT NULL REFERENCES tenants(id),
    uploaded_by   UUID NOT NULL,
    file_name     TEXT NOT NULL,
    storage_path  TEXT NOT NULL,
    allowed_roles TEXT[] NOT NULL,
    chunks_count  INT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);
```

```python
db.execute("""
    INSERT INTO files (id, tenant_id, uploaded_by, file_name, storage_path, allowed_roles, chunks_count)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (file_id, tenant_id, uploaded_by, file.filename, storage_path, allowed_roles, len(chunks)))
```

---

### Step 10 — Return success response

```python
return {
    "file_id":        file_id,
    "file_name":      file.filename,
    "chunks_created": len(chunks),
    "status":         "indexed"
}
```

---

## Full Route Handler (Python / FastAPI)

```python
@router.post("/api/rag/ingest")
async def ingest_file(
    file:          UploadFile = File(...),
    tenant_id:     str = Form(...),
    uploaded_by:   str = Form(...),
    uploader_role: str = Form(...),
    allowed_roles: str = Form(...),
    metadata:      str = Form(None),
    _: None = Depends(verify_internal_key)
):
    if uploader_role not in {"admin", "uploader"}:
        raise HTTPException(403, "Insufficient permissions")

    try:
        roles = json.loads(allowed_roles)
        assert isinstance(roles, list)
    except Exception:
        raise HTTPException(400, "allowed_roles must be a JSON array of strings")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 20MB limit")

    file_id = str(uuid.uuid4())
    storage_path = f"{tenant_id}/{file_id}/{file.filename}"
    save_to_storage(storage_path, contents)

    raw_text = extract_text(contents, ext)
    if not raw_text.strip():
        raise HTTPException(400, "File appears empty or unreadable")

    chunks  = chunk_text(raw_text)
    vectors = embed_chunks(chunks)
    records = build_chunk_records(file_id, file.filename, tenant_id, roles, chunks, vectors)

    upsert_to_vector_db(tenant_id, records)
    save_file_metadata(file_id, tenant_id, uploaded_by, file.filename, storage_path, roles, len(chunks))

    return {"file_id": file_id, "file_name": file.filename, "chunks_created": len(chunks), "status": "indexed"}
```

---

## Important Rules — Do Not Skip These

1. The X-Internal-API-Key secret must live in an environment variable. Never
   hard-code it. Rotate it if it is ever leaked.

2. This API must NOT be exposed to the public internet. Place it behind a private
   network boundary (VPC, internal DNS, or Docker network). The API key is a second
   line of defence, not the first.

3. Never accept tenant_id from any source other than the .NET backend's request body.
   The .NET backend is responsible for ensuring tenant_id corresponds to the
   authenticated user's organisation. The RAG API uses whatever value it receives.

4. allowed_roles must be stored on every chunk, not just the file record.
   The query endpoint reads it from chunk metadata — if it is missing there, RBAC
   breaks silently.

5. Use the exact same embedding model here as in the query endpoint. They must share
   a vector space. Changing the model requires re-indexing all chunks.

6. Wrap vector DB upsert and relational DB insert in compensating logic. If one
   succeeds and the other fails, you will have orphaned data. Either use a transaction
   or queue a cleanup job on failure.

7. Do not expose internal error messages in the HTTP response. Log them server-side
   and return a generic 500 message.
