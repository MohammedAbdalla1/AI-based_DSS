# Codex Implementation Instructions — RAG Query Endpoint

## Context

You are building a **SaaS RAG (Retrieval-Augmented Generation) API** where multiple
businesses (tenants) query files they have uploaded.

**IMPORTANT — caller identity:** This API is NOT called directly by end users or a
browser. It is called exclusively by the project's **.NET backend**, which is the only
client. The .NET backend has already authenticated the user with its own JWT, resolved
the tenant, and verified the user's role. It forwards tenant_id and user_role in the
request body. The RAG API must NOT re-implement user authentication — it only needs to
verify that the caller is the trusted .NET backend, then apply RBAC filtering on chunks.

---

## Tech Stack Assumptions

Same as the ingestion endpoint. Use the same embedding model, vector DB, and API key
middleware. These two endpoints are two halves of the same system.

---

## Endpoint Specification

```
POST /api/rag/query
Content-Type: application/json
X-Internal-API-Key: <shared-secret>
```

### Request body

```json
{
  "tenant_id": "uuid-of-the-business",
  "user_role": "hr",
  "query": "What is the refund policy for enterprise customers?",
  "top_k": 5,
  "filters": {
    "file_ids": ["uuid-1", "uuid-2"]
  }
}
```

| Field            | Type     | Required | Description                                                          |
|-----------------|----------|----------|----------------------------------------------------------------------|
| tenant_id        | string   | yes      | UUID of the business — forwarded by .NET backend from its own auth   |
| user_role        | string   | yes      | Role of the querying user e.g. "hr" — used for RBAC chunk filtering  |
| query            | string   | yes      | The natural language question                                        |
| top_k            | integer  | no       | Chunks to retrieve before filtering. Default 5, max 20               |
| filters.file_ids | string[] | no       | Scope search to specific files. Omit to search all accessible files  |

### Success response (200)

```json
{
  "answer": "Enterprise customers are eligible for a full refund within 30 days...",
  "sources": [
    {
      "file_id": "uuid-1",
      "file_name": "Refund_Policy_2024.pdf",
      "chunk_id": "uuid-1_chunk_4",
      "relevance_score": 0.91,
      "excerpt": "...refund within 30 days of purchase for enterprise tier..."
    }
  ],
  "tokens_used": 1240
}
```

### Error responses

| Code | Reason                                                                   |
|------|--------------------------------------------------------------------------|
| 400  | Missing or empty query field                                             |
| 401  | X-Internal-API-Key header is missing or does not match the secret        |
| 403  | Valid request but user role cannot access any of the retrieved chunks     |
| 404  | Vector search returned no results at all                                 |
| 429  | Rate limit exceeded                                                      |
| 500  | Embedding, vector search, or LLM error                                   |

---

## Implementation Steps — Do These in Order

### Step 1 — Internal API key check (same middleware as ingestion)

Reuse the exact same verify_internal_key dependency from the ingestion endpoint.
Do not duplicate it — import and apply it here.

```python
# Reuse from shared auth module
from auth import verify_internal_key

@router.post("/api/rag/query")
async def query_rag(body: dict, _: None = Depends(verify_internal_key)):
    ...
```

---

### Step 2 — Validate the request body

1. Check that query is present and not empty/whitespace. Return 400 if missing.
2. Clamp top_k to the range [1, 20]. Default to 5 if not provided.
3. Read tenant_id and user_role from the body — never from a JWT here.

```python
tenant_id  = body.get("tenant_id", "").strip()
user_role  = body.get("user_role", "").strip()
query_text = body.get("query", "").strip()

if not query_text:
    raise HTTPException(400, "query field is required and cannot be empty")
if not tenant_id or not user_role:
    raise HTTPException(400, "tenant_id and user_role are required")

top_k           = min(max(body.get("top_k", 5), 1), 20)
file_ids_filter = body.get("filters", {}).get("file_ids", None)
```

---

### Step 3 — Embed the query

Use the exact same embedding model used during ingestion.

```python
import openai

def embed_query(query: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",   # must match ingestion endpoint
        input=[query]
    )
    return response.data[0].embedding
```

---

### Step 4 — Resolve the tenant's collection

The collection name comes from tenant_id in the request body.
Never let the client specify the collection name directly.

```python
collection_name = f"tenant_{tenant_id}_chunks"
```

---

### Step 5 — Vector similarity search

Search the tenant's collection for the top-K most similar chunks.
If file_ids filter was provided, pass it as a metadata pre-filter.

```python
search_filter = {"file_id": {"$in": file_ids_filter}} if file_ids_filter else None

raw_results = vector_db.search(
    collection=collection_name,
    vector=query_vector,
    top_k=top_k,
    filter=search_filter
)
# raw_results: list of { id, score, metadata: { allowed_roles, file_id, ... } }
```

If raw_results is empty, return 404:
```python
if not raw_results:
    raise HTTPException(404, "No relevant documents found")
```

---

### Step 6 — RBAC post-filter (critical security step)

After retrieval, filter out every chunk whose allowed_roles does not include
the user's role. This is the enforcement point — the LLM must never receive
unauthorized content.

```python
authorized_chunks = [
    r for r in raw_results
    if user_role in r["metadata"]["allowed_roles"]
]
```

If authorized_chunks is empty after filtering, return 403:
```python
if not authorized_chunks:
    raise HTTPException(403, "You do not have permission to access the relevant documents")
```

Important distinction:
- 404 = the vector search itself returned nothing
- 403 = results were found but the user's role cannot access them
These are different situations and must return different codes.

---

### Step 7 — Build the prompt

Assemble the final prompt from a system instruction, the authorized chunk texts
as context, and the user's original question.

```python
def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        preview = chunk["metadata"]["text_preview"]
        fname   = chunk["metadata"]["file_name"]
        context_blocks.append(f"[Source {i+1}: {fname}]\n{preview}")

    context_str = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context below. Do not use any outside knowledge. "
        "If the context does not contain enough information to answer, say so clearly. "
        "Cite the source name when referencing specific information.\n\n"
        f"Context:\n{context_str}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query}
    ]
```

---

### Step 8 — Call the LLM

```python
llm_response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.2,     # keep low for factual grounded answers
    max_tokens=1000
)

answer      = llm_response.choices[0].message.content
tokens_used = llm_response.usage.total_tokens
```

---

### Step 9 — Build and return the response

```python
sources = [
    {
        "file_id":         r["metadata"]["file_id"],
        "file_name":       r["metadata"]["file_name"],
        "chunk_id":        r["id"],
        "relevance_score": round(r["score"], 4),
        "excerpt":         r["metadata"]["text_preview"][:300]
    }
    for r in authorized_chunks
]

return {"answer": answer, "sources": sources, "tokens_used": tokens_used}
```

---

## Full Route Handler (Python / FastAPI)

```python
@router.post("/api/rag/query")
async def query_rag(
    body: dict,
    _: None = Depends(verify_internal_key)
):
    tenant_id  = body.get("tenant_id", "").strip()
    user_role  = body.get("user_role", "").strip()
    query_text = body.get("query", "").strip()

    if not query_text:
        raise HTTPException(400, "query is required")
    if not tenant_id or not user_role:
        raise HTTPException(400, "tenant_id and user_role are required")

    top_k           = min(max(body.get("top_k", 5), 1), 20)
    file_ids_filter = body.get("filters", {}).get("file_ids", None)

    # Embed query
    query_vector = embed_query(query_text)

    # Resolve collection
    collection_name = f"tenant_{tenant_id}_chunks"

    # Vector search
    search_filter = {"file_id": {"$in": file_ids_filter}} if file_ids_filter else None
    raw_results   = vector_db.search(collection_name, query_vector, top_k, search_filter)

    if not raw_results:
        raise HTTPException(404, "No relevant documents found")

    # RBAC filter
    authorized_chunks = [r for r in raw_results if user_role in r["metadata"]["allowed_roles"]]

    if not authorized_chunks:
        raise HTTPException(403, "You do not have permission to access the relevant documents")

    # Build prompt and call LLM
    messages     = build_prompt(query_text, authorized_chunks)
    llm_response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=1000
    )
    answer      = llm_response.choices[0].message.content
    tokens_used = llm_response.usage.total_tokens

    sources = [
        {
            "file_id":         r["metadata"]["file_id"],
            "file_name":       r["metadata"]["file_name"],
            "chunk_id":        r["id"],
            "relevance_score": round(r["score"], 4),
            "excerpt":         r["metadata"]["text_preview"][:300]
        }
        for r in authorized_chunks
    ]

    return {"answer": answer, "sources": sources, "tokens_used": tokens_used}
```

---

## Important Rules — Do Not Skip These

1. tenant_id and user_role come from the request body, not a JWT. The .NET backend
   is the source of truth for these values. The RAG API must not re-derive them.

2. Never let the client specify the collection name or tenant_id in a way that could
   be spoofed. The .NET backend must validate that the tenant_id it sends matches the
   authenticated user's organisation before forwarding it.

3. RBAC filtering happens AFTER vector search, inside the application. The vector DB
   does not enforce roles — your code does. Never rely on the vector DB alone.

4. Never pass unauthorized chunks to the LLM. Even if you plan to hide them from
   the response, passing them in the context window risks the LLM leaking content.
   Filter first, then build the prompt.

5. Keep temperature low (0.0 to 0.3). This is a factual RAG endpoint. Higher
   temperatures increase hallucination risk on grounded-answer tasks.

6. The system prompt must instruct the LLM to answer ONLY from the provided context.
   Without this, the LLM will supplement with its own training knowledge, which can
   produce confidently wrong answers.

7. Return 403 vs 404 correctly. Returning 404 when chunks exist but are blocked by
   RBAC leaks information about what exists. 403 means auth succeeded but access was
   denied. 404 means the vector search returned nothing.

8. Rate-limit this endpoint. LLM calls are expensive. Apply per-tenant rate limiting
   (e.g. 60 requests/minute per tenant) before the embedding step so you fail fast
   without incurring API costs.

9. Do not expose internal error messages in the HTTP response. Log them server-side
   and return a generic 500 message.
