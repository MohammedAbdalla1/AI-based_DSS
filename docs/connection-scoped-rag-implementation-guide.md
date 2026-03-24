# Connection-Scoped RAG + DB Intelligence Implementation Guide (Beginner-Friendly)

## 1. Why this guide exists

This guide explains exactly how to implement a simple but strong graduation-project version of your business idea:

- One shared knowledge base per database connection
- Connection admin uploads documents and refreshes DB context
- All users with access to that connection can ask AI questions
- AI can answer both policy/document questions and database business questions

This document is intentionally detailed and practical for someone implementing this for the first time.

## 2. Business model translated to technical behavior

### 2.1 Target behavior (your Option A)

1. Connection admin uploads files (example: company handbook PDF)
2. System creates embeddings and stores chunks under that connection only
3. Any user who has access to that connection asks questions
4. AI retrieves relevant chunks and answers with context

### 2.2 Added behavior for database intelligence

1. Admin triggers DB context ingestion for a connection
2. System embeds:
   - schema description
   - column profiles
   - sampled rows (safe size)
3. Users ask business questions
4. System can:
   - answer directly from DB-context embeddings
   - or generate SQL, execute it, and summarize trends from results

## 3. Current codebase status (important before coding)

- Text-to-SQL exists and is already integrated:
  - API and orchestration in .NET
  - generation and validation in Python
- RAG endpoint exists as stub only:
  - AI-based_DSS/ai_services/rag/router.py currently returns not implemented

This is good news: most infrastructure patterns are already in your solution.

## 4. Final architecture (MVP)

## 4.1 Components

1. Frontend
   - Admin upload screen
   - Admin refresh DB context button
   - Chat screen for all authorized users
2. .NET Backend
   - Auth and access control
   - RAG orchestration service
   - Text-to-SQL orchestration and SQL execution
   - Persistence of metadata and audit
3. Python AI Service
   - Embedding generation
   - Chunking
   - Vector retrieval
   - Optional answer synthesis helpers
4. PostgreSQL Database
   - Existing product DB
   - New RAG metadata tables
   - pgvector extension for embeddings

## 4.2 Data boundaries (critical)

All RAG rows must include connection_id and all retrieval queries must filter by that same connection_id.

This is your isolation guarantee.

## 5. Project-by-project responsibilities

## 5.1 AI-based_DSS (Python)

Scope:

- Accept ingest requests for docs/DB context
- Chunk text and create embeddings
- Insert/update vectors in PostgreSQL (pgvector)
- Retrieve top-k similar chunks by connection_id filter
- Return context snippets with citations

Main files involved:

- AI-based_DSS/ai_services/main.py
- AI-based_DSS/ai_services/rag/router.py
- AI-based_DSS/ai_services/txt_to_sql/router.py
- AI-based_DSS/ai_services/txt_to_sql/prompt_builder.py
- AI-based_DSS/ai_services/txt_to_sql/service.py

Suggested new Python files:

- AI-based_DSS/ai_services/rag/models.py
- AI-based_DSS/ai_services/rag/service.py
- AI-based_DSS/ai_services/rag/repository.py
- AI-based_DSS/ai_services/rag/chunker.py
- AI-based_DSS/ai_services/rag/embedding_provider.py

## 5.2 src/SqlSpace.Api (.NET API layer)

Scope:

- Expose authenticated endpoints
- Validate request shape
- Read current user id from JWT/session
- Call application services and return standardized API responses

Suggested endpoints:

1. POST /api/connections/{connectionId}/knowledge-base/documents
2. POST /api/connections/{connectionId}/knowledge-base/db-refresh
3. POST /api/connections/{connectionId}/knowledge-base/ask
4. GET /api/connections/{connectionId}/knowledge-base/documents
5. DELETE /api/connections/{connectionId}/knowledge-base/documents/{documentId}

## 5.3 src/SqlSpace.Application (business orchestration)

Scope:

- Access checks (admin vs regular authorized user)
- Workflow orchestration
- Calls to Infrastructure clients (RAG + text-to-SQL)
- Persistence and audit writes
- Error mapping and retries policy

Suggested services:

- IKnowledgeBaseService / KnowledgeBaseService
- IKnowledgeQueryService / KnowledgeQueryService
- IConnectionContextIngestionService / ConnectionContextIngestionService

## 5.4 src/SqlSpace.Infrastructure (integration + data)

Scope:

- EF Core DbContext and repositories
- HttpClient to Python RAG service
- Configuration binding (RagApi section)
- Logging integration

Suggested additions:

- IRagClient + RagClient
- New EF entity mappings
- Dependency injection registrations

## 5.5 src/SqlSpace.Domain (entities and invariants)

Scope:

- RAG metadata entities
- Optional enums for source type, ingestion status

Suggested entities:

- KnowledgeDocument
- KnowledgeChunk (metadata only; vector in pgvector table)
- KnowledgeIngestionRun

## 5.6 FrontEnd

Scope:

- Admin-only controls for upload/refresh
- User chat with citation display
- Connection scope awareness in every API call

## 6. End-to-end flows (step-by-step)

## 6.1 Flow A: Admin uploads document

1. Admin opens connection workspace in frontend
2. Frontend sends file to .NET endpoint with connectionId
3. API verifies current user is connection admin
4. Application service stores document metadata row
5. Application calls Python /rag/ingest with:
   - connectionId
   - sourceType=document
   - document text and metadata
6. Python chunks text and generates embeddings
7. Python inserts vectors with connection_id filter key
8. .NET updates ingestion status to completed/failed
9. API returns summary (chunks count, duration, status)

## 6.2 Flow B: Admin refreshes DB context embeddings

1. Admin clicks Refresh DB Context
2. API verifies admin ownership/access
3. Application service fetches schema context + sampled rows using existing connection logic
4. Application sends context payload to Python /rag/ingest as sourceType=db_context
5. Python chunks and embeds DB context
6. Old context chunks for this connection/source can be replaced/upserted
7. Status returned to admin

## 6.3 Flow C: User asks a question

1. User asks: What products should we stock next month?
2. API verifies user has access to that connection
3. Application calls Python /rag/retrieve for relevant context chunks
4. Application classifies question:
   - doc/policy style -> direct answer synthesis
   - data-analytical style -> text-to-SQL path
5. If analytical:
   - Use existing text-to-SQL generation
   - Execute SQL with safe limits
   - Summarize trend and recommendation from returned result set
6. Return answer + citations + optional generated SQL

## 7. Access control model (must not be skipped)

Rules:

1. Only connection admin can ingest/update/delete corpus
2. Any user with granted connection access can ask queries
3. Retrieval must always include connection_id filter
4. No cross-connection retrieval, ever

Recommended enforcement points:

1. Controller-level authorization attribute (authenticated user)
2. Application-level business checks (admin/has access)
3. Repository-level filter guard (connection_id mandatory)

Defense-in-depth is required: do all three.

## 8. Database design (PostgreSQL + pgvector)

## 8.1 Required extension

Run once:

- CREATE EXTENSION IF NOT EXISTS vector;

## 8.2 Suggested tables

### Table: knowledge_documents

Purpose: one row per uploaded document

Columns:

- document_id UUID PK
- connection_id UUID NOT NULL
- uploaded_by_user_id TEXT NOT NULL
- file_name TEXT NOT NULL
- file_hash TEXT NULL
- source_type TEXT NOT NULL (document, db_context)
- status TEXT NOT NULL (pending, processing, completed, failed)
- created_at TIMESTAMPTZ NOT NULL
- processed_at TIMESTAMPTZ NULL
- error_message TEXT NULL
- is_deleted BOOLEAN NOT NULL DEFAULT FALSE

Indexes:

- idx_knowledge_documents_connection_id
- idx_knowledge_documents_status

### Table: knowledge_chunks

Purpose: chunk metadata + searchable vector

Columns:

- chunk_id UUID PK
- connection_id UUID NOT NULL
- document_id UUID NULL (nullable for db_context if desired)
- source_type TEXT NOT NULL
- source_reference TEXT NULL (table/column or file page)
- chunk_text TEXT NOT NULL
- token_count INT NOT NULL
- embedding vector(768) NOT NULL  (or provider dimension)
- created_at TIMESTAMPTZ NOT NULL

Indexes:

- idx_knowledge_chunks_connection_id
- idx_knowledge_chunks_source_type
- ivfflat/hnsw index on embedding for ANN search

### Table: knowledge_ingestion_runs

Purpose: operational audit of ingestion jobs

Columns:

- run_id UUID PK
- connection_id UUID NOT NULL
- initiated_by_user_id TEXT NOT NULL
- trigger_type TEXT NOT NULL (upload, db_refresh, reindex)
- started_at TIMESTAMPTZ NOT NULL
- completed_at TIMESTAMPTZ NULL
- status TEXT NOT NULL
- processed_items INT NOT NULL DEFAULT 0
- error_message TEXT NULL

Indexes:

- idx_ingestion_runs_connection_id
- idx_ingestion_runs_started_at

## 8.3 Notes on embedding dimensions

Your vector column dimension must match your embedding model output exactly.

If you switch providers, you may need a new column/table or migration strategy.

## 9. API contract suggestions

## 9.1 Ingest document (admin)

Request:

- multipart/form-data
- file
- optional title/description

Response:

- documentId
- chunksCreated
- status
- warnings

## 9.2 Refresh DB context (admin)

Request body:

- profileMode: basic | full
- sampleRowsPerTable
- includeTables/excludeTables optional

Response:

- runId
- tablesProcessed
- chunksCreated
- status

## 9.3 Ask question (authorized users)

Request body:

- question
- mode: auto | docs | analytics
- includeSqlInResponse: bool

Response body:

- answer
- citations[]
- generatedSql (optional)
- summaryMetrics (optional)

## 10. Python AI service details

## 10.1 RAG ingest endpoint behavior

Pseudo-flow:

1. Validate payload
2. Build chunks by source type
3. Generate embeddings in batches
4. Upsert chunks into pgvector table
5. Return counts and timing

Chunking guidance:

- Start with chunk size 600-900 tokens
- Overlap 80-120 tokens
- Keep chunk source metadata for citation quality

## 10.2 RAG retrieve endpoint behavior

Pseudo-flow:

1. Embed user question
2. Vector search with WHERE connection_id = :connectionId
3. Take top-k (start with 5)
4. Optional rerank by lexical score
5. Return context snippets with source references

## 10.3 Question routing strategy

Basic classifier categories:

1. docs_qa
2. analytics_qa
3. hybrid

Start with simple keyword + prompt-based classifier. Keep deterministic fallback to analytics when numbers/time periods are requested.

## 11. .NET orchestration details

## 11.1 Service call order for Ask

1. Check access to connection
2. Retrieve RAG context from Python service
3. Decide route:
   - Direct answer from context, or
   - Text-to-SQL generation + execution + summarization
4. Persist query history
5. Return final answer object

## 11.2 Error handling policy

Recommended user-facing error classes:

1. AccessDenied
2. ConnectionNotFound
3. IngestionFailed
4. RagServiceUnavailable
5. SqlGenerationFailed
6. SqlExecutionFailed

Always return safe, non-sensitive details to frontend.

## 12. Configuration you need

## 12.1 .NET appsettings

Example section:

{
  "RagApi": {
    "BaseUrl": "http://localhost:8000",
    "TimeoutSeconds": 60,
    "ApiKey": ""
  },
  "KnowledgeBase": {
    "MaxUploadSizeMb": 20,
    "AllowedExtensions": [".pdf", ".txt", ".md", ".csv"],
    "DefaultTopK": 5,
    "DbContextSampleRowsPerTable": 50
  }
}

## 12.2 Python environment variables

Suggested:

- RAG_DB_DSN
- EMBEDDING_PROVIDER
- EMBEDDING_MODEL
- LLM_MODEL
- MAX_CHUNK_TOKENS
- CHUNK_OVERLAP_TOKENS

## 13. Security, privacy, and safety

Minimum controls:

1. Validate file type and size before processing
2. Strip executable content/macros from uploads
3. Log ingestion and ask events with user and connection id
4. Parameterize SQL and enforce safe query limits
5. Restrict retrieval to connection_id server-side only
6. Do not expose secrets in logs

## 14. Performance and scaling guidance (MVP-safe)

Start simple:

- Synchronous ingestion
- Batch embeddings (for example 32-128 chunks per call)
- ANN index on embedding column
- Top-k small (3-8)

If growth increases:

1. Move ingestion to background queue
2. Add scheduled DB-context refresh
3. Add chunk dedup by file hash or content hash

## 15. Suggested implementation sequence (practical)

1. Create DB tables + migration + pgvector setup
2. Implement Python ingest/retrieve endpoints
3. Add .NET RagClient + DI config
4. Add admin upload and DB refresh endpoints
5. Add ask endpoint with route-to-analytics behavior
6. Add tests for access and isolation
7. Wire frontend pages/buttons

This order gives you visible progress quickly while keeping risk low.

## 16. Testing checklist (very important)

## 16.1 Access tests

1. Non-admin cannot upload
2. Non-admin cannot refresh DB context
3. Authorized non-admin can ask
4. Unauthorized user cannot ask

## 16.2 Isolation tests

1. Same question in connection A cannot retrieve chunks from connection B
2. Deleting document removes its chunks from retrieval

## 16.3 Functional tests

1. Upload handbook -> ask PTO policy -> correct citation appears
2. Refresh sales DB context -> ask top selling category trend -> answer based on SQL result

## 16.4 Failure tests

1. Python service down
2. Embedding provider timeout
3. Invalid file upload
4. Empty retrieval result (graceful fallback)

## 17. What success looks like (demo-ready)

At demo time, show this sequence:

1. Admin uploads PDF once
2. Employee asks a policy question and gets grounded answer with citation
3. Admin refreshes DB context for sales data
4. Employee asks sales trend/recommendation question
5. System shows answer based on SQL + context, scoped only to that connection

If these 5 steps work reliably, your MVP is strong.

## 18. Common beginner mistakes to avoid

1. Forgetting connection_id filter in retrieval query
2. Mixing admin and user privileges in one endpoint
3. Embedding too much raw table data in MVP
4. Returning generated SQL without execution safeguards
5. Missing ingestion/error observability

## 19. Optional phase 2 enhancements

1. Background ingestion jobs
2. Scheduled DB-context refresh per connection
3. Forecasting module (moving average, Prophet, or other)
4. Better reranking and citation confidence scoring
5. Feedback loop for answer quality

## 20. Final recommendation

For graduation scope, keep it simple and reliable:

- Shared connection-level RAG
- Strong access/isolation
- SQL-backed descriptive analytics
- Good observability and tests

This gives a professional system design without overengineering.
