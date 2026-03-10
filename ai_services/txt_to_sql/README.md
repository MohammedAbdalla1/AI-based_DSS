
## Text-to-SQL Error Contract

This module returns either:

- `TextToSQLSuccess`
- `TextToSQLError`

`TextToSQLError` response shape:

```json
{
  "status": "error",
  "error_code": "SQL_REJECTED",
  "error_subcode": "WILDCARD_SELECT",
  "message": "Wildcard '*' is not allowed; select explicit columns"
}
```

`error_code` is high-level and stable.  
`error_subcode` is the specific internal reason and is used by retry logic.

## Error Codes

- `INVALID_REQUEST`
- `GENERATION_FAILED`
- `SQL_REJECTED`
- `INTERNAL_ERROR`

## Error Subcodes By Category

### INVALID_REQUEST

- `PAYLOAD_VALIDATION`
- `PAYLOAD_TYPE`

Retry policy: no retry.

### GENERATION_FAILED

- `MISSING_API_KEY`
- `CLIENT_INIT_FAILED`
- `PROMPT_EMPTY`
- `PROVIDER_FAILURE`
- `INVALID_RESPONSE_TYPE`
- `EMPTY_RESPONSE`
- `UNKNOWN` (fallback)

Retry policy:

- Retry once (`max_attempts=2`) for:
  - `EMPTY_RESPONSE`
  - `INVALID_RESPONSE_TYPE`
  - `PROVIDER_FAILURE`
- No retry for all other generation subcodes.

### SQL_REJECTED

- `EMPTY_SQL`
- `PARSE_ERROR`
- `MULTI_STATEMENT`
- `MARKDOWN_OR_COMMENTS`
- `NON_SELECT_QUERY`
- `SELECT_INTO`
- `WILDCARD_SELECT`
- `SELECT_WITHOUT_FROM`
- `UNION_NOT_SUPPORTED`
- `RECURSIVE_CTE`
- `SCHEMA_QUALIFIED_FUNCTION`
- `UNKNOWN_FUNCTION`
- `FUNCTION_NOT_ALLOWED`
- `TABLE_NAME_MISSING`
- `SYSTEM_SCHEMA_ACCESS`
- `TABLE_NOT_ALLOWED`
- `UNAUTHORIZED_TABLE_ACCESS`
- `UNAUTHORIZED_COLUMN_ACCESS`
- `UNRESOLVED_COLUMN`
- `SCHEMA_RESOLUTION_ERROR`
- `UNKNOWN` (fallback)

Retry policy:

- Local cleanup + revalidate once (`max_attempts=2`) for:
  - `MARKDOWN_OR_COMMENTS`
- Regenerate SQL once (`max_attempts=2`) for:
  - `EMPTY_SQL`
  - `PARSE_ERROR`
- No retry for all other SQL rejection subcodes.

### INTERNAL_ERROR

- `UNEXPECTED_EXCEPTION`

Retry policy: no retry.

## Notes

- `error_subcode` is produced at source in `sql_generator.py` and `sql_validator.py`.
- `service.py` uses `error_subcode` for retry decisions.
- Message-based fallback classification is kept for compatibility with legacy/custom raised exceptions.

## OpenAPI Spec

- OpenAPI source of truth: FastAPI runtime schema
- Paths defined:
  - `GET /health`
  - `POST /txt-to-sql`
  - `POST /rag/ask` (placeholder until RAG pipeline is implemented)
- Main schemas:
  - `HealthResponse`
  - `TextToSQLRequest`
  - `TextToSQLSuccess`
  - `TextToSQLError`
  - `TextToSQLResponse`

Use FastAPI docs for interactive contract review.

### Run API Endpoint

Run the shared FastAPI app from the `AI-based_DSS/ai_services` directory:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then call:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/txt-to-sql`
- `POST http://127.0.0.1:8000/rag/ask` (placeholder until RAG pipeline is implemented)
- `Swagger UI: http://127.0.0.1:8000/docs`
- `OpenAPI JSON: http://127.0.0.1:8000/openapi.json`

### HTTP Status Mapping

- `200` (health): service alive (`{"status":"ok"}`)
- `200`: success (`TextToSQLSuccess`)
- `400`: `INVALID_REQUEST`
- `422`: `SQL_REJECTED`
- `502`: `GENERATION_FAILED`
- `500`: `INTERNAL_ERROR` (or unknown fallback)

## .NET Backend Integration

Use `HttpClient` to call:

- `GET /health` for startup/readiness checks
- `POST /txt-to-sql` for SQL generation

Request requirements for `POST /txt-to-sql`:

- Header: `Content-Type: application/json`
- Body:
  - `question` (string, at least 3 words)
  - `db_type` (`postgres` | `mysql` | `sqlserver`)
  - `role_schema` (`{ table: { column: type } }`, non-empty)

Response handling:

- If `status == "success"`, consume `sql`
- If `status == "error"`, use `error_code` and `error_subcode` to branch retry/fallback logic

## Environment Variables

- The API loads `.env` from the `txt_to_sql` directory during startup (`router.py`).
- `GEMINI_API_KEY` can be provided in that `.env` file for local/dev runs.
- Existing process environment values are not overridden (`override=False`).
