# AI Services

Shared FastAPI app for the graduation project AI features:

- `POST /txt-to-sql`
- `POST /api/rag/ingest`
- `POST /api/rag/query`
- `GET /health`

## Local Setup

1. Copy `.env.example` to `.env`
2. Fill in at least `GEMINI_API_KEY`
3. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. If you want scanned/image-only PDF OCR on Windows, install Tesseract and set:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

5. By default, uploaded source files are processed in memory and are not kept on disk. If you want to retain original files locally, set:

```env
RAG_PERSIST_SOURCE_FILES=true
```

6. Run the API from this folder:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI:

```txt
http://127.0.0.1:8000/docs
```

## Environment Notes

- If `QDRANT_URL` is empty, the app uses local embedded Qdrant under `.data/rag/qdrant`
- If `QDRANT_URL` is set, the app uses remote Qdrant Cloud or a self-hosted Qdrant server
- If `RAG_PERSIST_SOURCE_FILES=false`, original uploaded files are not stored locally; only vectors and metadata are kept
- If `ENFORCE_INTERNAL_API_KEY=true`, both RAG endpoints require `X-Internal-API-Key`

## Docker

Build from the `AI-based_DSS/ai_services` folder:

```powershell
docker build -t ai-services .
```

Run with your env file:

```powershell
docker run --rm -p 8000:8000 --env-file .env ai-services
```

The Docker image installs Tesseract on Linux, so `TESSERACT_CMD` can stay empty there unless you want a custom path.
