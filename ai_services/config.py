"""
Shared configuration helpers for ai_services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


AI_SERVICES_DIR = Path(__file__).resolve().parent
_ENV_LOADED = False


def load_environment() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidates = (
        AI_SERVICES_DIR / ".env",
        AI_SERVICES_DIR / "txt_to_sql" / ".env",
    )
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)

    _ENV_LOADED = True


@dataclass(frozen=True)
class Settings:
    ai_services_dir: Path
    data_dir: Path
    rag_data_dir: Path
    rag_uploads_dir: Path
    rag_persist_source_files: bool
    rag_sqlite_path: Path
    rag_qdrant_path: Path
    rag_qdrant_collection: str
    qdrant_url: str
    qdrant_api_key: str
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_top_k: int
    rag_relevance_threshold: float
    rag_rate_limit_per_minute: int
    rag_generation_model: str
    rag_embedding_model: str
    tesseract_cmd: str
    internal_api_key: str
    enforce_internal_api_key: bool

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)
        if self.rag_persist_source_files:
            self.rag_uploads_dir.mkdir(parents=True, exist_ok=True)
        self.rag_qdrant_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_environment()

    data_dir = Path(os.getenv("AI_SERVICES_DATA_DIR", AI_SERVICES_DIR / ".data"))
    rag_data_dir = Path(os.getenv("RAG_DATA_DIR", data_dir / "rag"))

    return Settings(
        ai_services_dir=AI_SERVICES_DIR,
        data_dir=data_dir,
        rag_data_dir=rag_data_dir,
        rag_uploads_dir=Path(os.getenv("RAG_UPLOADS_DIR", rag_data_dir / "uploads")),
        rag_persist_source_files=os.getenv("RAG_PERSIST_SOURCE_FILES", "false").strip().lower() in {"1", "true", "yes", "on"},
        rag_sqlite_path=Path(os.getenv("RAG_SQLITE_PATH", rag_data_dir / "rag.sqlite3")),
        rag_qdrant_path=Path(os.getenv("RAG_QDRANT_PATH", rag_data_dir / "qdrant")),
        rag_qdrant_collection=os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks"),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "2000")),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_relevance_threshold=float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.45")),
        rag_rate_limit_per_minute=int(os.getenv("RAG_RATE_LIMIT_PER_MINUTE", "60")),
        rag_generation_model=os.getenv("RAG_GENERATION_MODEL", "models/gemini-2.5-flash"),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001"),
        tesseract_cmd=os.getenv("TESSERACT_CMD", ""),
        internal_api_key=os.getenv("INTERNAL_API_KEY", ""),
        enforce_internal_api_key=os.getenv("ENFORCE_INTERNAL_API_KEY", "false").strip().lower() in {"1", "true", "yes", "on"},
    )
