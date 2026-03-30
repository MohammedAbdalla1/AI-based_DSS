# Codex Instructions — Image & Scan Handling in ingestion.py

## Context

The current `ingestion.py` only extracts machine-readable text from PDFs and DOCX files.
It silently skips all images, embedded charts, diagrams, and scanned pages.

This instruction adds full image awareness to the ingestion pipeline.
The change is **entirely inside `ingestion.py`**. No other file needs to change.

The project already uses Gemini (via `generator.py`) so vision calls also use Gemini.
Tesseract is used for scanned pages (OCR) — it is free, fast, and accurate for text.

---

## New behavior after this change

| Content type                        | How it is handled                              |
|-------------------------------------|------------------------------------------------|
| Normal PDF text                     | Extracted directly as before — unchanged       |
| Scanned PDF page (no machine text)  | Page rasterized to image → Tesseract OCR       |
| PDF page with embedded images       | Text extracted normally + each image sent to Gemini vision for description |
| PDF with charts or diagrams         | Gemini vision describes what the chart shows   |
| DOCX with images inside             | Paragraph text extracted + each image sent to Gemini vision |
| DOCX with no images                 | Extracted as before — unchanged                |

All extracted text and image descriptions are merged into one string per file,
then passed to the existing chunker and embedder unchanged.

---

## New dependencies to install

```bash
pip install pymupdf pytesseract pillow
```

Also install Tesseract on the system (not pip — it is a system binary):

- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
  and add it to PATH, or set `TESSERACT_CMD` in .env to the full path.
- **Linux/Mac:** `sudo apt install tesseract-ocr` or `brew install tesseract`

---

## New .env variable to add

Add this to the `.env` file. Leave empty on Linux/Mac if tesseract is on PATH.

```env
TESSERACT_CMD=
```

Example for Windows:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Changes to config.py

**File:** `config.py` (root folder)

Add one field to the `Settings` dataclass:

```python
tesseract_cmd: str
```

Add one line to the `get_settings()` return:

```python
tesseract_cmd=os.getenv("TESSERACT_CMD", ""),
```

---

## Full replacement of ingestion.py

**File:** `rag/ingestion.py`

Replace the entire file with the following. Every existing function is preserved
with the same signature. Only `extract_text` and the PDF/DOCX parsing internals change.

```python
"""
File validation, storage, parsing, chunking, and image-aware text extraction for RAG ingestion.
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from config import get_settings


SUPPORTED_FILE_TYPES = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


class DocumentIngestionError(Exception):
    def __init__(self, message: str, subcode: str = "UNKNOWN"):
        super().__init__(message)
        self.subcode = subcode


@dataclass(frozen=True)
class StoredUpload:
    file_name: str
    ext: str
    mime_type: str
    storage_path: Path
    contents: bytes


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str


def _ingestion_error(subcode: str, message: str) -> DocumentIngestionError:
    return DocumentIngestionError(message=message, subcode=subcode)


def sanitize_file_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "document.txt"


async def read_and_validate_upload(uploaded_file) -> tuple[bytes, str, str, str]:
    file_name = sanitize_file_name(getattr(uploaded_file, "filename", "") or "document.txt")
    ext = Path(file_name).suffix.lower()
    mime_type = getattr(uploaded_file, "content_type", None) or "application/octet-stream"

    if ext not in SUPPORTED_FILE_TYPES:
        raise _ingestion_error("UNSUPPORTED_FILE_TYPE", f"Unsupported file type: {ext}")

    contents = await uploaded_file.read()
    if not contents:
        raise _ingestion_error("EMPTY_FILE", "Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise _ingestion_error("FILE_TOO_LARGE", "File exceeds 20MB limit.")

    return contents, file_name, ext, mime_type


def save_to_storage(*, tenant_id: str, file_id: str, file_name: str, contents: bytes) -> Path:
    settings = get_settings()
    settings.ensure_runtime_dirs()

    destination = settings.rag_uploads_dir / tenant_id / file_id / file_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        destination.write_bytes(contents)
    except Exception as exc:
        raise _ingestion_error(
            "FILE_SAVE_FAILED",
            f"Failed to persist uploaded file: {type(exc).__name__}",
        ) from exc

    return destination


def delete_saved_file(storage_path: Path) -> None:
    try:
        if storage_path.exists():
            shutil.rmtree(storage_path.parent)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Internal image helpers
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    """Point pytesseract at the system binary if TESSERACT_CMD is set."""
    settings = get_settings()
    cmd = settings.tesseract_cmd.strip() if settings.tesseract_cmd else ""
    if cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = cmd
        except ModuleNotFoundError:
            pass


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run Tesseract OCR on raw image bytes and return extracted text."""
    try:
        import pytesseract
        from PIL import Image
        _configure_tesseract()
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image).strip()
    except ModuleNotFoundError:
        # Tesseract or Pillow not installed — skip silently
        return ""
    except Exception:
        return ""


def _vision_describe_image(image_bytes: bytes) -> str:
    """
    Send an image to Gemini vision and return a detailed text description.
    Used for charts, diagrams, photos, and any non-OCR image content.
    """
    try:
        import os
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        client = genai.Client(api_key=api_key)
        settings = get_settings()

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.models.generate_content(
            model=settings.rag_generation_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(
                    "Describe all information visible in this image in detail. "
                    "If it contains a chart or graph, extract all values, labels, axes, and trends. "
                    "If it contains a table, reproduce it as plain text. "
                    "If it contains a diagram or flowchart, describe the structure and all labels. "
                    "If it is a photo or illustration, describe what it shows. "
                    "Be thorough — this description will be used for document search and question answering."
                ),
            ],
            config={"temperature": 0.1},
        )
        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""
    except Exception:
        return ""


def _process_image(image_bytes: bytes) -> str:
    """
    Decide how to extract meaning from an image:
    - First try OCR (Tesseract) — fast and free, works well for text-heavy images
    - If OCR returns very little text, fall back to Gemini vision for semantic description
    This covers both scanned text images AND charts/diagrams.
    """
    ocr_text = _ocr_image_bytes(image_bytes)

    # If OCR found meaningful text (more than a few words), trust it
    if len(ocr_text.split()) >= 10:
        return ocr_text

    # Otherwise use vision LLM for semantic understanding (charts, diagrams, photos)
    vision_text = _vision_describe_image(image_bytes)
    if vision_text:
        return vision_text

    # Fall back to whatever OCR got, even if sparse
    return ocr_text


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(contents: bytes) -> str:
    """
    Extract all text from a PDF, including:
    - Machine-readable text (via PyMuPDF)
    - Embedded images → OCR or Gemini vision
    - Fully scanned pages → page rasterized to image → OCR
    """
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as exc:
        raise _ingestion_error(
            "PARSER_NOT_AVAILABLE",
            "pymupdf is not installed. Run: pip install pymupdf",
        ) from exc

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception as exc:
        raise _ingestion_error(
            "TEXT_EXTRACTION_FAILED",
            f"Failed to open PDF: {type(exc).__name__}",
        ) from exc

    page_texts: list[str] = []

    for page in doc:
        parts: list[str] = []

        # 1. Extract machine-readable text from this page
        page_text = page.get_text().strip()
        if page_text:
            parts.append(page_text)

        # 2. Extract and process embedded images on this page
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image")
                if not img_bytes:
                    continue
                description = _process_image(img_bytes)
                if description:
                    parts.append(f"[Image: {description}]")
            except Exception:
                continue

        # 3. If the page has NO machine text at all (scanned page),
        #    rasterize the whole page and run it through OCR/vision
        if not page_text and not page.get_images(full=True):
            try:
                pix = page.get_pixmap(dpi=200)
                page_img_bytes = pix.tobytes("png")
                description = _process_image(page_img_bytes)
                if description:
                    parts.append(description)
            except Exception:
                pass

        if parts:
            page_texts.append("\n".join(parts))

    doc.close()
    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

def _extract_docx_text(contents: bytes) -> str:
    """
    Extract all text and images from a DOCX file.
    Paragraph text is extracted directly.
    Embedded images are processed via OCR or Gemini vision.
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except ModuleNotFoundError as exc:
        raise _ingestion_error(
            "PARSER_NOT_AVAILABLE",
            "python-docx is not installed. Run: pip install python-docx",
        ) from exc

    try:
        document = Document(io.BytesIO(contents))
    except Exception as exc:
        raise _ingestion_error(
            "TEXT_EXTRACTION_FAILED",
            f"Failed to open DOCX: {type(exc).__name__}",
        ) from exc

    parts: list[str] = []

    # Extract paragraph text
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    # Extract images from the document's relationships
    for rel in document.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            image_part = rel.target_part
            img_bytes = image_part.blob
            if not img_bytes:
                continue
            description = _process_image(img_bytes)
            if description:
                parts.append(f"[Image: {description}]")
        except Exception:
            continue

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------

def extract_text(contents: bytes, ext: str) -> str:
    try:
        if ext == ".txt":
            text = contents.decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            text = _extract_pdf_text(contents)
        elif ext == ".docx":
            text = _extract_docx_text(contents)
        else:
            raise _ingestion_error("UNSUPPORTED_FILE_TYPE", f"Unsupported file type: {ext}")
    except DocumentIngestionError:
        raise
    except Exception as exc:
        raise _ingestion_error(
            "TEXT_EXTRACTION_FAILED",
            f"Failed to parse file: {type(exc).__name__}",
        ) from exc

    normalized = normalize_text(text)
    if not normalized:
        raise _ingestion_error("EMPTY_DOCUMENT", "File appears empty or unreadable.")
    return normalized


# ---------------------------------------------------------------------------
# Text normalization and chunking — unchanged from original
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def chunk_text(text: str) -> list[TextChunk]:
    settings = get_settings()
    chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap

    if overlap >= chunk_size:
        raise _ingestion_error("INVALID_CHUNK_CONFIG", "RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
    if not text.strip():
        raise _ingestion_error("EMPTY_DOCUMENT", "Document text is empty after normalization.")

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_value = text[start:end].strip()
        if chunk_value:
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    text=chunk_value,
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        start += chunk_size - overlap

    if not chunks:
        raise _ingestion_error("EMPTY_DOCUMENT", "No chunks were produced from the document.")

    return chunks
```

---

## What changed vs the original ingestion.py

| Section | Original | After this change |
|---|---|---|
| PDF extraction | `pypdf` text only | `PyMuPDF` text + embedded images + scanned page fallback |
| DOCX extraction | paragraph text only | paragraph text + all embedded images |
| Image processing | not handled | OCR first, Gemini vision if OCR returns < 10 words |
| TXT extraction | unchanged | unchanged |
| chunk_text | unchanged | unchanged |
| normalize_text | unchanged | unchanged |
| All function signatures | unchanged | unchanged |

## Install summary

```bash
# Python packages
pip install pymupdf pytesseract pillow

# System binary (required for Tesseract OCR)
# Linux:
sudo apt install tesseract-ocr
# Mac:
brew install tesseract
# Windows:
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Then set TESSERACT_CMD in .env to the full path
```
