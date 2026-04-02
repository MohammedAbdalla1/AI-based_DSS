"""
File validation, storage, parsing, chunking, and image-aware text extraction for RAG ingestion.
"""

from __future__ import annotations

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


def build_storage_reference(*, tenant_id: str, file_id: str, file_name: str, persisted_path: Path | None = None) -> str:
    if persisted_path is not None:
        return str(persisted_path)
    return f"transient://{tenant_id}/{file_id}/{file_name}"


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
        # Tesseract or Pillow not installed - skip silently
        return ""
    except Exception:
        return ""


def _vision_describe_image(image_bytes: bytes) -> str:
    """
    Send an image to Gemini vision and return a detailed text description.
    Used for charts, diagrams, photos, and any non-OCR image content.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        client = genai.Client(api_key=api_key)
        settings = get_settings()

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
                    "Be thorough - this description will be used for document search and question answering."
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
    - First try OCR (Tesseract) - fast and free, works well for text-heavy images
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
    - Embedded images -> OCR or Gemini vision
    - Fully scanned pages -> whole page rasterized at high DPI -> OCR/vision
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

        # 2. Check if this page is a scanned page.
        #    A scanned page has no machine text AND has one or more images
        #    that together cover most of the page area.
        #    In this case, rasterize the whole page at high DPI for best quality.
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        images_on_page = page.get_images(full=True)

        is_scanned_page = False
        if not page_text and images_on_page:
            # Calculate what fraction of the page area is covered by images
            covered_area = 0.0
            for img_info in images_on_page:
                # get_image_rects returns the bounding boxes of where the image
                # appears on the page
                rects = page.get_image_rects(img_info[0])
                for rect in rects:
                    covered_area += rect.width * rect.height

            # If images cover more than 60% of the page, treat as scanned
            if page_area > 0 and (covered_area / page_area) >= 0.6:
                is_scanned_page = True

        if is_scanned_page:
            # Rasterize the whole page at high DPI for best OCR/vision quality
            try:
                pix = page.get_pixmap(dpi=200)
                page_img_bytes = pix.tobytes("png")
                description = _process_image(page_img_bytes)
                if description:
                    parts.append(description)
            except Exception:
                pass
        else:
            # 3. Process individual embedded images (charts, diagrams, photos
            #    embedded in a text-heavy page)
            for img_info in images_on_page:
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
# Text normalization and chunking - unchanged from original
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
