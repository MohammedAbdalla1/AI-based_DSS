# Codex Instructions — Fix Scanned PDF Detection in ingestion.py

## The problem

In `rag/ingestion.py`, the function `_extract_pdf_text` tries to detect
fully scanned pages (pages that are one big image) with this condition:

```python
if not page_text and not page.get_images(full=True):
```

This is wrong. A scanned PDF page appears as a large embedded image in
`page.get_images()`, so this condition is NEVER true for scanned pages.
The result is scanned pages get processed as small embedded images at
low resolution, producing poor or empty OCR output.

## The fix

Replace the entire `_extract_pdf_text` function in `rag/ingestion.py`.

Find this function:

```python
def _extract_pdf_text(contents: bytes) -> str:
```

Replace the entire function body (everything from the opening `"""` docstring
to the final `return` statement and `doc.close()`) with the following:

```python
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
```

## Also fix the dead line in _vision_describe_image

In the same file `rag/ingestion.py`, inside `_vision_describe_image`,
find and delete this line (it computes a value and immediately throws it away):

```python
        base64.b64encode(image_bytes).decode("utf-8")
```

The line directly above it (`api_key = os.getenv(...)`) and the line
directly below it (`client = genai.Client(...)`) stay unchanged.
Only this one line is removed.

Also remove the `import base64` line at the top of the file since
base64 is no longer used anywhere:

Find at the top of the file:
```python
import base64
import io
```

Replace with:
```python
import io
```

## What the fix does

The key change is the scanned page detection logic. Instead of checking
`not page.get_images()` (which is always False for scanned pages), it now:

1. Gets all images on the page
2. Calculates what percentage of the page area they cover
3. If images cover 60%+ of the page with no machine text → it is a scanned
   page → rasterize the whole page at 200 DPI and send to OCR/vision
4. If images are present but only cover a small area → they are embedded
   diagrams/charts in a text document → process each image individually

This correctly handles all four document types:

| Document type | Detection | Processing |
|---|---|---|
| Normal text PDF | page_text is non-empty | text extracted directly |
| PDF with embedded charts | page_text present + small images | text + per-image description |
| Scanned PDF (whole page is image) | no text + images cover 60%+ of page | full page rasterized at 200 DPI |
| Mixed scan + text | partial text + large image | text + full page rasterize |
