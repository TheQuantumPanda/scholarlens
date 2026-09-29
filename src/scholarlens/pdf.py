from __future__ import annotations

from pathlib import Path

import pymupdf

from scholarlens.models import PageText


def default_paper_id(filename: str) -> str:
    stem = Path(filename).stem.strip()
    return stem or "uploaded-paper"


def extract_pdf_pages(
    pdf_bytes: bytes,
    source_filename: str,
    paper_id: str | None = None,
) -> list[PageText]:
    """Extract text from a PDF while preserving page-level provenance."""
    resolved_paper_id = paper_id or default_paper_id(source_filename)
    pages: list[PageText] = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            pages.append(
                PageText(
                    paper_id=resolved_paper_id,
                    source_filename=source_filename,
                    page_number=page_index,
                    text=text,
                )
            )

    return pages
