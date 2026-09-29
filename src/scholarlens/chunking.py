from __future__ import annotations

from collections.abc import Iterable

from scholarlens.models import PageText, TextChunk

DEFAULT_CHUNK_WORDS = 250
DEFAULT_OVERLAP_WORDS = 30


def chunk_pages(
    pages: Iterable[PageText],
    max_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Create simple word-based chunks without crossing page boundaries."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    chunks: list[TextChunk] = []
    chunk_sequence = 1

    for page in pages:
        words = page.text.split()
        if not words:
            continue

        start = 0
        page_chunk_sequence = 1
        step = max_words - overlap_words

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunks.append(
                TextChunk(
                    paper_id=page.paper_id,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_id=(
                        f"{page.paper_id}:chunk-{chunk_sequence:04d}:"
                        f"p{page.page_number}-c{page_chunk_sequence}"
                    ),
                    text=" ".join(chunk_words),
                )
            )
            chunk_sequence += 1
            page_chunk_sequence += 1
            start += step

    return chunks
