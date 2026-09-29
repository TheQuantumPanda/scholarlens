from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    paper_id: str
    source_filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    paper_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    text: str
