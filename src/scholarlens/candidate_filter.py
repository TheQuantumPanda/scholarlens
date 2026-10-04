"""Conservative, deterministic removal of obvious non-evidence passages."""

from __future__ import annotations

import re
import unicodedata

_CAPTION_PREFIX = re.compile(
    r"^(?:(?:distractor\s+document\s+is)\s+|caption:\s*)?"
    r"(?:figure|fig\.?|table)\s+\d+[a-z]?\s*[:.\-–—]\s*"
)
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_REFERENCE_HEADING = re.compile(
    r"\b(?:references|bibliography|works\s+cited|literature\s+cited)\b"
)
_ENTRY_LEAD = re.compile(
    r"(?:\[\d{1,3}\]\s*|\d{1,3}[.)]\s+|"
    r"[a-z][a-z'’.-]+,\s*(?:[a-z]\.\s*){1,3})"
)
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
_PUBLICATION_CUE = re.compile(
    r"\b(?:journal|proceedings|conference|arxiv|doi|vol\.?|volume|"
    r"pp\.?|pages|publisher|press)\b|10\.\d{4,9}/|https?://"
)


def obvious_junk_reason(text: str) -> str | None:
    """Return a narrow junk category without rewriting the source passage."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not normalized:
        return None

    if len(_WORD.findall(normalized)) <= 40 and _CAPTION_PREFIX.match(normalized):
        return "short_caption"
    if _is_reference_dominated(normalized):
        return "reference_dominated"
    return None


def _is_reference_dominated(text: str) -> bool:
    heading = _REFERENCE_HEADING.search(text)
    entries = _recognizable_entries(text, heading.end() if heading else 0)
    if len(entries) < (1 if heading else 3):
        return False

    covered = 0
    for index, start in enumerate(entries):
        end = entries[index + 1] if index + 1 < len(entries) else len(text)
        covered += sum(not char.isspace() for char in text[start:end])
    total = sum(not char.isspace() for char in text)
    threshold = 0.60 if heading else 0.75
    return total > 0 and covered / total >= threshold


def _recognizable_entries(text: str, search_from: int) -> list[int]:
    starts: list[int] = []
    for match in _ENTRY_LEAD.finditer(text, search_from):
        start = match.start()
        prefix = text[search_from:start].rstrip()
        if start != search_from and prefix and not prefix.endswith((".", "!", "?", "\n")):
            continue
        window = text[match.end():match.end() + 500]
        if _YEAR.search(window) and _PUBLICATION_CUE.search(window):
            starts.append(start)
    return starts
