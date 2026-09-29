# ScholarLens

ScholarLens is an evidence-grounded multi-paper research synthesis system. The current implementation is only the Phase 1 ingestion and chunk-inspection baseline.

## Status

ScholarLens currently provides a Streamlit app that accepts text-based PDFs, extracts page text, and displays simple page-bounded word chunks with provenance. It does not yet implement embeddings, retrieval, LLM answering, synthesis, or evidence verification.

## Prerequisites

- Python 3.11
- `uv`

## Setup

Sync the project environment:

```bash
uv sync
```

## Launch the App

Run the Streamlit chunk inspector:

```bash
uv run streamlit run src/scholarlens/app.py
```

## Tests

Run the current test suite:

```bash
uv run python -m unittest discover -s tests
```

## Current Chunking Behavior

The Phase 1 chunker uses simple word-based chunking within each page. By default, it creates 250-word chunks with a 30-word overlap. Chunks do not cross page boundaries.

These settings are provisional and will be evaluated and likely replaced as retrieval development progresses.

## Provenance

The current ingestion baseline retains:

- `paper_id`
- source filename
- page number
- `chunk_id`
- source text

## Current Limitations

- Text-based PDFs only.
- No OCR.
- Simple page-bounded word chunking only.
- PDF text extraction and layout artifacts may occur.
- No embeddings, retrieval, vector database, LLM answering, or synthesis yet.
