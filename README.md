# ScholarLens

ScholarLens is an evidence-grounded multi-paper research synthesis system. The current implementation covers Phase 1 ingestion and chunk inspection plus the Phase 2 semantic retrieval baseline.

## Status

ScholarLens currently provides a Streamlit app that accepts multiple text-based PDFs, extracts page text with PyMuPDF, creates page-bounded chunks, embeds them with `BAAI/bge-small-en-v1.5`, and indexes them in a transient Chroma collection. A user can submit a semantic query and directly inspect the ranked Top-K retrieved passages with their provenance and cosine distance.

The app displays retrieved evidence only. It does not generate, summarize, or synthesize an answer.

## Prerequisites

- Python 3.11
- `uv`

## Setup

Sync the project environment:

```bash
uv sync
```

## Launch the App

Run the Streamlit evidence retriever:

```bash
uv run streamlit run src/scholarlens/app.py
```

On first indexing, Sentence Transformers downloads the `BAAI/bge-small-en-v1.5` model and caches it locally. This requires network access and can take some time. Later runs reuse the local model cache, and the Streamlit app caches the loaded model resource while the server is running.

## Tests

Run the current test suite:

```bash
uv run python -m unittest discover -s tests
```

## Current Chunking Behavior

The Phase 1 chunker uses simple word-based chunking within each page. By default, it creates 250-word chunks with a 30-word overlap. Chunks do not cross page boundaries.

These settings remain provisional and will be evaluated or replaced as retrieval development progresses. The embedding model has a finite input length, so sufficiently long chunks may be truncated by Sentence Transformers during embedding. Phase 2 keeps this concern visible without changing the chunking strategy.

## Semantic Retrieval

After processing the uploaded PDFs, ScholarLens embeds each existing `TextChunk` and stores its text, embedding, and provenance metadata in an in-memory Chroma collection configured for cosine distance. Queries use the same embedding model, and Top-K results are shown in Chroma rank order.

The collection is transient: it is held for the current Streamlit session and is not persisted across server or session restarts.

## Provenance

The ingestion and retrieval pipeline retains:

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
- Embedding inputs may be truncated at the BGE model's finite input length.
- The Chroma index is transient and must be rebuilt after a session restart.
- No LLM answer generation, synthesis, or evidence verification yet.
