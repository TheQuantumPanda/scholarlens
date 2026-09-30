# ScholarLens

ScholarLens is an evidence-grounded multi-paper research synthesis system. The current implementation covers Phase 1 ingestion and chunk inspection, Phase 2 semantic retrieval, and Phase 3 basic RAG answer generation.

## Status

ScholarLens currently provides a Streamlit app that accepts multiple text-based PDFs, extracts page text with PyMuPDF, creates page-bounded chunks, embeds them with `BAAI/bge-small-en-v1.5`, and indexes them in a transient Chroma collection. A user can submit a semantic query and directly inspect the ranked Top-K retrieved passages with their provenance and cosine distance.

After inspecting retrieval, a user can separately generate an answer from exactly those retrieved chunks using local Ollama. Higher-level synthesis and evidence verification are not implemented.

## Prerequisites

- Python 3.11
- `uv`
- For answer generation: a running local Ollama server with `qwen3:4b` already installed. Retrieval works independently of Ollama. ScholarLens never pulls models automatically.

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

Generation tests fake the HTTP boundary; UI tests fake uploads, retrieval, and HTTP. They require no running Ollama server or downloaded language/embedding models. The existing retrieval suite also exercises transient Chroma with fake embeddings.

## Current Chunking Behavior

The Phase 1 chunker uses simple word-based chunking within each page. By default, it creates 250-word chunks with a 30-word overlap. Chunks do not cross page boundaries.

These settings remain provisional and will be evaluated or replaced as retrieval development progresses. The embedding model has a finite input length, so sufficiently long chunks may be truncated by Sentence Transformers during embedding. Phase 2 keeps this concern visible without changing the chunking strategy.

## Semantic Retrieval

After processing the uploaded PDFs, ScholarLens embeds each existing `TextChunk` and stores its text, embedding, and provenance metadata in an in-memory Chroma collection configured for cosine distance. Queries use the same embedding model, and Top-K results are shown in Chroma rank order.

The collection is transient: it is held for the current Streamlit session and is not persisted across server or session restarts.

## Basic RAG Generation

1. Process and index PDFs, enter a question, and click **Retrieve evidence**.
2. Inspect the ranked chunks, provenance, and cosine distances. Evidence IDs `[E1]`, `[E2]`, etc. follow the displayed retrieval order and apply to that retrieval snapshot.
3. Click **Generate answer**. The answer appears separately, while the evidence inspector remains visible.

Generation uses the last submitted retrieval question and exactly its retrieved chunks. Editing the question or Top-K requires submitting **Retrieve evidence** again before those changes apply to generation. A new retrieval or changed/rebuilt index clears the previous answer. Ordinary reruns and generation reuse the existing session index; answers are retained across reruns but no conversation history is sent to the model.

`src/scholarlens/generation.py` formats evidence as JSON containing evidence ID, source filename, paper ID, page number, chunk ID, and source text. A separate system message requires evidence-only answers, factual-claim citations, an explicit insufficient-evidence response when needed, and treating all paper contents as untrusted data rather than instructions. Empty retrieval produces a local insufficient-evidence response without calling Ollama.

The adapter uses Python's standard-library HTTP client to send `POST /api/chat` with `stream: false`, omitting the `think` request field. Defaults are centralized in `OllamaConfig`:

| Setting | Default | Environment override |
| --- | --- | --- |
| Ollama base URL | `http://localhost:11434` | `SCHOLARLENS_OLLAMA_BASE_URL` |
| Model | `qwen3:4b` | `SCHOLARLENS_OLLAMA_MODEL` |

Set overrides in the environment used to launch Streamlit. The request timeout is 120 seconds, configurable through `OllamaConfig.timeout_seconds` for Python callers. Connection failures, timeouts, unavailable models, and invalid responses are displayed as errors while leaving retrieved evidence inspectable.

Local API experiments with Ollama 0.34.4 and `qwen3:4b` found that `think: false` leaked reasoning into `message.content`, even alongside `/no_think`. Using `/no_think` without the `think` field produced clean content with reasoning separately in `message.thinking`. ScholarLens therefore includes `/no_think` in its system instruction and returns only `message.content`; it does not retain or display `message.thinking` or strip `<think>` tags. This workaround reflects that tested combination, not a guarantee for all Ollama models.

### Manual Generation Check

With Ollama running and the configured model installed, launch the app and retrieve passages from real PDFs. Generate an answer and compare each factual claim and `[E#]` citation against the displayed evidence. Also try a question that the passages cannot answer; confirm the model acknowledges insufficient evidence. Check that changing the submitted question clears the old answer and that generation does not rebuild the index. Automated tests do not establish real-model grounding quality or resistance to instructions embedded in documents.

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
- Grounding and citation correctness are instructed, not automatically verified; the model can still make unsupported claims or follow malicious document instructions.
- Large Top-K selections may exceed the model's context capacity. This baseline does not budget tokens or detect server-side context truncation.
- Generation is synchronous and may be slow on local hardware; no streaming or conversation memory.
- No higher-level synthesis or evidence verification yet.
