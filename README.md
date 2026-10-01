# ScholarLens

ScholarLens is an evidence-grounded multi-paper research synthesis system. The current implementation covers Phase 1 ingestion and chunk inspection, Phase 2 semantic retrieval, Phase 3 basic RAG answer generation, and a Phase 4B grouped structured individual-paper analysis prototype.

## Status

ScholarLens currently provides a Streamlit app that accepts multiple text-based PDFs, extracts page text with PyMuPDF, creates page-bounded chunks, embeds them with `BAAI/bge-small-en-v1.5`, and indexes them in a transient Chroma collection. A user can submit a semantic query and directly inspect the ranked Top-K retrieved passages with their provenance and cosine distance.

After inspecting retrieval, a user can separately generate an answer from exactly those retrieved chunks using local Ollama. A user can also select one indexed paper for structured extraction of three analysis fields. Higher-level synthesis and evidence verification are not implemented.

## Prerequisites

- Python 3.11
- `uv`
- For answer generation and structured analysis: a running local Ollama server with `qwen3:4b` already installed. Retrieval works independently of Ollama. ScholarLens never pulls models automatically.

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

Generation and analysis tests fake the HTTP boundary; UI tests fake uploads, retrieval, and HTTP. They require no running Ollama server or downloaded language/embedding models. The retrieval suite also exercises transient Chroma with fake embeddings, including paper isolation before Top-K selection. Analysis tests cover all three fields, structured validation failures, evidence-ID resolution, and UI invalidation when the selected paper or index changes.

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
| Request timeout | 240 seconds | `SCHOLARLENS_OLLAMA_TIMEOUT_SECONDS` |

Set overrides in the environment used to launch Streamlit. Python callers can also override the request timeout through `OllamaConfig.timeout_seconds`. Connection failures, timeouts, unavailable models, and invalid responses are displayed as errors while leaving retrieved evidence inspectable.

Local API experiments with Ollama 0.34.4 and `qwen3:4b` found that `think: false` leaked reasoning into `message.content`, even alongside `/no_think`. Using `/no_think` without the `think` field produced clean content with reasoning separately in `message.thinking`. ScholarLens therefore includes `/no_think` in its system instruction and returns only `message.content`; it does not retain or display `message.thinking` or strip `<think>` tags. This workaround reflects that tested combination, not a guarantee for all Ollama models.

### Manual Generation Check

With Ollama running and the configured model installed, launch the app and retrieve passages from real PDFs. Generate an answer and compare each factual claim and `[E#]` citation against the displayed evidence. Also try a question that the passages cannot answer; confirm the model acknowledges insufficient evidence. Check that changing the submitted question clears the old answer and that generation does not rebuild the index. Automated tests do not establish real-model grounding quality or resistance to instructions embedded in documents.

## Grouped Structured Individual-Paper Analysis (Phase 4B)

After processing/indexing PDFs, select **Paper to analyze** and click **Analyze selected paper**. The app displays exactly three fields: `research_problem`, `methodology`, and `key_results`, each with a status, value, and expandable supporting passages with their original provenance. This is an intentionally limited validation subset of the eventual larger analysis schema; other fields are not implemented.

Each field uses its own semantic query describing the concept, without requiring particular section headings. `SemanticRetriever.query_paper` applies a Chroma `paper_id` metadata filter before selecting up to five passages. Interactive QA continues using the existing global `query` method and its unchanged Top-K behavior. Both paths share the existing transient index and embedder.

Each targeted retrieval still requests up to five results, but grouped analysis selects at most six unique chunks. It first cycles through the three fields, allowing up to two distinct candidates per field in rank order (distance breaks ties), then fills remaining slots by rank, distance, and field order. Duplicate `chunk_id`s occupy one slot; when retrieved by multiple fields, the best-ranked occurrence supplies the retained original result. A prompt-size budget can select fewer than six.

Evidence IDs are assigned deterministically only after selection is final. One Ollama generation request receives those IDs, the selected passage text, definitions for all three fields, and the structured JSON schema. Pydantic validates each field independently; application code resolves supplied IDs to the original `RetrievalResult` objects and verifies selected-paper scope. Model-generated provenance and extra fields are rejected. Evidence IDs are local to one analysis and are not shared with interactive QA.

Before assigning IDs, ScholarLens omits lower-priority candidates that would push the request over budget. The estimate includes serialized instructions, evidence and request framing at six UTF-8 bytes per token, plus the structured schema at 32 bytes per token; it adds 25% headroom. This is a conservative approximation informed by local Qwen request logs, not an exact tokenizer. The final request is checked again against the default 2,000-token prompt budget before transport. If the budget cannot include at least one passage from every field that retrieved evidence, analysis fails visibly rather than favoring only earlier fields or being sent for Ollama to truncate. Grouped generation retains Ollama's existing output behavior; `num_ctx` is not changed. The portable limits can be adjusted with `SCHOLARLENS_ANALYSIS_MAX_EVIDENCE` and `SCHOLARLENS_ANALYSIS_PROMPT_BUDGET` if the local Ollama context differs.

- `SUPPORTED` (`supported` in JSON) requires a nonempty value and supplied evidence IDs.
- `INSUFFICIENT_EVIDENCE` (`insufficient_evidence` in JSON) requires `value: null` and no evidence IDs. Empty retrieval produces this result without calling Ollama.
- Malformed JSON, missing fields, invalid statuses, unknown IDs, contradictory values/statuses, and Ollama failures produce visible errors, not apparently valid analysis. A failure aborts the analysis; no partial result or old result is displayed. Retry explicitly with the analysis button.

Analysis uses the same model configuration and `/no_think` behavior as Phase 3. Connection failures, timeouts, and HTTP errors are reported separately; only recognized Ollama infrastructure error details are shown, never raw prompts or arbitrary response bodies. The UI reports retrieval/evidence preparation, generation, and total elapsed time for development comparison; results are not persisted as telemetry. Results remain in session state across ordinary reruns, and clear when selecting another paper or changing/rebuilding the index. There is no confidence score, additional status, comparative synthesis, or verification subsystem. A valid schema and valid citations do **not** prove that a claim is supported: users must inspect the passages. Insufficient evidence refers to the retrieved passages, not necessarily the entire paper. Retrieval can miss relevant passages. Grouping reduces the three sequential generation calls to one, though local generation can still be slow.

### Manual Phase 4B Check

Index the two real PDFs together and analyze each in turn. Check all three fields against the actual source passages and pages, verify each cited passage belongs to the selected paper and actually supports that field, and check that missing information produces an insufficient field without a substantive value. Observe that all three targeted searches feed one grouped generation request, and inspect the timing display. Check that paper selection and index rebuilds clear previous analysis, and that global retrieval and answer generation still work. Automated tests use fake responses and do not establish live extraction quality.

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
