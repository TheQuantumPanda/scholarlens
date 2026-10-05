# ScholarLens

ScholarLens is an evidence-grounded multi-paper research synthesis system. The current implementation covers PDF ingestion and chunk inspection, semantic retrieval, evidence-grounded Q&A, grouped structured individual-paper analysis, a descriptive comparative synthesis matrix, and paper-aware cross-paper Q&A. Evidence remains independently inspectable.

## Status

ScholarLens currently provides a Streamlit app that accepts multiple text-based PDFs, extracts page text with PyMuPDF, creates page-bounded chunks, embeds them with `BAAI/bge-small-en-v1.5`, and indexes them in a transient Chroma collection. A user can submit a semantic query and directly inspect the ranked Top-K retrieved passages with their provenance and cosine distance.

After inspecting retrieval, a user can separately generate an answer from exactly those retrieved chunks using Groq or local Ollama. A user can also select one indexed paper for structured extraction of 11 analysis fields in three groups, compare analyses for two to five selected papers, or explicitly ask a question across a selected subset of papers. Consensus/disagreement analysis and evidence verification are not implemented.

## Prerequisites

- Python 3.11
- `uv`
- For Groq generation: a Groq API key. For local generation: a running Ollama server with `qwen3:4b` already installed. Retrieval works independently of either provider. ScholarLens never downloads models automatically.

## Setup

Sync the project environment:

```bash
uv sync
```

Copy `.env.example` to `.env` and configure the desired provider, or set the same variables in the environment that launches Streamlit. `.env` is loaded from the project root and is ignored by Git. Keep the real key private; never paste it into source code, screenshots, logs, or commits. Groq sends the selected evidence passages to Groq's hosted API. Ollama keeps inference local. Provider availability and free-tier rate limits can change; when Groq is rate-limited, wait and retry later or select Ollama.

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

Generation and analysis tests fake the HTTP boundary; UI tests fake uploads, retrieval, and HTTP. They require no running Ollama server or downloaded language/embedding models. The retrieval suite also exercises transient Chroma with fake embeddings, including paper isolation before Top-K selection. Analysis tests cover all 11 fields, structured validation failures, evidence-ID resolution, and UI invalidation when the selected paper or index changes. Cross-paper tests cover scoped Top-3 retrieval, deterministic selection and budgets, query-level IDs, partial failures, insufficient comparisons, provider-neutral generation, provenance display, and session invalidation.

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

Generation uses the last submitted retrieval question and exactly its retrieved chunks. Editing the question or Top-K requires submitting **Retrieve evidence** again before those changes apply to generation. A new retrieval or changed/rebuilt index clears the previous answer. Ordinary reruns and generation reuse the existing session index; answers are retained across reruns but no conversation history is sent to the model. Choose **Groq** or **Ollama** in the sidebar; the selected provider and model are shown near generated results.

`src/scholarlens/generation.py` formats evidence as JSON containing evidence ID, source filename, paper ID, page number, chunk ID, and source text. A separate system message requires evidence-only answers, factual-claim citations, an explicit insufficient-evidence response when needed, and treating all paper contents as untrusted data rather than instructions. Empty retrieval produces a local insufficient-evidence response without calling a provider.

The provider-neutral generation interface supports two transports. Defaults and overrides are centralized in provider configuration:

| Setting | Default | Environment override |
| --- | --- | --- |
| Provider | `groq` | `SCHOLARLENS_LLM_PROVIDER` (`groq` or `ollama`) |
| Groq model | `qwen/qwen3.8-27b` | `SCHOLARLENS_GROQ_MODEL` |
| Groq API key | none | `GROQ_API_KEY` (required for Groq) |
| Groq request timeout | 240 seconds | `SCHOLARLENS_GROQ_TIMEOUT_SECONDS` |
| Ollama model | `qwen3:4b` | `SCHOLARLENS_OLLAMA_MODEL` |
| Ollama base URL | `http://localhost:11434` | `SCHOLARLENS_OLLAMA_BASE_URL` |
| Ollama request timeout | 240 seconds | `SCHOLARLENS_OLLAMA_TIMEOUT_SECONDS` |

Groq uses the OpenAI-compatible chat-completions endpoint, `reasoning_effort="none"`, and low-variance generation settings. Its free-tier limits may apply and can change; no paid upgrade is required by ScholarLens. Rate/quota errors advise waiting or explicitly selecting Ollama. Ollama remains fully local and preserves its `/api/chat`, `/no_think`, and structured-output behavior. Both transports use the standard-library HTTP client. Connection, timeout, authentication, rate-limit, HTTP/API, and malformed-response failures are reported without exposing the key or full evidence prompt.

Local API experiments with Ollama 0.34.4 and `qwen3:4b` found that `think: false` leaked reasoning into `message.content`, even alongside `/no_think`. Using `/no_think` without the `think` field produced clean content with reasoning separately in `message.thinking`. ScholarLens therefore includes `/no_think` in its system instruction and returns only `message.content`; it does not retain or display `message.thinking` or strip `<think>` tags. This workaround reflects that tested combination, not a guarantee for all Ollama models.

### Manual Generation Check

Configure Groq in `.env` or start Ollama with the selected model installed, then launch the app and retrieve passages from real PDFs. Generate an answer and compare each factual claim and `[E#]` citation against the displayed evidence. Also try a question that the passages cannot answer; confirm the model acknowledges insufficient evidence. Check that changing the submitted question clears the old answer and that generation does not rebuild the index. Automated tests do not establish real-model grounding quality or resistance to instructions embedded in documents.

## Grouped Structured Individual-Paper Analysis (Phase 4D)

After processing/indexing PDFs, select **Paper to analyze** and click **Analyze selected paper**. The app displays 11 fields, each with a status, value, and expandable supporting passages with their original provenance:

| Group | Fields |
| --- | --- |
| Research Framing | `research_problem`, `research_question`, `research_gap`, `contributions` |
| Technical Approach | `methodology`, `dataset`, `proposed_method` |
| Evaluation & Outcomes | `evaluation_metrics`, `key_results`, `limitations`, `future_work` |

Each field uses its own semantic query describing the concept, without requiring particular section headings. `SemanticRetriever.query_paper` applies a Chroma `paper_id` metadata filter before selecting up to five passages. Interactive QA continues using the existing global `query` method and its unchanged Top-K behavior. Both paths share the existing transient index and embedder.

There are 11 targeted retrieval queries, each requesting up to five results. Evidence selection is independent for each group, with a limit of six unique chunks per group. It first cycles through fields in rank order (distance breaks ties), with a quota of one distinct candidate per field in the four-field groups and two in the three-field group under the default limit, then fills remaining slots by rank, distance, and field order. Duplicate `chunk_id`s occupy one slot within a group; when retrieved by multiple fields, the best-ranked occurrence supplies the retained original result. A prompt-size budget can select fewer than six.

Evidence IDs are assigned deterministically only after selection is final. Each nonempty group makes one provider request containing its IDs, selected passage text, field definitions, and structured JSON schema: three calls for a paper with evidence in all groups, never 11. Empty groups skip generation. Pydantic validates each field independently; application code resolves supplied IDs to the original `RetrievalResult` objects and verifies selected-paper scope. Model-generated provenance and extra fields are rejected. Evidence IDs are local to each group; `E1` in one group can identify a different passage from `E1` in another group or interactive QA.

Before assigning IDs, ScholarLens omits lower-priority candidates that would push the request over budget. It keeps selected chunks intact. The estimate includes serialized instructions, evidence and request framing at six UTF-8 bytes per token, plus the structured schema at 32 bytes per token; it adds 25% headroom. This is a heuristic informed by local Qwen request logs, not an exact tokenizer or a guarantee against server-side truncation. Each group's final request is checked again against the unchanged 2,000-token prompt budget before transport. If the budget cannot include at least one passage from every field that retrieved evidence, analysis fails visibly. Grouped generation retains Ollama's existing output behavior; `num_ctx` is not changed. The portable limits can be adjusted with `SCHOLARLENS_ANALYSIS_MAX_EVIDENCE` and `SCHOLARLENS_ANALYSIS_PROMPT_BUDGET`, but increasing the budget requires separately verifying the server's effective input capacity and output headroom.

The default budget does not guarantee room for four distinct 250-word chunks. In a synthetic technical-text check with `qwen3:4b`, one 250-word passage per field estimates 2,485 / 1,957 / 2,483 prompt tokens for Research Framing / Technical Approach / Evaluation & Outcomes respectively. The four-field groups correctly reject those sets. One 100-word passage per field estimates 1,385 / 1,132 / 1,383 and fits. These are heuristic estimates for that fixture, not measured model token counts. Real retrieval may share chunks across fields or return shorter passages; if it cannot fit sufficient coverage, reduce the configured chunk size and rebuild the index explicitly. No chunks or context settings are changed automatically.

- `SUPPORTED` (`supported` in JSON) requires a nonempty value and supplied evidence IDs.
- `INSUFFICIENT_EVIDENCE` (`insufficient_evidence` in JSON) requires `value: null` and no evidence IDs. Empty retrieval produces this result without calling Ollama.
- Malformed JSON, missing fields, invalid statuses, unknown IDs, contradictory values/statuses, and provider failures produce visible errors, not apparently valid analysis. A failure aborts the analysis; no partial result or old result is displayed. Retry explicitly with the analysis button.

Analysis uses the selected provider. Ollama retains its `/no_think` behavior; Groq uses `reasoning_effort="none"`. Connection failures, timeouts, authentication failures, rate limits, and HTTP errors are reported separately without displaying secrets, raw prompts, or arbitrary response bodies. The UI reports retrieval/evidence preparation, generation, total elapsed time, per-group generation time, and the actual generation call count; results are not persisted as telemetry. The displayed individual result clears when changing papers or rebuilding the index; valid session analysis cache entries are reused only while their inputs and provider/model remain current. A valid schema and valid citations do **not** prove that a claim is supported: users must inspect the passages. Insufficient evidence refers to the retrieved passages, not necessarily the entire paper. Retrieval can miss relevant passages. Groups run sequentially, and a later group failure discards the partial analysis; earlier calls may already have completed. Local generation can still be slow.

## Comparative Synthesis Matrix (Phase 5A)

Select two to five indexed papers in **Comparative synthesis matrix** and explicitly choose **Analyze selected papers**. Completed analyses from the current session are reused when the indexed PDF contents, chunking settings, analysis budget, provider, and model still match. Missing or stale analyses are generated only after that button is pressed; a failure for one paper does not discard successful results for the others. A matrix can be built when at least two selected analyses succeed.

The matrix is a deterministic transpose of the existing 11-field `PaperAnalysis` results. Building or showing it makes no additional retrieval, embedding, or LLM request. Each row is one existing field and each paper has a column of extracted values; insufficient fields are shown explicitly as **Insufficient evidence**. Every supported cell retains its original evidence objects and can expand to show source filename, paper ID, page, chunk ID, and passage text. Evidence IDs can repeat across papers and analysis groups, so provenance identity is taken from the application-resolved evidence object.

The matrix is descriptive: it presents extracted claims with their supporting passages, does not normalize different analyses, and does not rank papers or infer consensus/disagreement. Inspect the original evidence before relying on a cell value. Cross-paper Q&A is a separate explicit workflow; disagreement analysis remains a later capability.

## Paper-Aware Cross-Paper Q&A (Phase 5B)

1. Process and index the PDFs as usual.
2. In **Cross-paper Q&A**, select **2–5 indexed papers** in the desired order, enter a **Cross-paper question**, and click **Ask across selected papers**.
3. Inspect the answer, provider/model, per-paper candidate/retained counts, and the separate evidence viewer. Each `[E#]` shows its original filename, paper ID, page, chunk ID, paper-local retrieval rank, cosine distance, and intact passage.

This explicitly selected subset is the entire retrieval scope. The existing global Top-K retrieval and **Generate answer** workflow remain available unchanged. Both paths use the same BGE model and transient Chroma index; cross-paper retrieval reuses `SemanticRetriever.query_paper` with a metadata filter applied before Top-K. Each selected paper gets **up to Top 3 candidate chunks**, independently of other papers. Three is a cap, not a required contribution.

`cross_paper.py` preserves each paper's candidate rank, removes identical retrieval entries, and selects a bounded merged evidence pool. In each of three rounds, one candidate per paper competes in ascending cosine distance order; ties follow selected-paper order. First candidates get an opportunity before second/third candidates. Candidates that exceed the count or prompt budget are omitted intact, with no chunk truncation. If this leaves fewer than two represented papers, a bounded fallback tries candidate pairs in the same priority order, retains the first fitting pair from different papers, and fills remaining space. This prevents a large early passage from excluding a smaller two-paper pool that would fit. An empty, failed, or budget-excluded paper can contribute zero chunks. The final pool is ordered by selected paper, then local retrieval rank/distance. Query-level IDs `E1`, `E2`, … are assigned from that final order, independently of chunk IDs or analysis-group IDs. Two papers with the same local chunk ID cannot collide in this mapping.

| Cross-paper setting | Default | Environment override |
| --- | --- | --- |
| Maximum final evidence chunks | 6 | `SCHOLARLENS_CROSS_PAPER_MAX_EVIDENCE` |
| Estimated prompt token budget | 2000 | `SCHOLARLENS_CROSS_PAPER_PROMPT_BUDGET` |

Cross-paper accounting measures the fully serialized compact structured request: instructions, evidence aliases/text, query, selected-paper status, final evidence IDs, adapted response schema and transport framing. It uses six bytes/token with a 1.60 safety factor for all fields, with no schema discount. The larger of the Groq and Ollama request estimates is used during selection and final preflight so both use identical conservative accounting. The nominal budget remains 2,000; individual-analysis accounting is unchanged. This is a heuristic, not provider tokenization. Large questions are rejected and oversized passages are skipped intact. Model context limits and chunk settings are unchanged. Schema overhead can reduce the retained pool; a fixed pool that exceeds budget is rejected before transport.

**No hard relevance threshold has been calibrated.** Distance only orders candidates competing for space; it is not confidence or a semantic relevance verdict. A weak paper can still have passages retained when they fit. The single synthesis call is instructed to ignore irrelevant passages, use only supplied evidence, distinguish inference, preserve attribution, cite factual comparative claims, and abstain unless useful evidence addresses the question in at least two papers. Missing retrieval is never evidence that a paper does not discuss a topic. Document text and metadata are untrusted data, not instructions.

If fewer than two successfully retrieved papers remain in the final pool, the application returns **“The available retrieved evidence does not support a cross-paper comparison.”** without calling a provider. Retrieval failures are reported per paper without discarding successful retrievals. With at least two represented papers, generation makes one schema-constrained `generate_chat` call through the existing Groq/Ollama abstraction. When a structurally valid response contains a possible two-paper comparison, claim verification makes one additional batched call. Groq's existing single bounded 429 retry, User-Agent, and safe errors are reused; there is no new scheduler, proactive delay, or repair call. Selected evidence may be sent to Groq when Groq is active, as in existing Q&A.

Selected papers receive deterministic request-local aliases `P1`, `P2`, etc. in selected order. Only aliases, evidence IDs and intact passage text enter the evidence payload; full filenames, canonical paper IDs, pages and chunk IDs remain in the application-owned evidence snapshot. An ephemeral alias mapping resolves back to canonical paper IDs and known filenames, without changing the index, retrieval objects or stored result provenance. This prevents long filenames and IDs from inflating request metadata.

The response contains at most three aspects, each with one claim or null for every selected paper, identified by its request alias. Each claim carries evidence references and optional verbatim anchors. Strict Pydantic validation rejects extra fields; application checks reject unknown aliases, unknown or wrong-paper evidence IDs, duplicate references, missing/duplicate/unselected paper sides, inconsistent null claims, and anchors that are not exact case-sensitive substrings of their specific passage. Invalid structured output produces a safe validation error, never raw model output or a repair call.

The application renders validated claims in selected-paper order and appends resolved citations. Null sides become scoped insufficient-evidence statements. No model-generated opening summary, comparison sentence or limitations prose is accepted. At least one aspect must contain claims from two distinct papers; otherwise the exact insufficient-comparison fallback is rendered. Other one-sided aspects may be displayed as asymmetry. Rendered paper labels use application-owned filenames and canonical paper IDs, falling back to the canonical ID if the filename is unavailable. Aliases are never the only displayed paper names. The evidence viewer retains original filenames/pages/chunks.

### Claim verification (Phase 5C MVP)

After structural validation, a second schema-constrained request reviews all non-null claims together against only their cited same-paper passages. The verifier returns a status and short reason per claim; it cannot supply replacement claims, citations, paper IDs, or evidence. Application checks reject missing, duplicate, or unknown claim keys and malformed results. Supported claims render normally. Unsupported claims are withheld behind a fixed warning; insufficient claims use the existing scoped insufficiency statement. If fewer than two claims in every aspect remain supported, the existing insufficient-comparison response is shown. A verifier failure withholds the answer and shows **“Could not verify this answer.”** The retained evidence viewer stays available.

The verifier is another model judgment, not a proof of entailment. It may miss unsupported paraphrases or reject supported claims. Automated tests mock provider judgments; real-model verification quality and resistance to instructions in documents remain unverified. No repair or rewrite call is made.

Ordinary reruns preserve a completed answer without retrieval or generation. Changing selection/order, question, index/upload/chunk settings, provider/model, or cross-paper budget clears the answer and its evidence snapshot. Generation failures keep the prepared evidence viewer available while removing the old answer. Cross-paper Q&A does not use or alter the comparative matrix or analysis cache. Consensus/disagreement classification, contradiction detection, ranking, recommendations, and retrieval evaluation remain later work.

### Manual Phase 5B Validation

Launch with `uv run streamlit run src/scholarlens/app.py`. Upload the current brain-tumor EEG/deep-learning PDF and blockchain malicious-text PDF, process/index them with the existing defaults, and select both in **Cross-paper Q&A**. Check the active provider/model first. Run only the following three generation requests initially; no matrix/individual analyses are needed.

1. **Test A:** Ask “How do the methodologies of the two papers differ?” Verify separate per-paper candidate counts (at most three), useful retained passages from both where available, citations to both, and correct source/page/chunk attribution. Check every comparative claim against its passage.
2. **Test B:** Ask “How does the blockchain-based system detect malicious content?” Inspect the blockchain evidence. Brain-tumor passages may still be retained because there is no semantic relevance cutoff; they must not be treated as relevant merely because retrieval returned them. If only the blockchain paper provides useful evidence, expect the insufficient-comparison response, not an invented comparison.
3. **Test C:** Ask “What caused the 2008 financial crisis?” Expect insufficient comparison evidence and no outside-knowledge answer. Treat a substantive answer as a grounding failure even if its IDs exist.
4. **Test D:** For each pool, verify every displayed `E#` is unique within that query and resolves to the original filename, paper ID, page, chunk ID, and full passage. IDs are allowed to be reused in a different query or the separate global Q&A panel.
5. **Test E:** Trigger an ordinary rerun and confirm the current answer persists without another call. Change the cross-paper question, paper order/selection, provider/model, or uploaded/indexed content and confirm the old answer and pool disappear. Reindexing also clears them. Re-entering an earlier question does not restore an old answer automatically.

Automated tests fake generation and do not establish real-model grounding or instruction resistance. No deterministic runtime relevance rule guarantees Tests B/C: the model must correctly abstain when retrieved passages are off topic. Do not introduce a distance cutoff in response to a single manual run.

### Manual Phase 4D Check

Index two real PDFs together and analyze each in turn. Check all 11 fields against the actual source passages and pages, verify each cited passage belongs to the selected paper and supports that field, and check that missing information produces an insufficient field without a substantive value. Distinguish methodology from proposed method and evaluation metrics from result values. Check that research questions, gaps, limitations, and future work are not invented. Observe 11 targeted searches and three generation requests when all groups have evidence; check group-local evidence IDs and the timing display. Exercise a capacity rejection and confirm no truncated or stale result is displayed. Check that paper selection and index rebuilds clear previous analysis, and that global retrieval and answer generation still work. Automated tests use fake responses and do not establish live extraction quality.

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
- Global Q&A and individual analysis still rely on instructed grounding. Cross-paper claims receive an additional model verification judgment, which can still miss unsupported claims or follow malicious document instructions.
- Large global Top-K selections may exceed the model's context capacity. Cross-paper Q&A and structured analysis use heuristic prompt budgets; the global Q&A baseline does not. No path detects server-side context truncation.
- Generation is synchronous and may be slow on local hardware; no streaming or conversation memory.
- No cross-paper consensus/disagreement classification or deterministic entailment proof. Cross-paper verification remains model-dependent, and relevance thresholds remain uncalibrated.
