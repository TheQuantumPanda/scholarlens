from __future__ import annotations

from hashlib import sha256
from typing import Any

import streamlit as st

from scholarlens.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, chunk_pages
from scholarlens.embeddings import SentenceTransformerEmbedder
from scholarlens.generation import GenerationError, OllamaConfig, generate_answer
from scholarlens.models import PageText, RetrievalResult, TextChunk
from scholarlens.pdf import default_paper_id, extract_pdf_pages
from scholarlens.retrieval import SemanticRetriever


def _paper_id_for_upload(filename: str, index: int) -> str:
    return f"{default_paper_id(filename)}-{index}"


@st.cache_resource(show_spinner=False)
def _load_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def _initialize_state() -> None:
    st.session_state.setdefault("pages", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("retrieval_results", [])
    st.session_state.setdefault("retrieval_question", None)
    st.session_state.setdefault("generation_result", None)
    st.session_state.setdefault("index_signature", None)


def _store_index(
    pages: list[PageText],
    chunks: list[TextChunk],
    retriever: SemanticRetriever | None,
    index_signature: tuple[Any, ...] | None,
) -> None:
    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.retriever = retriever
    st.session_state.retrieval_results = []
    st.session_state.retrieval_question = None
    st.session_state.generation_result = None
    st.session_state.index_signature = index_signature


def _index_signature(
    uploaded_files: list[Any],
    max_words: int,
    overlap_words: int,
) -> tuple[Any, ...]:
    uploaded_documents = tuple(
        (uploaded_file.name, sha256(uploaded_file.getvalue()).hexdigest())
        for uploaded_file in uploaded_files
    )
    return (uploaded_documents, max_words, overlap_words)


def _render_retrieval_results(results: list[RetrievalResult]) -> None:
    st.subheader("Retrieved evidence")
    for evidence_number, result in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"**[E{evidence_number}] · Rank {result.rank}**")
            st.text(f"Filename: {result.source_filename}")
            st.text(f"Paper: {result.paper_id}")
            st.text(f"Page: {result.page_number}")
            st.text(f"Chunk ID: {result.chunk_id}")
            st.text(f"Cosine distance: {result.distance:.6f} (lower is closer)")
            st.text_area(
                "Retrieved text",
                result.text,
                height=180,
                disabled=True,
                key=f"retrieved-{result.rank}-{result.chunk_id}",
            )


def main() -> None:
    st.set_page_config(page_title="ScholarLens Evidence Retriever", layout="wide")
    st.title("ScholarLens Evidence Retriever")
    st.caption("Phase 3: inspect retrieved evidence, then generate an evidence-grounded answer.")
    _initialize_state()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    with st.sidebar:
        st.header("Chunking")
        max_words = st.number_input(
            "Words per chunk",
            min_value=25,
            max_value=1000,
            value=DEFAULT_CHUNK_WORDS,
        )
        overlap_words = st.number_input(
            "Overlapping words",
            min_value=0,
            max_value=max_words - 1,
            value=min(DEFAULT_OVERLAP_WORDS, max_words - 1),
        )
        st.caption(
            "Chunking is provisional. BGE has a finite input length and may truncate "
            "long embedding inputs."
        )

    process_clicked = st.button(
        "Process and index PDFs",
        type="primary",
        icon=":material/database_upload:",
        disabled=not uploaded_files,
    )

    if not uploaded_files:
        _store_index([], [], None, None)
        st.info("Upload one or more PDFs to extract, index, and inspect retrieved evidence.")
        return

    current_signature = _index_signature(
        uploaded_files,
        int(max_words),
        int(overlap_words),
    )
    if st.session_state.index_signature not in (None, current_signature):
        _store_index([], [], None, None)

    if process_clicked:
        all_pages: list[PageText] = []
        for index, uploaded_file in enumerate(uploaded_files, start=1):
            paper_id = _paper_id_for_upload(uploaded_file.name, index)
            try:
                pages = extract_pdf_pages(
                    uploaded_file.getvalue(),
                    source_filename=uploaded_file.name,
                    paper_id=paper_id,
                )
            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")
                continue

            all_pages.extend(pages)

        chunks = chunk_pages(
            all_pages,
            max_words=int(max_words),
            overlap_words=int(overlap_words),
        )
        if chunks:
            try:
                with st.spinner("Loading the embedding model and indexing chunks..."):
                    retriever = SemanticRetriever(_load_embedder())
                    retriever.index(chunks)
            except Exception as exc:
                _store_index(all_pages, chunks, None, current_signature)
                st.error(f"Could not build the semantic index: {exc}")
            else:
                _store_index(all_pages, chunks, retriever, current_signature)
                st.success(f"Indexed {len(chunks)} chunks for semantic retrieval.")
        else:
            _store_index(all_pages, chunks, None, current_signature)

    all_pages = st.session_state.pages
    chunks = st.session_state.chunks
    retriever = st.session_state.retriever
    if not all_pages and not chunks:
        st.info("Select the PDFs and click Process and index PDFs.")
        return

    pages_without_text = [page for page in all_pages if not page.text.split()]

    st.subheader("Summary")
    col_pages, col_empty, col_chunks = st.columns(3)
    col_pages.metric("Pages extracted", len(all_pages))
    col_empty.metric("Pages without text", len(pages_without_text))
    col_chunks.metric("Chunks", len(chunks))

    if pages_without_text:
        with st.expander("Pages without extractable text"):
            for page in pages_without_text:
                st.write(
                    {
                        "paper_id": page.paper_id,
                        "source_filename": page.source_filename,
                        "page_number": page.page_number,
                    }
                )

    if not chunks:
        st.warning("No extractable text was found in the uploaded PDFs.")
        return

    if retriever is not None:
        st.subheader("Semantic retrieval")
        with st.form("semantic-query", border=False):
            query_text = st.text_input(
                "Query",
                placeholder="Enter a question or concept to retrieve relevant evidence",
            )
            # Conservative Phase 3 guard for local qwen3:4b's limited context;
            # proper context budgeting will be evaluated later.
            top_k = st.number_input(
                "Top-K results",
                min_value=1,
                max_value=min(5, len(chunks)),
                value=min(5, len(chunks)),
            )
            query_submitted = st.form_submit_button(
                "Retrieve evidence",
                icon=":material/search:",
            )

        if query_submitted:
            st.session_state.retrieval_results = []
            st.session_state.retrieval_question = None
            st.session_state.generation_result = None
            if not query_text.strip():
                st.warning("Enter a query before retrieving evidence.")
            else:
                try:
                    st.session_state.retrieval_results = retriever.query(
                        query_text,
                        top_k=int(top_k),
                    )
                    st.session_state.retrieval_question = query_text
                except Exception as exc:
                    st.error(f"Could not retrieve evidence: {exc}")

        if st.session_state.retrieval_question is not None:
            st.text(f"Retrieved for: {st.session_state.retrieval_question}")
            results = st.session_state.retrieval_results
            if results:
                _render_retrieval_results(results)
            else:
                st.info("No evidence was retrieved for this question.")

            st.subheader("Generated answer")
            config = OllamaConfig.from_env()
            st.caption(
                f"Model: {config.model}. Generate using exactly the evidence above and its "
                "submitted question. Submit Retrieve evidence again to use a different question "
                "or Top-K. Check the answer's citations against the evidence."
            )
            if st.button("Generate answer", key="generate-answer"):
                st.session_state.generation_result = None
                try:
                    with st.spinner("Generating an answer from retrieved evidence..."):
                        st.session_state.generation_result = generate_answer(
                            st.session_state.retrieval_question, results, config=config
                        )
                except GenerationError as exc:
                    st.error(str(exc))

            generated = st.session_state.generation_result
            if generated is not None:
                st.text(f"Answer for: {generated.question}")
                st.caption(f"Generated by: {generated.model or 'No model call (no evidence)'}")
                st.markdown(generated.answer)

    st.subheader("Indexed chunks")
    for chunk in chunks:
        with st.expander(f"{chunk.chunk_id} - {chunk.source_filename}, page {chunk.page_number}"):
            st.write(
                {
                    "paper_id": chunk.paper_id,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                }
            )
            st.text_area("Chunk text", chunk.text, height=180, disabled=True, key=chunk.chunk_id)


if __name__ == "__main__":
    main()
