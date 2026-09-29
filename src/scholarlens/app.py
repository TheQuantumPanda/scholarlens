from __future__ import annotations

import streamlit as st

from scholarlens.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, chunk_pages
from scholarlens.pdf import default_paper_id, extract_pdf_pages


def _paper_id_for_upload(filename: str, index: int) -> str:
    return f"{default_paper_id(filename)}-{index}"


def main() -> None:
    st.set_page_config(page_title="ScholarLens Chunk Inspector", layout="wide")
    st.title("ScholarLens Chunk Inspector")

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

    if not uploaded_files:
        st.info("Upload one or more PDFs to inspect extracted, provenance-preserving chunks.")
        return

    all_pages = []
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

    chunks = chunk_pages(all_pages, max_words=int(max_words), overlap_words=int(overlap_words))
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

    st.subheader("Chunks")
    if not chunks:
        st.warning("No extractable text was found in the uploaded PDFs.")
        return

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
