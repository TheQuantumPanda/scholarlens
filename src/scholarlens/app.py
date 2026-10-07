from __future__ import annotations

from hashlib import sha256
from typing import Any

import streamlit as st

from scholarlens.analysis import AnalysisConfig, EXTRACTION_GROUPS, analyze_paper
from scholarlens.chunking import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, chunk_pages
from scholarlens.comparison import (
    ComparisonAnalysisRun,
    ComparisonMatrix,
    PaperAnalysisCacheKey,
    analyze_selected_papers,
    build_comparison_matrix,
    make_analysis_cache_key,
)
from scholarlens.cross_paper import (
    CrossPaperConfig,
    CrossPaperGenerationResult,
    generate_cross_paper_answer,
    retrieve_cross_paper_evidence,
)
from scholarlens.evidence_view import (
    ClaimEvidenceView,
    prepare_analysis_claim,
    prepare_cross_paper_claims,
    prepare_synthesis_claims,
)
from scholarlens.embeddings import SentenceTransformerEmbedder
from scholarlens.generation import (
    DEFAULT_LLM_PROVIDER,
    GenerationError,
    LLMConfig,
    configured_provider,
    generate_answer,
    get_llm_config,
)
from scholarlens.models import PageText, PaperAnalysis, RetrievalResult, TextChunk
from scholarlens.pdf import default_paper_id, extract_pdf_pages
from scholarlens.retrieval import SemanticRetriever
from scholarlens.synthesis import Classification, SynthesisResult, synthesize


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
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_result_key", None)
    st.session_state.setdefault("comparison_analysis_cache", {})
    st.session_state.setdefault("comparison_cache_context", None)
    st.session_state.setdefault("comparison_analysis_run", None)
    st.session_state.setdefault("comparison_matrix", None)
    st.session_state.setdefault("comparison_matrix_paper_ids", None)
    st.session_state.setdefault("index_signature", None)
    st.session_state.setdefault("cross_paper_pool", None)
    st.session_state.setdefault("cross_paper_result", None)
    st.session_state.setdefault("cross_paper_context", None)
    st.session_state.setdefault("synthesis_context", None)
    st.session_state.setdefault("synthesis_result", None)


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
    st.session_state.analysis_result = None
    st.session_state.analysis_result_key = None
    st.session_state.comparison_analysis_cache = {}
    st.session_state.comparison_cache_context = None
    st.session_state.comparison_analysis_run = None
    st.session_state.comparison_matrix = None
    st.session_state.comparison_matrix_paper_ids = None
    st.session_state.pop("analysis-paper", None)
    st.session_state.pop("comparison-paper-selection", None)
    st.session_state.cross_paper_pool = None
    st.session_state.cross_paper_result = None
    st.session_state.cross_paper_context = None
    st.session_state.synthesis_context = None
    st.session_state.synthesis_result = None
    st.session_state.pop("cross-paper-selection", None)
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


def _render_retrieval_results(
    results: list[RetrievalResult], *, key_prefix: str = "retrieved",
) -> None:
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
                key=f"{key_prefix}-{evidence_number}-{result.paper_id}-{result.chunk_id}",
            )


def _render_claim_evidence(
    view: ClaimEvidenceView, *, insufficient_message: str = "Insufficient evidence",
) -> None:
    """Show claim status and exact cited passages without interpreting paper text."""
    st.text(f"Status: {view.status}")
    if view.status == "UNSUPPORTED":
        st.warning("Withheld: the verifier did not find support for this claim.")
    elif view.status in ("INVALID_REFERENCE", "UNVERIFIED"):
        st.warning("This claim is not verified as supported.")
    elif view.status == "INSUFFICIENT_EVIDENCE":
        st.info(insufficient_message)
    if view.claim is not None:
        st.text(view.claim)
    if not view.evidence:
        st.caption("No cited evidence.")
        return
    with st.expander("View evidence"):
        for item in view.evidence:
            with st.container(border=True):
                st.text(f"Evidence ID: {item.evidence_id}")
                st.text(f"Claim status: {view.status}")
                result = item.result
                if result is None:
                    st.error("Cited evidence is missing or does not belong to this paper.")
                    continue
                st.text(f"Filename: {result.source_filename}")
                st.text(f"Paper: {result.paper_id}")
                st.text(f"Page: {result.page_number}")
                st.text(f"Chunk ID: {result.chunk_id}")
                section = getattr(result, "section", None)
                if section:
                    st.text(f"Section: {section}")
                st.text(result.text)


def _render_cross_paper_qa(
    retriever: SemanticRetriever,
    chunks: list[TextChunk],
    config: LLMConfig,
    index_signature: tuple[Any, ...],
) -> None:
    st.subheader("Cross-paper Q&A")
    papers = {chunk.paper_id: chunk.source_filename for chunk in chunks}
    selected_ids = st.multiselect(
        "Select 2–5 papers for cross-paper Q&A",
        options=list(papers),
        format_func=lambda paper_id: f"{papers[paper_id]} ({paper_id})",
        max_selections=5,
        key="cross-paper-selection",
    )
    question = st.text_input("Cross-paper question", key="cross-paper-question")
    st.caption(
        f"Provider/model: {config.provider} · {config.model}. Up to three candidate "
        "passages per selected paper; the final evidence pool is bounded. "
        "Retrieval distance is not confidence, and retained passages may be insufficient."
    )
    try:
        budget = CrossPaperConfig.from_env()
    except ValueError:
        st.session_state.cross_paper_pool = None
        st.session_state.cross_paper_result = None
        st.error("Cross-paper evidence and prompt budgets must be positive integers.")
        return
    context = (index_signature, tuple(selected_ids), question, config.provider, config.model, budget)
    if st.session_state.cross_paper_context != context:
        st.session_state.cross_paper_pool = None
        st.session_state.cross_paper_result = None
        st.session_state.cross_paper_context = context
    valid_selection = (
        2 <= len(selected_ids) <= 5
        and len(set(selected_ids)) == len(selected_ids)
        and all(paper_id in papers for paper_id in selected_ids)
    )
    if not valid_selection:
        st.caption("Choose two to five distinct indexed papers.")
    if st.button(
        "Ask across selected papers", key="ask-cross-paper",
        disabled=not valid_selection or not question.strip(),
    ):
        st.session_state.cross_paper_pool = None
        st.session_state.cross_paper_result = None
        # Validate again on submission; widget limits are only UI guardrails.
        if not valid_selection:
            st.error("Choose two to five distinct indexed papers.")
            return
        try:
            with st.spinner("Retrieving selected papers and answering from bounded evidence..."):
                pool = retrieve_cross_paper_evidence(
                    question, selected_ids, retriever, model=config.model, budget=budget,
                )
                st.session_state.cross_paper_pool = pool
                st.session_state.cross_paper_result = generate_cross_paper_answer(
                    pool, config, budget=budget,
                    on_rate_limit=lambda delay: st.toast(
                        f"Groq rate limit reached. Retrying in {delay:.0f} s…", icon="⏳",
                    ),
                )
        except GenerationError as exc:
            st.error(str(exc))
        except ValueError:
            st.error("Invalid cross-paper question or paper selection.")

    generated = st.session_state.cross_paper_result
    if generated is not None:
        st.text(f"Cross-paper answer for: {generated.question}")
        st.caption(
            f"Generated by: {generated.provider or 'No provider call'} · "
            f"{generated.model or 'No model call (insufficient comparison evidence)'}"
        )
        st.markdown(generated.answer)
        if isinstance(generated, CrossPaperGenerationResult):
            claim_views = prepare_cross_paper_claims(
                generated, st.session_state.cross_paper_pool,
            )
            if claim_views:
                st.caption("Generated paper claims and their cited passages")
                for view in claim_views:
                    with st.container(border=True):
                        st.text(f"{view.aspect} · Paper {view.paper_id}")
                        _render_claim_evidence(view)
    pool = st.session_state.cross_paper_pool
    if pool is not None:
        for paper in pool.papers:
            name = f"{papers[paper.paper_id]} ({paper.paper_id})"
            count = sum(r.paper_id == paper.paper_id for r in pool.evidence)
            if paper.retrieval_failed:
                st.warning(f"Retrieval failed for {name}; other selected papers were processed.")
            elif not paper.results:
                st.info(f"No candidate evidence was retrieved from {name}.")
            elif paper.filtered_candidate_count == len(paper.results):
                st.info(f"All retrieved candidates for {name} were excluded as obvious captions or reference material.")
            elif not count:
                st.info(f"No evidence from {name} was retained within the evidence budget.")
            st.text(
                f"{name}: {len(paper.results)} candidates, "
                f"{paper.filtered_candidate_count} junk passages filtered, {count} retained"
            )
        st.caption(
            "E# IDs apply only to this cross-paper answer. Missing evidence does not "
            "establish that a paper never discusses the topic. Check citations below."
        )
        _render_retrieval_results(list(pool.evidence), key_prefix="cross-paper-evidence")


def _field_display_name(field_name: str) -> str:
    """Convert snake_case field name to readable display name."""
    return field_name.replace("_", " ").capitalize()


def _render_consensus_disagreement(
    retriever: SemanticRetriever,
    chunks: list[TextChunk],
    config: LLMConfig,
    index_signature: tuple[Any, ...],
) -> None:
    st.subheader("Consensus & Disagreement")
    papers = {chunk.paper_id: chunk.source_filename for chunk in chunks}
    selected = st.multiselect(
        "Select 2–5 papers for synthesis", list(papers),
        format_func=lambda paper_id: f"{papers[paper_id]} ({paper_id})",
        max_selections=5, key="synthesis-paper-selection",
    )
    topic = st.text_input("Topic or finding to compare", key="synthesis-topic")
    st.caption("Findings describe the retained evidence, not the full literature. Different study contexts may limit comparability.")
    try:
        budget = CrossPaperConfig.from_env()
    except ValueError:
        st.error("Cross-paper evidence and prompt budgets must be positive integers.")
        return
    context = (index_signature, tuple(selected), topic, config.provider, config.model, budget)
    if st.session_state.synthesis_context != context:
        st.session_state.synthesis_result = None
        st.session_state.synthesis_context = context
    valid = 2 <= len(selected) <= 5 and len(set(selected)) == len(selected) and all(p in papers for p in selected)
    if st.button("Analyze consensus & disagreement", key="analyze-synthesis",
                 disabled=not valid or not topic.strip()):
        st.session_state.synthesis_result = None
        try:
            with st.spinner("Reviewing bounded evidence across selected papers..."):
                pool = retrieve_cross_paper_evidence(topic, selected, retriever,
                                                     model=config.model, budget=budget)
                st.session_state.synthesis_result = synthesize(
                    pool, config, budget=budget,
                    on_rate_limit=lambda delay: st.toast(
                        f"Groq rate limit reached. Retrying in {delay:.0f} s…", icon="⏳",
                    ),
                )
        except GenerationError as exc:
            st.error(str(exc))
    result: SynthesisResult | None = st.session_state.synthesis_result
    if result is None:
        return
    labels = {
        Classification.CONSENSUS: "Consensus",
        Classification.POTENTIAL_DISAGREEMENT: "Potential disagreement",
        Classification.INSUFFICIENT_EVIDENCE: "Insufficient evidence",
    }
    st.caption(f"Generated by: {result.provider or 'No provider call'} · {result.model or 'No model call'}")
    for finding in result.findings:
        with st.container(border=True):
            st.text(f"{labels[finding.classification]} · {finding.aspect}")
            st.text(finding.summary)
            if finding.context_note:
                st.caption(finding.context_note)
            for position, view in zip(finding.positions, prepare_synthesis_claims(finding, result), strict=True):
                st.text(f"{position.source_filename or position.paper_id} ({position.paper_id})")
                _render_claim_evidence(view)


def _analysis_cache_context(index_signature: tuple[Any, ...], config: LLMConfig) -> tuple[Any, ...]:
    return (index_signature, config.provider, config.model, AnalysisConfig.from_env())


def _analysis_key(
    paper_id: str,
    index_signature: tuple[Any, ...],
    config: LLMConfig,
    analysis_config: AnalysisConfig,
) -> PaperAnalysisCacheKey:
    return make_analysis_cache_key(paper_id, index_signature, config, analysis_config)


def _render_paper_analysis(
    retriever: SemanticRetriever,
    chunks: list[TextChunk],
    config: LLMConfig,
    index_signature: tuple[Any, ...],
) -> None:
    st.subheader("Individual-paper analysis (Phase 4D)")
    papers = {chunk.paper_id: chunk.source_filename for chunk in chunks}
    selected_paper = st.selectbox(
        "Paper to analyze",
        list(papers),
        format_func=lambda paper_id: f"{papers[paper_id]} ({paper_id})",
        key="analysis-paper",
    )
    if selected_paper not in papers:
        st.session_state.analysis_result = None
        st.session_state.analysis_result_key = None
        st.error("Select an indexed paper.")
        return
    analysis_config = AnalysisConfig.from_env()
    cache_key = _analysis_key(selected_paper, index_signature, config, analysis_config)
    analysis = st.session_state.analysis_result
    if analysis is not None and (
        analysis.paper_id != selected_paper
        or st.session_state.analysis_result_key != cache_key
    ):
        st.session_state.analysis_result = None
        st.session_state.analysis_result_key = None

    st.caption(
        "Extract 11 structured fields from the selected paper using three semantic "
        "extraction groups. Check each extracted claim against its supporting passages."
    )
    if st.button("Analyze selected paper", key="analyze-paper"):
        st.session_state.analysis_result = None
        st.session_state.analysis_result_key = None
        try:
            with st.spinner("Retrieving evidence and extracting 11 fields (3 groups)..."):
                result = analyze_paper(
                    selected_paper,
                    retriever,
                    config=config,
                    analysis_config=analysis_config,
                )
            st.session_state.analysis_result = result
            st.session_state.analysis_result_key = cache_key
            st.session_state.comparison_analysis_cache[cache_key] = result
        except Exception as exc:
            st.error(f"Could not analyze paper: {exc}")

    analysis = st.session_state.analysis_result
    if analysis is None:
        return
    st.text(f"Analysis for: {papers[analysis.paper_id]} ({analysis.paper_id})")
    if analysis.timing is not None:
        group_detail = ""
        if analysis.timing.group_timings:
            parts = [
                f"{gt.group_name}: {gt.generation_seconds:.1f}s"
                for gt in analysis.timing.group_timings
            ]
            group_detail = f"  ·  Group generation: {', '.join(parts)}"
        generation_calls = sum(gt.generation_calls for gt in analysis.timing.group_timings)
        st.caption(
            f"Generated by: {analysis.provider or 'No provider call'} · "
            f"{analysis.model or 'No model call (no evidence)'}  ·  "
            f"Retrieval/evidence: {analysis.timing.retrieval_seconds:.1f}s · "
            f"Generation: {analysis.timing.generation_seconds:.1f}s · "
            f"Total: {analysis.timing.total_seconds:.1f}s · "
            f"{generation_calls} generation call{'s' if generation_calls != 1 else ''}"
            f"{group_detail}"
        )
    else:
        st.caption(
            f"Generated by: {analysis.provider or 'No provider call'} · "
            f"{analysis.model or 'No model call (no evidence)'}"
        )
    st.caption("Evidence IDs are shared across fields within each group, not across groups.")

    for group in EXTRACTION_GROUPS:
        st.markdown(f"#### {group.display_name}")
        for definition in group.fields:
            field = getattr(analysis, definition.name)
            with st.container(border=True):
                st.markdown(f"**{_field_display_name(definition.name)}**")
                _render_claim_evidence(
                    prepare_analysis_claim(field),
                    insufficient_message="The retrieved passages do not establish this field.",
                )


def _render_comparison_matrix(
    retriever: SemanticRetriever,
    chunks: list[TextChunk],
    config: LLMConfig,
    index_signature: tuple[Any, ...],
) -> None:
    st.subheader("Comparative synthesis matrix")
    paper_names: dict[str, str] = {}
    for chunk in chunks:
        paper_names.setdefault(chunk.paper_id, chunk.source_filename)
    paper_ids = list(paper_names)
    if len(paper_ids) < 2:
        st.info("Index at least two papers to compare their structured analyses.")
        return

    selected_ids = st.multiselect(
        "Select 2–5 papers for comparison",
        options=paper_ids,
        format_func=lambda paper_id: f"{paper_names[paper_id]} ({paper_id})",
        max_selections=5,
        key="comparison-paper-selection",
    )
    if len(selected_ids) > 5:
        st.warning("Select no more than five papers.")
    valid_selection = 2 <= len(selected_ids) <= 5
    if not valid_selection and len(selected_ids) < 2:
        st.caption("Choose at least two indexed papers.")

    analysis_config = AnalysisConfig.from_env()
    cache_keys = {
        paper_id: _analysis_key(paper_id, index_signature, config, analysis_config)
        for paper_id in selected_ids
    }
    run: ComparisonAnalysisRun | None = st.session_state.comparison_analysis_run
    run_matches_selection = run is not None and run.selected_paper_ids == tuple(selected_ids)

    with st.container(horizontal=True):
        analyze_clicked = st.button(
            "Analyze selected papers",
            key="analyze-comparison-papers",
            disabled=not valid_selection,
        )
        build_clicked = st.button(
            "Build / show comparison matrix",
            key="build-comparison-matrix",
            disabled=not valid_selection,
        )

    if analyze_clicked:
        with st.status("Analyzing selected papers…", expanded=True) as progress:
            result = analyze_selected_papers(
                selected_ids,
                st.session_state.comparison_analysis_cache,
                cache_keys,
                lambda paper_id: analyze_paper(
                    paper_id,
                    retriever,
                    config=config,
                    analysis_config=analysis_config,
                ),
                on_progress=lambda index, count, paper_id: progress.update(
                    label=f"Paper {index} of {count}: {paper_names[paper_id]}"
                ),
            )
            st.session_state.comparison_analysis_run = result
            st.session_state.comparison_matrix = None
            st.session_state.comparison_matrix_paper_ids = None
            progress.update(label="Paper analyses complete", state="complete", expanded=False)
        run = result
        run_matches_selection = True

    if run_matches_selection and run is not None:
        if run.reused_paper_ids:
            st.success("Reused completed analyses: " + ", ".join(
                f"{paper_names[paper_id]} ({paper_id})" for paper_id in run.reused_paper_ids
            ))
        if run.analyzed_paper_ids:
            st.success("Analyzed in this run: " + ", ".join(
                f"{paper_names[paper_id]} ({paper_id})" for paper_id in run.analyzed_paper_ids
            ))
        for failure in run.failures:
            st.error(f"Analysis failed for {paper_names[failure.paper_id]} ({failure.paper_id}): {failure.reason}")
        if len(run.analyses) < 2:
            st.info("At least two selected papers must analyze successfully before building a matrix.")

    if build_clicked and run_matches_selection and run is not None and len(run.analyses) >= 2:
        st.session_state.comparison_matrix = build_comparison_matrix(
            run.analyses,
            paper_names,
        )
        st.session_state.comparison_matrix_paper_ids = run.selected_paper_ids
    elif build_clicked:
        st.info("Analyze the currently selected papers successfully before building the matrix.")

    matrix: ComparisonMatrix | None = st.session_state.comparison_matrix
    if matrix is None or st.session_state.comparison_matrix_paper_ids != tuple(selected_ids):
        return

    st.caption(
        "Descriptive comparison of extracted claims. Building this matrix makes no additional LLM call; "
        "inspect each cell’s original passages before relying on its value."
    )
    for field_name in matrix.fields:
        st.markdown(f"#### {_field_display_name(field_name)}")
        for paper in matrix.papers:
            cell = matrix.get_cell(paper.paper_id, field_name)
            with st.container(border=True):
                st.markdown(f"**{paper.source_filename} ({paper.paper_id})**")
                _render_claim_evidence(prepare_analysis_claim(cell))


def main() -> None:
    st.set_page_config(page_title="ScholarLens Evidence Retriever", layout="wide")
    st.title("ScholarLens Evidence Retriever")
    st.caption("Inspect evidence, generate answers, or analyze one paper's 11 structured fields.")
    _initialize_state()

    try:
        default_provider = configured_provider()
    except GenerationError as exc:
        st.error(str(exc))
        default_provider = DEFAULT_LLM_PROVIDER

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    with st.sidebar:
        provider_names = ("groq", "ollama")
        selected_provider = st.selectbox(
            "LLM provider",
            provider_names,
            index=provider_names.index(default_provider),
            format_func=str.title,
            key="llm-provider",
        )
        config = get_llm_config(selected_provider)
        st.caption(f"Active model: {config.model}")
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
            st.caption(
                f"Provider/model: {config.provider} · {config.model}. Generate using exactly the evidence above and its "
                "submitted question. Submit Retrieve evidence again to use a different question "
                "or Top-K. Check the answer's citations against the evidence."
            )
            if st.button("Generate answer", key="generate-answer"):
                st.session_state.generation_result = None
                try:
                    with st.spinner("Generating an answer from retrieved evidence..."):
                        st.session_state.generation_result = generate_answer(
                            st.session_state.retrieval_question, results, config=config,
                            on_rate_limit=lambda delay: st.toast(
                                f"Groq rate limit reached. Retrying in {delay:.0f} s…",
                                icon="⏳",
                            ),
                        )
                except GenerationError as exc:
                    st.error(str(exc))

            generated = st.session_state.generation_result
            if generated is not None:
                st.text(f"Answer for: {generated.question}")
                st.caption(
                    f"Generated by: {generated.provider or 'No provider call'} · "
                    f"{generated.model or 'No model call (no evidence)'}"
                )
                st.markdown(generated.answer)

    if retriever is not None:
        analysis_index_signature = st.session_state.index_signature
        if analysis_index_signature is not None:
            _render_cross_paper_qa(retriever, chunks, config, analysis_index_signature)
            context = _analysis_cache_context(analysis_index_signature, config)
            if st.session_state.comparison_cache_context != context:
                st.session_state.comparison_analysis_cache = {}
                st.session_state.comparison_analysis_run = None
                st.session_state.comparison_matrix = None
                st.session_state.comparison_matrix_paper_ids = None
                st.session_state.comparison_cache_context = context
            _render_paper_analysis(retriever, chunks, config, analysis_index_signature)
            _render_comparison_matrix(retriever, chunks, config, analysis_index_signature)
            _render_consensus_disagreement(retriever, chunks, config, analysis_index_signature)

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
