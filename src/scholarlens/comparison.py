"""Deterministic, evidence-preserving comparison of completed paper analyses."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scholarlens.analysis import ALL_FIELD_NAMES, AnalysisConfig, FIELD_DEFINITIONS
from scholarlens.generation import LLMConfig
from scholarlens.models import AnalysisEvidence, AnalysisStatus, PaperAnalysis


@dataclass(frozen=True)
class ComparisonPaper:
    paper_id: str
    source_filename: str


@dataclass(frozen=True)
class ComparisonCell:
    paper_id: str
    source_filename: str
    field_name: str
    status: AnalysisStatus
    value: str | None
    evidence: tuple[AnalysisEvidence, ...]


@dataclass(frozen=True)
class ComparisonMatrix:
    papers: tuple[ComparisonPaper, ...]
    fields: tuple[str, ...]
    cells: tuple[ComparisonCell, ...]

    def get_cell(self, paper_id: str, field_name: str) -> ComparisonCell:
        for cell in self.cells:
            if cell.paper_id == paper_id and cell.field_name == field_name:
                return cell
        raise KeyError((paper_id, field_name))


@dataclass(frozen=True)
class PaperAnalysisCacheKey:
    paper_id: str
    index_signature: tuple[Any, ...]
    provider: str
    model: str
    analysis_config: AnalysisConfig


@dataclass(frozen=True)
class PaperAnalysisFailure:
    paper_id: str
    reason: str


@dataclass(frozen=True)
class ComparisonAnalysisRun:
    selected_paper_ids: tuple[str, ...]
    reused_paper_ids: tuple[str, ...]
    analyzed_paper_ids: tuple[str, ...]
    failures: tuple[PaperAnalysisFailure, ...]
    analyses: tuple[PaperAnalysis, ...]


def make_analysis_cache_key(
    paper_id: str,
    index_signature: tuple[Any, ...],
    config: LLMConfig,
    analysis_config: AnalysisConfig,
) -> PaperAnalysisCacheKey:
    """Include every current input/configuration that can change extraction."""
    return PaperAnalysisCacheKey(
        paper_id=paper_id,
        index_signature=index_signature,
        provider=config.provider,
        model=config.model,
        analysis_config=analysis_config,
    )


def build_comparison_matrix(
    analyses: Sequence[PaperAnalysis],
    source_filenames: Mapping[str, str],
) -> ComparisonMatrix:
    """Transpose completed analyses without retrieval, generation, or mutation."""
    if not 2 <= len(analyses) <= 5:
        raise ValueError("A comparison requires between 2 and 5 successful analyses")
    paper_ids = tuple(analysis.paper_id for analysis in analyses)
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Comparison analyses must belong to distinct papers")

    papers: list[ComparisonPaper] = []
    cells: list[ComparisonCell] = []
    ordered_fields = tuple(definition.name for definition in FIELD_DEFINITIONS)
    if set(ordered_fields) != ALL_FIELD_NAMES:
        raise ValueError("Analysis field definitions are inconsistent")

    for analysis in analyses:
        filename = source_filenames.get(analysis.paper_id)
        if not filename:
            raise ValueError(f"No indexed source filename for paper {analysis.paper_id}")
        papers.append(ComparisonPaper(analysis.paper_id, filename))
        for field_name in ordered_fields:
            field = getattr(analysis, field_name)
            for evidence in field.evidence:
                result = evidence.result
                if result.paper_id != analysis.paper_id:
                    raise ValueError(
                        f"{field_name} evidence does not belong to paper {analysis.paper_id}"
                    )
                if result.source_filename != filename:
                    raise ValueError(
                        f"{field_name} evidence filename does not match indexed paper {analysis.paper_id}"
                    )
            cells.append(ComparisonCell(
                paper_id=analysis.paper_id,
                source_filename=filename,
                field_name=field_name,
                status=field.status,
                value=field.value,
                evidence=field.evidence,
            ))

    return ComparisonMatrix(tuple(papers), ordered_fields, tuple(cells))


def analyze_selected_papers(
    paper_ids: Sequence[str],
    cache: dict[PaperAnalysisCacheKey, PaperAnalysis],
    cache_keys: Mapping[str, PaperAnalysisCacheKey],
    analyze: Callable[[str], PaperAnalysis],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> ComparisonAnalysisRun:
    """Reuse valid entries and independently retain successes if another paper fails."""
    selected = tuple(paper_ids)
    if not 2 <= len(selected) <= 5:
        raise ValueError("Select between 2 and 5 papers for comparison")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected paper IDs must be unique")
    if any(paper_id not in cache_keys for paper_id in selected):
        raise ValueError("Missing analysis cache key for a selected paper")

    reused: list[str] = []
    newly_analyzed: list[str] = []
    failures: list[PaperAnalysisFailure] = []
    completed: list[PaperAnalysis] = []

    for index, paper_id in enumerate(selected, start=1):
        if on_progress is not None:
            on_progress(index, len(selected), paper_id)
        key = cache_keys[paper_id]
        cached = cache.get(key)
        if cached is not None:
            if (
                cached.paper_id == paper_id
                and cached.provider in (None, key.provider)
                and cached.model in (None, key.model)
            ):
                reused.append(paper_id)
                completed.append(cached)
                continue
            del cache[key]

        try:
            result = analyze(paper_id)
            if result.paper_id != paper_id:
                raise ValueError("Analysis returned a different paper ID")
            if result.provider not in (None, key.provider) or result.model not in (None, key.model):
                raise ValueError("Analysis provider/model does not match its cache key")
        except Exception as exc:
            failures.append(PaperAnalysisFailure(paper_id, str(exc)))
            continue
        cache[key] = result
        newly_analyzed.append(paper_id)
        completed.append(result)

    return ComparisonAnalysisRun(
        selected_paper_ids=selected,
        reused_paper_ids=tuple(reused),
        analyzed_paper_ids=tuple(newly_analyzed),
        failures=tuple(failures),
        analyses=tuple(completed),
    )
