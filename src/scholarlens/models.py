from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PageText:
    paper_id: str
    source_filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    paper_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    paper_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    text: str
    distance: float


class AnalysisStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class AnalysisEvidence:
    # IDs are local to one grouped paper-analysis evidence pool.
    evidence_id: str
    result: RetrievalResult


@dataclass(frozen=True)
class AnalysisField:
    """Validated field value with application-resolved supporting passages."""

    status: AnalysisStatus
    value: str | None
    evidence: tuple[AnalysisEvidence, ...]


@dataclass(frozen=True)
class GroupTiming:
    """Wall-clock measurements for one extraction group."""

    group_name: str
    generation_seconds: float
    generation_calls: int = 0


@dataclass(frozen=True)
class AnalysisTiming:
    """Non-persistent wall-clock measurements for one analysis operation."""

    retrieval_seconds: float
    generation_seconds: float
    group_timings: tuple[GroupTiming, ...] = ()

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True)
class PaperAnalysis:
    paper_id: str
    # None means no field had retrieved evidence, so no provider was called.
    model: str | None
    provider: str | None
    # Group 1: Research framing
    research_problem: AnalysisField
    research_question: AnalysisField
    research_gap: AnalysisField
    contributions: AnalysisField
    # Group 2: Technical approach
    methodology: AnalysisField
    dataset: AnalysisField
    proposed_method: AnalysisField
    # Group 3: Evaluation & outcomes
    evaluation_metrics: AnalysisField
    key_results: AnalysisField
    limitations: AnalysisField
    future_work: AnalysisField
    timing: AnalysisTiming | None = None
