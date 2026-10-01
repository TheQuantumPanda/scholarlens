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
class AnalysisTiming:
    """Non-persistent wall-clock measurements for one analysis operation."""

    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds


@dataclass(frozen=True)
class PaperAnalysis:
    paper_id: str
    # None means no field had retrieved evidence, so Ollama was not called.
    model: str | None
    research_problem: AnalysisField
    methodology: AnalysisField
    key_results: AnalysisField
    timing: AnalysisTiming | None = None
