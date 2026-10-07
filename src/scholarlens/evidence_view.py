"""Prepare existing claim and passage objects for evidence inspection in the UI."""

from __future__ import annotations

from dataclasses import dataclass

from scholarlens.cross_paper import CrossPaperGenerationResult, CrossPaperPool, paper_aliases
from scholarlens.comparison import ComparisonCell
from scholarlens.generation import assign_evidence_ids
from scholarlens.models import AnalysisField, RetrievalResult
from scholarlens.synthesis import SynthesisFinding, SynthesisResult


@dataclass(frozen=True)
class EvidenceViewItem:
    evidence_id: str
    result: RetrievalResult | None


@dataclass(frozen=True)
class ClaimEvidenceView:
    claim: str | None
    status: str
    evidence: tuple[EvidenceViewItem, ...]
    paper_id: str | None = None
    aspect: str | None = None


def prepare_analysis_claim(field: AnalysisField | ComparisonCell) -> ClaimEvidenceView:
    """Keep field status and the original resolved evidence in citation order."""
    evidence = tuple(
        EvidenceViewItem(item.evidence_id, item.result) for item in field.evidence
    )
    status = field.status.name
    if status == "SUPPORTED" and (not evidence or field.value is None):
        status = "INVALID_REFERENCE"
    if status == "INSUFFICIENT_EVIDENCE" and evidence:
        status = "INVALID_REFERENCE"
    return ClaimEvidenceView(field.value, status, evidence)


def prepare_cross_paper_claims(
    generated: CrossPaperGenerationResult,
    pool: CrossPaperPool,
) -> tuple[ClaimEvidenceView, ...]:
    """Resolve cited IDs against the exact answer snapshot; never trust a missing ID."""
    if generated.response is None:
        return ()
    if generated.evidence != pool.evidence or generated.question != pool.question:
        raise ValueError("Cross-paper answer and evidence pool do not match")

    aliases = paper_aliases(pool)
    by_id = assign_evidence_ids(generated.evidence)
    decisions = {decision.claim_key: decision for decision in generated.decisions}
    views: list[ClaimEvidenceView] = []
    for aspect_index, aspect in enumerate(generated.response.aspects):
        sides = {side.paper_id: side for side in aspect.sides}
        for alias, identity in aliases.items():
            side = sides.get(alias)
            if side is None:
                views.append(ClaimEvidenceView(None, "INVALID_REFERENCE", (), identity.paper_id, aspect.aspect))
                continue
            if side.claim is None:
                views.append(ClaimEvidenceView(None, "INSUFFICIENT_EVIDENCE", (), identity.paper_id, aspect.aspect))
                continue
            cited = tuple(
                EvidenceViewItem(
                    ref.evidence_id,
                    result if (result := by_id.get(ref.evidence_id)) is not None
                    and result.paper_id == identity.paper_id else None,
                )
                for ref in side.evidence
            )
            decision = decisions.get(f"{aspect_index}:{identity.paper_id}")
            status = decision.status.name if decision is not None else "UNVERIFIED"
            if not cited or any(item.result is None for item in cited):
                status = "INVALID_REFERENCE"
            views.append(ClaimEvidenceView(side.claim, status, cited, identity.paper_id, aspect.aspect))
    return tuple(views)


def prepare_synthesis_claims(
    finding: SynthesisFinding, result: SynthesisResult,
) -> tuple[ClaimEvidenceView, ...]:
    """Resolve verified synthesis positions against their retained evidence snapshot."""
    by_id = assign_evidence_ids(result.pool.evidence)
    views = []
    for position in finding.positions:
        cited = tuple(EvidenceViewItem(eid, record if (
            (record := by_id.get(eid)) is not None and record.paper_id == position.paper_id
        ) else None) for eid in position.cited_evidence_ids)
        status = position.status.name
        if position.claim is not None and (not cited or any(item.result is None for item in cited)):
            status = "INVALID_REFERENCE"
        views.append(ClaimEvidenceView(position.claim, status, cited, position.paper_id, finding.aspect))
    return tuple(views)
