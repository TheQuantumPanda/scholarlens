"""Offline Phase 6A records and scoring. This module never invokes the pipeline."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Answerability(str, Enum):
    ALL_THREE = "all_three"
    EXACTLY_TWO = "exactly_two"
    ONE = "one"
    NONE = "none"


class Support(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class BudgetLoss(str, Enum):
    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    PIPELINE_ERROR = "PIPELINE_ERROR"
    VERIFIER_FALSE_APPROVAL = "VERIFIER_FALSE_APPROVAL"
    VERIFIER_FALSE_REJECTION = "VERIFIER_FALSE_REJECTION"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GENERATION_GROUNDING_FAILURE = "GENERATION_GROUNDING_FAILURE"
    GENERATION_OVER_ABSTENTION = "GENERATION_OVER_ABSTENTION"
    CORRECT_ABSTENTION = "CORRECT_ABSTENTION"
    PASS = "PASS"


class Facet(_Record):
    paper_id: str
    facet: str


class ReferenceQuestion(_Record):
    question_id: str
    question: str
    focused_query: str | None
    answerability: Answerability
    relevant_papers: list[str]
    expected_facets: list[Facet]
    unanswerable_reason: str | None
    qualifier_checks: list[str]
    scope_checks: list[str]
    retrieval_dev_reference: str | None
    expected_support_review: Literal["whole_corpus_review", "judged_retrieval_window", "not_applicable"]
    absence_review: Literal["whole_corpus_review", "whole_corpus_text_search", "judged_retrieval_window", "not_asserted"]

    @model_validator(mode="after")
    def check_distribution(self) -> ReferenceQuestion:
        expected = {Answerability.ALL_THREE: 3, Answerability.EXACTLY_TWO: 2,
                    Answerability.ONE: 1, Answerability.NONE: 0}[self.answerability]
        if len(set(self.relevant_papers)) != expected or len(self.relevant_papers) != expected:
            raise ValueError("Relevant paper count does not match answerability")
        if {item.paper_id for item in self.expected_facets} != set(self.relevant_papers):
            raise ValueError("Expected facets must cover exactly the relevant papers")
        if expected < 2 and not self.unanswerable_reason:
            raise ValueError("Insufficient comparison needs a reason")
        return self


class ReferenceFixture(_Record):
    schema_version: Literal[1]
    retrieval_fixture: str
    notes: str
    questions: list[ReferenceQuestion]

    @model_validator(mode="after")
    def unique_questions(self) -> ReferenceFixture:
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate question IDs")
        return self


class RunMetadata(_Record):
    run_id: str
    provider: str
    model: str
    prompt_version_or_hash: str | None = None
    schema_version_or_hash: str | None = None
    timestamp: str | None = None
    corpus_hash: str | None = None
    evaluation_fixture_hash: str | None = None
    provider_audit_artifact: bool = False


class RetainedEvidence(_Record):
    evidence_id: str
    paper_id: str
    page: int
    chunk_id: str
    text_hash: str
    text: str | None = None


class CandidateObservation(_Record):
    paper_id: str
    rank: int
    page: int
    chunk_id: str
    text_hash: str
    distance: float
    filtered_reason: str | None = None
    retained: bool
    text: str | None = None


class RetrievalRecord(_Record):
    candidate_counts: dict[str, int]
    junk_filtered_counts: dict[str, int]
    retained_evidence: list[RetainedEvidence]
    candidate_observations: list[CandidateObservation] = []
    selector_observations: list[str] = []
    latency_seconds: float | None = None
    useful_evidence_present_by_expected_paper: dict[str, bool]
    budget_loss: BudgetLoss = BudgetLoss.UNKNOWN
    budget_loss_evidence: str | None = None

    @model_validator(mode="after")
    def require_selector_observation(self) -> RetrievalRecord:
        if self.budget_loss is not BudgetLoss.UNKNOWN and not self.budget_loss_evidence:
            raise ValueError("Budget-loss classification requires selector evidence")
        return self


class ClaimRecord(_Record):
    claim_key: str
    paper_id: str
    text: str
    cited_evidence_ids: list[str]
    anchors: list[str | None] = []
    rendered: bool = False


class GenerationRecord(_Record):
    non_empty: bool
    structurally_valid: bool
    represented_papers: list[str]
    claims: list[ClaimRecord]
    raw_structured_response: dict[str, Any] | None = None
    raw_response_text: str | None = None
    latency_seconds: float | None = None
    retry_delays_seconds: list[float] = []


class Decision(_Record):
    claim_key: str
    status: Support
    reason: str | None = None


class ClaimAudit(_Record):
    claim_key: str
    reference_support: Support
    notes: str = ""


class VerificationRecord(_Record):
    deterministic_decisions: list[Decision]
    llm_decisions: list[Decision]
    audit_annotations: list[ClaimAudit]
    raw_structured_response: dict[str, Any] | None = None
    raw_response_text: str | None = None
    schema_valid: bool | None = None
    keys_valid: bool | None = None
    latency_seconds: float | None = None
    retry_delays_seconds: list[float] = []


class FinalRecord(_Record):
    rendered_comparison_available: bool
    correct_insufficiency: bool
    unsupported_claim_rendered: bool | None = None
    outcome: Outcome | None = None
    rendered_result: str | None = None
    qualifying_aspects: list[str] = []
    citations_shown: list[str] = []
    safe_abstention: bool | None = None


class EvaluationRun(_Record):
    schema_version: Literal[1]
    question_id: str
    question: str
    metadata: RunMetadata
    retrieval: RetrievalRecord
    generation: GenerationRecord
    verification: VerificationRecord
    final: FinalRecord
    pipeline_error: str | None = None

    @model_validator(mode="after")
    def unique_keys(self) -> EvaluationRun:
        claims = [claim.claim_key for claim in self.generation.claims]
        if len(claims) != len(set(claims)):
            raise ValueError("Duplicate claim keys")
        for records in (self.verification.deterministic_decisions,
                        self.verification.llm_decisions,
                        self.verification.audit_annotations):
            keys = [item.claim_key for item in records]
            if len(keys) != len(set(keys)) or not set(keys) <= set(claims):
                raise ValueError("Invalid decision or audit claim keys")
        if (any(evidence.text is not None for evidence in self.retrieval.retained_evidence)
                or any(candidate.text is not None for candidate in self.retrieval.candidate_observations)) \
                and not self.metadata.provider_audit_artifact:
            raise ValueError("Passage text requires an explicit provider audit artifact")
        audited_bad = {item.claim_key for item in self.verification.audit_annotations
                       if item.reference_support in (Support.UNSUPPORTED, Support.INSUFFICIENT_EVIDENCE)}
        if (self.final.unsupported_claim_rendered is False
                and any(claim.rendered and claim.claim_key in audited_bad
                        for claim in self.generation.claims)):
            raise ValueError("Final unsupported-claim observation contradicts audited rendered claims")
        return self


def load_fixture(path: Path) -> tuple[ReferenceFixture, str]:
    content = path.read_bytes()
    fixture = ReferenceFixture.model_validate_json(content)
    return fixture, hashlib.sha256(content).hexdigest()


def load_runs(path: Path) -> list[EvaluationRun]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [EvaluationRun.model_validate(item) for item in data]
    return [EvaluationRun.model_validate(data)]


def save_run(path: Path, run: EvaluationRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


class Diagnosis(_Record):
    primary_outcome: Outcome
    stage_flags: dict[str, bool]
    retrieval_diagnosis: str
    generation_diagnosis: str
    verification_diagnosis: str
    unaudited_claims: int


def _decisions(run: EvaluationRun) -> dict[str, Support]:
    # Deterministic blocks take precedence; the LLM only sees remaining claims.
    return {**{item.claim_key: item.status for item in run.verification.llm_decisions},
            **{item.claim_key: item.status for item in run.verification.deterministic_decisions}}


def diagnose(reference: ReferenceQuestion, run: EvaluationRun) -> Diagnosis:
    if (reference.question_id, reference.question) != (run.question_id, run.question):
        raise ValueError("Run question does not match reference")
    expected = set(reference.relevant_papers)
    useful = run.retrieval.useful_evidence_present_by_expected_paper
    if not set(useful) <= expected:
        raise ValueError("Useful-evidence observations contain an unexpected paper")
    known_useful = sum(useful.values())
    unknown = len(expected - useful.keys())
    answerable = reference.answerability in (Answerability.ALL_THREE, Answerability.EXACTLY_TWO)
    retrieval_failure = answerable and known_useful + unknown < 2
    retrieval_text = (f"useful {known_useful}/{len(expected)} expected papers"
                      + (f", {unknown} unknown" if unknown else ""))
    if run.retrieval.budget_loss is BudgetLoss.CONFIRMED:
        retrieval_text += ", budget loss confirmed"
    elif run.retrieval.budget_loss is BudgetLoss.UNKNOWN:
        retrieval_text += ", budget loss unknown"
    audits = {item.claim_key: item.reference_support for item in run.verification.audit_annotations}
    decisions = _decisions(run)
    claims = run.generation.claims
    bad = {Support.UNSUPPORTED, Support.INSUFFICIENT_EVIDENCE}
    false_approval = any(audits.get(c.claim_key) in bad
                         and decisions.get(c.claim_key) is Support.SUPPORTED and c.rendered
                         for c in claims)
    false_rejection = any(audits.get(c.claim_key) is Support.SUPPORTED
                          and decisions.get(c.claim_key) in bad for c in claims) and not run.final.rendered_comparison_available
    grounding = any(audits.get(c.claim_key) in bad for c in claims)
    generated_comparison = run.generation.structurally_valid and len(
        {c.paper_id for c in claims}) >= 2
    over_abstention = (answerable and known_useful >= 2 and not generated_comparison)
    correct_abstention = not answerable and run.final.correct_insufficiency and not run.final.rendered_comparison_available
    pipeline_error = bool(run.pipeline_error) or (run.generation.non_empty and not run.generation.structurally_valid)
    flags = {
        "pipeline_error": pipeline_error,
        "verifier_false_approval": false_approval,
        "verifier_false_rejection": false_rejection,
        "retrieval_failure": retrieval_failure,
        "generation_grounding_failure": grounding,
        "generation_over_abstention": over_abstention,
        "correct_abstention": correct_abstention,
    }
    precedence = (
        ("pipeline_error", Outcome.PIPELINE_ERROR),
        ("verifier_false_approval", Outcome.VERIFIER_FALSE_APPROVAL),
        ("verifier_false_rejection", Outcome.VERIFIER_FALSE_REJECTION),
        ("retrieval_failure", Outcome.RETRIEVAL_FAILURE),
        ("generation_grounding_failure", Outcome.GENERATION_GROUNDING_FAILURE),
        ("generation_over_abstention", Outcome.GENERATION_OVER_ABSTENTION),
        ("correct_abstention", Outcome.CORRECT_ABSTENTION),
    )
    primary = next((outcome for flag, outcome in precedence if flags[flag]), Outcome.PASS)
    if primary is Outcome.PASS and not run.final.rendered_comparison_available:
        # An incomplete/ambiguous observation must not look like successful completion.
        primary = Outcome.PIPELINE_ERROR
        flags["pipeline_error"] = True
    if run.final.outcome is not None and run.final.outcome is not primary:
        raise ValueError("Recorded final outcome disagrees with the classifier")
    unaudited = sum(c.claim_key not in audits for c in claims)
    return Diagnosis(
        primary_outcome=primary, stage_flags=flags,
        retrieval_diagnosis=retrieval_text,
        generation_diagnosis=("comparison generated" if generated_comparison else "no useful comparison generated")
        + (", audited grounding failure" if grounding else ""),
        verification_diagnosis=("false approval" if false_approval else "false rejection" if false_rejection else "no audited verifier error")
        + f", {unaudited} unaudited claims",
        unaudited_claims=unaudited,
    )


def with_outcome(reference: ReferenceQuestion, run: EvaluationRun) -> EvaluationRun:
    """Return a serializable scored copy while preserving the raw observations."""
    outcome = diagnose(reference, run).primary_outcome
    return run.model_copy(update={"final": run.final.model_copy(update={"outcome": outcome})})


class Metric(_Record):
    numerator: int
    denominator: int
    percentage: float | None


def aggregate(pairs: list[tuple[ReferenceQuestion, EvaluationRun]]) -> dict[str, dict[str, Metric]]:
    names = (
        "answerable_question_completion_rate", "correct_abstention_rate", "over_abstention_rate",
        "unsupported_claim_escape_rate", "verifier_false_approval_rate", "verifier_false_rejection_rate",
        "expected_paper_retrieval_coverage", "supported_final_paper_coverage",
        "retrieval_failure_rate", "generation_grounding_failure_rate",
    )
    counts: dict[str, dict[str, list[int]]] = {}

    def add(group: str, name: str, numerator: int, denominator: int) -> None:
        bucket = counts.setdefault(group, {key: [0, 0] for key in names})[name]
        bucket[0] += numerator
        bucket[1] += denominator

    for ref, run in pairs:
        diagnosis = diagnose(ref, run)
        groups = ("all", ref.answerability.value)
        answerable = ref.answerability in (Answerability.ALL_THREE, Answerability.EXACTLY_TWO)
        audits = {a.claim_key: a.reference_support for a in run.verification.audit_annotations}
        decisions = _decisions(run)
        bad = {Support.UNSUPPORTED, Support.INSUFFICIENT_EVIDENCE}
        bad_claims = [c for c in run.generation.claims if audits.get(c.claim_key) in bad]
        supported_claims = [c for c in run.generation.claims if audits.get(c.claim_key) is Support.SUPPORTED]
        useful = run.retrieval.useful_evidence_present_by_expected_paper
        supported_final = {c.paper_id for c in supported_claims if c.rendered}
        for group in groups:
            add(group, "answerable_question_completion_rate", int(answerable and run.final.rendered_comparison_available), int(answerable))
            add(group, "correct_abstention_rate", int(not answerable and diagnosis.stage_flags["correct_abstention"]), int(not answerable))
            add(group, "over_abstention_rate", int(answerable and diagnosis.stage_flags["generation_over_abstention"]), int(answerable))
            add(group, "unsupported_claim_escape_rate", sum(c.rendered for c in bad_claims), len(bad_claims))
            add(group, "verifier_false_approval_rate", sum(decisions.get(c.claim_key) is Support.SUPPORTED for c in bad_claims), len([c for c in bad_claims if c.claim_key in decisions]))
            add(group, "verifier_false_rejection_rate", sum(decisions.get(c.claim_key) in bad for c in supported_claims), len([c for c in supported_claims if c.claim_key in decisions]))
            add(group, "expected_paper_retrieval_coverage", sum(useful.values()), len(useful))
            add(group, "supported_final_paper_coverage", len(supported_final & set(ref.relevant_papers)), len(ref.relevant_papers))
            add(group, "retrieval_failure_rate", int(diagnosis.stage_flags["retrieval_failure"]), int(answerable))
            add(group, "generation_grounding_failure_rate", int(diagnosis.stage_flags["generation_grounding_failure"]), 1)
    return {group: {name: Metric(numerator=n, denominator=d,
                                 percentage=round(100 * n / d, 1) if d else None)
                    for name, (n, d) in metrics.items()}
            for group, metrics in counts.items()}
