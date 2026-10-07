import unittest

from scholarlens.cross_paper import (
    CrossPaperGenerationResult,
    CrossPaperPool,
    CrossPaperResponse,
    PaperCandidates,
)
from scholarlens.evidence_view import prepare_analysis_claim, prepare_cross_paper_claims
from scholarlens.models import AnalysisEvidence, AnalysisField, AnalysisStatus, RetrievalResult
from scholarlens.verification import VerificationDecision, VerificationStatus


class EvidenceViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            RetrievalResult(1, "a", "a.pdf", 2, "a-1", "Exact first passage.\nSecond line.", .1),
            RetrievalResult(2, "a", "a.pdf", 4, "a-2", "Exact second passage.", .2),
            RetrievalResult(1, "b", "b.pdf", 7, "b-1", "Another paper's passage.", .3),
        )
        self.pool = CrossPaperPool(
            "Compare methods",
            (PaperCandidates("a", self.records[:2]), PaperCandidates("b", self.records[2:])),
            self.records,
        )

    def generated(self, *, refs=("E2", "E1"), first_status=VerificationStatus.SUPPORTED,
                  second_status=VerificationStatus.INSUFFICIENT_EVIDENCE):
        response = CrossPaperResponse.model_validate({"aspects": [{
            "aspect": "Method",
            "sides": [
                {"paper_id": "P1", "claim": "Specific method.",
                 "evidence": [{"evidence_id": ref, "anchor": None} for ref in refs]},
                {"paper_id": "P2", "claim": "Other method.",
                 "evidence": [{"evidence_id": "E3", "anchor": None}]},
            ],
        }]})
        decisions = (
            VerificationDecision(claim_key="0:a", status=first_status, reason="Reviewed."),
            VerificationDecision(claim_key="0:b", status=second_status, reason="Reviewed."),
        )
        return CrossPaperGenerationResult(
            self.pool.question, "Rendered answer", "test-model", "test-provider",
            self.pool.evidence, response, decisions,
        )

    def test_multiple_references_preserve_citation_order_and_exact_provenance(self):
        views = prepare_cross_paper_claims(self.generated(), self.pool)
        self.assertEqual([(view.paper_id, view.claim) for view in views],
                         [("a", "Specific method."), ("b", "Other method.")])
        self.assertEqual(views[0].status, "SUPPORTED")
        self.assertEqual([item.evidence_id for item in views[0].evidence], ["E2", "E1"])
        self.assertIs(views[0].evidence[0].result, self.records[1])
        self.assertEqual(views[0].evidence[1].result.source_filename, "a.pdf")
        self.assertEqual(views[0].evidence[1].result.page_number, 2)
        self.assertEqual(views[0].evidence[1].result.chunk_id, "a-1")
        self.assertEqual(views[0].evidence[1].result.text, "Exact first passage.\nSecond line.")
        self.assertEqual(views[1].status, "INSUFFICIENT_EVIDENCE")

    def test_unsupported_claim_keeps_status_and_cited_source_for_inspection(self):
        view = prepare_cross_paper_claims(
            self.generated(first_status=VerificationStatus.UNSUPPORTED), self.pool,
        )[0]
        self.assertEqual(view.status, "UNSUPPORTED")
        self.assertEqual(view.claim, "Specific method.")
        self.assertEqual([item.result for item in view.evidence],
                         [self.records[1], self.records[0]])

    def test_missing_or_wrong_paper_reference_is_never_presented_as_valid(self):
        for refs in (("E999",), ("E3",)):
            with self.subTest(refs=refs):
                view = prepare_cross_paper_claims(self.generated(refs=refs), self.pool)[0]
                self.assertEqual(view.status, "INVALID_REFERENCE")
                self.assertIsNone(view.evidence[0].result)

    def test_unverified_claim_is_not_labeled_supported(self):
        generated = self.generated()
        generated = CrossPaperGenerationResult(
            generated.question, generated.answer, generated.model, generated.provider,
            generated.evidence, generated.response, (),
        )
        self.assertEqual(prepare_cross_paper_claims(generated, self.pool)[0].status, "UNVERIFIED")

    def test_answer_snapshot_must_match_pool(self):
        other = CrossPaperPool("Different question", self.pool.papers, self.pool.evidence)
        with self.assertRaisesRegex(ValueError, "do not match"):
            prepare_cross_paper_claims(self.generated(), other)

    def test_analysis_and_matrix_fields_use_resolved_evidence_without_copying(self):
        field = AnalysisField(
            AnalysisStatus.SUPPORTED, "Supported finding.",
            (AnalysisEvidence("E1", self.records[0]), AnalysisEvidence("E2", self.records[1])),
        )
        view = prepare_analysis_claim(field)
        self.assertEqual(view.claim, "Supported finding.")
        self.assertEqual(view.status, "SUPPORTED")
        self.assertEqual([item.evidence_id for item in view.evidence], ["E1", "E2"])
        self.assertIs(view.evidence[0].result, self.records[0])

    def test_insufficient_and_missing_analysis_evidence(self):
        insufficient = prepare_analysis_claim(
            AnalysisField(AnalysisStatus.INSUFFICIENT_EVIDENCE, None, ()),
        )
        self.assertEqual((insufficient.claim, insufficient.status, insufficient.evidence),
                         (None, "INSUFFICIENT_EVIDENCE", ()))
        invalid = prepare_analysis_claim(AnalysisField(AnalysisStatus.SUPPORTED, "Claim", ()))
        self.assertEqual(invalid.status, "INVALID_REFERENCE")


if __name__ == "__main__":
    unittest.main()
