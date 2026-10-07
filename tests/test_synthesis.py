import json
import unittest
from unittest.mock import patch

from scholarlens.cross_paper import CrossPaperConfig, CrossPaperPool, PaperCandidates
from scholarlens.evidence_view import prepare_synthesis_claims
from scholarlens.generation import GenerationError, OllamaConfig
from scholarlens.models import RetrievalResult
from scholarlens.synthesis import Classification, _validate, synthesize
from scholarlens.verification import VerificationDecision, VerificationStatus


def record(paper, text):
    return RetrievalResult(1, paper, f"{paper}.pdf", 2, f"{paper}-chunk", text, .1)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.a = record("a", "On dataset Alpha, method M reached 80% accuracy.")
        self.b = record("b", "On dataset Alpha, method M reached 82% accuracy.")
        self.pool = CrossPaperPool("method M accuracy", (
            PaperCandidates("a", (self.a,)), PaperCandidates("b", (self.b,))),
            (self.a, self.b),
        )
        self.config = OllamaConfig(model="test")
        self.budget = CrossPaperConfig(safe_prompt_tokens=10000)
        self.transport = patch("scholarlens.generation.urlopen", side_effect=AssertionError("No live HTTP"))
        self.transport.start()
        self.addCleanup(self.transport.stop)

    def response(self, classification="CONSENSUS", contexts=(), claims=("M reached 80% on Alpha.", "M reached 82% on Alpha."), refs=("E1", "E2")):
        return json.dumps({"findings": [{"aspect": "Accuracy of M", "classification": classification,
            "context_differences": list(contexts), "sides": [
                {"paper_id": alias, "claim": claim, "evidence": ([{"evidence_id": ref, "anchor": None}] if claim else [])}
                for alias, claim, ref in zip(("P1", "P2"), claims, refs)
            ]}]})

    def run_synthesis(self, raw, statuses=(VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED), pool=None):
        with patch("scholarlens.synthesis.generate_chat", return_value=raw) as generate, patch(
            "scholarlens.synthesis.verify_claims", side_effect=lambda claims, config, **kw: {
                claim.claim_key: VerificationDecision(claim_key=claim.claim_key, status=status, reason="Reviewed")
                for claim, status in zip(claims, statuses)
            }
        ) as verify:
            result = synthesize(pool or self.pool, self.config, budget=self.budget)
        return result, generate, verify

    def test_two_paper_consensus_and_evidence_view(self):
        result, generate, verify = self.run_synthesis(self.response())
        finding = result.findings[0]
        self.assertEqual(finding.classification, Classification.CONSENSUS)
        self.assertIn("Multiple papers report", finding.summary)
        self.assertEqual([p.paper_id for p in finding.positions], ["a", "b"])
        self.assertEqual([p.cited_evidence_ids for p in finding.positions], [("E1",), ("E2",)])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(verify.call_count, 1)
        views = prepare_synthesis_claims(finding, result)
        self.assertEqual([v.status for v in views], ["SUPPORTED", "SUPPORTED"])
        self.assertIs(views[0].evidence[0].result, self.a)
        self.assertEqual(views[1].evidence[0].result.chunk_id, "b-chunk")

    def test_different_datasets_are_potential_disagreement_with_context(self):
        b = record("b", "On dataset Beta, method M reached 60% accuracy.")
        pool = CrossPaperPool(self.pool.question, (self.pool.papers[0], PaperCandidates("b", (b,))), (self.a, b))
        result, _, _ = self.run_synthesis(self.response("POTENTIAL_DISAGREEMENT", ("dataset",),
            ("M reached 80% on Alpha.", "M reached 60% on Beta.")), pool=pool)
        self.assertEqual(result.findings[0].classification, Classification.POTENTIAL_DISAGREEMENT)
        self.assertIn("dataset", result.findings[0].context_note)
        self.assertNotIn("contradiction", result.findings[0].summary)

    def test_materially_different_results_in_same_context(self):
        result, _, _ = self.run_synthesis(self.response("POTENTIAL_DISAGREEMENT", (),
            ("M improved accuracy on Alpha.", "M reduced accuracy on Alpha.")))
        self.assertEqual(result.findings[0].classification, Classification.POTENTIAL_DISAGREEMENT)
        self.assertIsNone(result.findings[0].context_note)

    def test_context_difference_prevents_consensus_label(self):
        result, _, _ = self.run_synthesis(self.response("CONSENSUS", ("dataset",)))
        self.assertEqual(result.findings[0].classification, Classification.INSUFFICIENT_EVIDENCE)

    def test_identical_positions_cannot_be_labeled_disagreement(self):
        result, _, _ = self.run_synthesis(self.response("POTENTIAL_DISAGREEMENT", (),
            ("Same reported result.", "Same reported result.")))
        self.assertEqual(result.findings[0].classification, Classification.INSUFFICIENT_EVIDENCE)

    def test_one_paper_evidence_fails_closed_without_generation(self):
        pool = CrossPaperPool(self.pool.question, self.pool.papers, (self.a,))
        result, generate, verify = self.run_synthesis(self.response(), pool=pool)
        self.assertEqual(result.findings[0].classification, Classification.INSUFFICIENT_EVIDENCE)
        generate.assert_not_called()
        verify.assert_not_called()

    def test_failed_verifier_side_downgrades_and_withholds_claim(self):
        result, _, _ = self.run_synthesis(self.response(), (VerificationStatus.SUPPORTED, VerificationStatus.UNSUPPORTED))
        finding = result.findings[0]
        self.assertEqual(finding.classification, Classification.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(finding.positions[1].claim)
        self.assertEqual(finding.positions[1].cited_evidence_ids, ())
        self.assertEqual(prepare_synthesis_claims(finding, result)[1].evidence, ())

    def test_missing_side_evidence_downgrades(self):
        result, _, _ = self.run_synthesis(self.response(claims=("M reached 80% on Alpha.", None)))
        self.assertEqual(result.findings[0].classification, Classification.INSUFFICIENT_EVIDENCE)

    def test_invalid_reference_is_rejected(self):
        with self.assertRaises(GenerationError):
            self.run_synthesis(self.response(refs=("E2", "E1")))

    def test_order_follows_selected_papers_not_model_order(self):
        raw = json.loads(self.response())
        raw["findings"][0]["sides"].reverse()
        result, _, _ = self.run_synthesis(json.dumps(raw))
        self.assertEqual([p.paper_id for p in result.findings[0].positions], ["a", "b"])


class SynthesisOutputContractTests(unittest.TestCase):
    """The live failure's recorded structure; its raw wording was not retained."""

    def setUp(self):
        records = tuple(record(paper, f"Passage from {paper}.") for paper in "abc")
        self.pool = CrossPaperPool("Retrieval quality", tuple(
            PaperCandidates(paper, (item,)) for paper, item in zip("abc", records)
        ), records)

    def side(self, alias, claim=None, refs=()):
        return {"paper_id": alias, "claim": claim,
                "evidence": [{"evidence_id": eid, "anchor": None} for eid in refs]}

    def finding(self, sides):
        return {"aspect": "Retrieval quality", "classification": "INSUFFICIENT_EVIDENCE",
                "sides": sides, "context_differences": []}

    def validate(self, sides):
        return _validate(json.dumps({"findings": [self.finding(sides)]}), self.pool)

    def test_recorded_live_failure_shape_is_rejected(self):
        # The first two recorded findings both repeated P3 with non-null claims
        # and no references; the third had P1/P2/P3 but a P2 claim with no refs.
        raw = {"findings": [
            self.finding([self.side("P3", "Claim A"), self.side("P3", "Claim B")]),
            self.finding([self.side("P3", "Claim C"), self.side("P3", "Claim D")]),
            self.finding([self.side("P1"), self.side("P2", "Claim E"), self.side("P3")]),
        ]}
        with self.assertRaises(GenerationError):
            _validate(json.dumps(raw), self.pool)

    def test_duplicate_alias_is_rejected_independently(self):
        with self.assertRaises(GenerationError):
            self.validate([self.side("P1"), self.side("P2"), self.side("P2")])

    def test_claim_without_evidence_is_rejected_independently(self):
        with self.assertRaises(GenerationError):
            self.validate([self.side("P1", "Claim without reference"), self.side("P2"), self.side("P3")])

    def test_null_claim_and_empty_evidence_are_accepted(self):
        response = self.validate([self.side("P1"), self.side("P2"), self.side("P3")])
        self.assertTrue(all(side.claim is None and not side.evidence
                            for side in response.findings[0].sides))

    def test_one_entry_per_paper_with_owned_evidence_is_accepted(self):
        response = self.validate([self.side("P1", "First claim", ("E1",)),
                                  self.side("P2", "Second claim", ("E2",)), self.side("P3")])
        self.assertEqual([side.paper_id for side in response.findings[0].sides],
                         ["P1", "P2", "P3"])

    def test_unknown_evidence_id_is_rejected(self):
        with self.assertRaises(GenerationError):
            self.validate([self.side("P1", "First claim", ("E999",)),
                           self.side("P2"), self.side("P3")])

    def test_wrong_paper_evidence_id_is_rejected(self):
        with self.assertRaises(GenerationError):
            self.validate([self.side("P1", "First claim", ("E2",)),
                           self.side("P2"), self.side("P3")])


if __name__ == "__main__":
    unittest.main()
