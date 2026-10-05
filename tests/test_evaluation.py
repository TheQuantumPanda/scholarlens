"""Phase 6A is record-only: every stage observation here is mocked."""

import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scholarlens.evaluation import (
    BudgetLoss, EvaluationRun, Outcome,
    aggregate, diagnose, load_fixture, load_runs, save_run, with_outcome,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/phase6a_end_to_end_reference.json"


def mocked_run(ref, *, useful=None, claims=None, audits=None, decisions=None,
               comparison=True, insufficiency=False, error=None):
    useful = useful if useful is not None else {p: True for p in ref.relevant_papers}
    claims = claims if claims is not None else [
        {"claim_key": f"0:{paper}", "paper_id": paper, "text": "Mock claim",
         "cited_evidence_ids": [f"E{index}"], "rendered": comparison}
        for index, paper in enumerate(ref.relevant_papers[:2], 1)
    ]
    audits = audits if audits is not None else [
        {"claim_key": claim["claim_key"], "reference_support": "SUPPORTED", "notes": "Human mock label"}
        for claim in claims
    ]
    decisions = decisions if decisions is not None else [
        {"claim_key": claim["claim_key"], "status": "SUPPORTED"}
        for claim in claims
    ]
    return EvaluationRun.model_validate({
        "schema_version": 1, "question_id": ref.question_id, "question": ref.question,
        "metadata": {"run_id": "mock-1", "provider": "mock", "model": "mock", "evaluation_fixture_hash": None},
        "retrieval": {"candidate_counts": {p: 3 for p in ref.relevant_papers},
                      "junk_filtered_counts": {p: 0 for p in ref.relevant_papers},
                      "retained_evidence": [{"evidence_id": f"E{i}", "paper_id": p,
                                              "page": 1, "chunk_id": f"chunk-{i}", "text_hash": "a" * 64}
                                             for i, p in enumerate(ref.relevant_papers, 1)],
                      "useful_evidence_present_by_expected_paper": useful,
                      "budget_loss": "unknown"},
        "generation": {"non_empty": bool(claims), "structurally_valid": True,
                       "represented_papers": [claim["paper_id"] for claim in claims], "claims": claims},
        "verification": {"deterministic_decisions": [], "llm_decisions": decisions,
                         "audit_annotations": audits},
        "final": {"rendered_comparison_available": comparison,
                  "correct_insufficiency": insufficiency,
                  "unsupported_claim_rendered": None},
        "pipeline_error": error,
    })


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.fixture_hash = load_fixture(FIXTURE)
        cls.refs = {q.question_id: q for q in cls.fixture.questions}

    def test_reference_fixture_distribution_and_serialization(self):
        counts = {answer: sum(q.answerability.value == answer for q in self.fixture.questions)
                  for answer in ("all_three", "exactly_two", "one", "none")}
        self.assertEqual(counts, {"all_three": 4, "exactly_two": 4, "one": 2, "none": 2})
        self.assertEqual(self.fixture.model_validate_json(self.fixture.model_dump_json()), self.fixture)
        dev = json.loads((FIXTURE.parent / self.fixture.retrieval_fixture).read_text())
        dev_ids = {q["id"] for q in dev["questions"]}
        self.assertTrue(all(q.retrieval_dev_reference in dev_ids
                            for q in self.fixture.questions if q.retrieval_dev_reference))
        self.assertTrue(all("labels" not in q.model_dump() and "candidates" not in q.model_dump()
                            for q in self.fixture.questions))

    def test_pass_and_run_serialization(self):
        ref = self.refs["q1"]
        run = with_outcome(ref, mocked_run(ref))
        self.assertEqual(diagnose(ref, run).primary_outcome, Outcome.PASS)
        self.assertEqual(run.final.outcome, Outcome.PASS)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            save_run(path, run)
            self.assertEqual(load_runs(path), [run])
            self.assertNotIn("Mock passage", path.read_text())

    def test_correct_abstention_for_one_and_none(self):
        for key in ("q4", "q11"):
            with self.subTest(key=key):
                ref = self.refs[key]
                run = mocked_run(ref, claims=[], comparison=False, insufficiency=True)
                self.assertEqual(diagnose(ref, run).primary_outcome, Outcome.CORRECT_ABSTENTION)

    def test_retrieval_failure(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, useful={ref.relevant_papers[0]: True,
                                      ref.relevant_papers[1]: False,
                                      ref.relevant_papers[2]: False},
                         claims=[], comparison=False)
        self.assertEqual(diagnose(ref, run).primary_outcome, Outcome.RETRIEVAL_FAILURE)

    def test_over_abstention_with_empty_aspects(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, claims=[], comparison=False)
        result = diagnose(ref, run)
        self.assertEqual(result.primary_outcome, Outcome.GENERATION_OVER_ABSTENTION)
        self.assertTrue(result.stage_flags["generation_over_abstention"])

    def test_grounding_failure_safely_withheld(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, audits=[{"claim_key": f"0:{p}",
                                       "reference_support": "UNSUPPORTED" if i == 0 else "SUPPORTED"}
                                      for i, p in enumerate(ref.relevant_papers[:2])],
                         decisions=[{"claim_key": f"0:{p}",
                                     "status": "UNSUPPORTED" if i == 0 else "SUPPORTED"}
                                    for i, p in enumerate(ref.relevant_papers[:2])],
                         comparison=False)
        self.assertEqual(diagnose(ref, run).primary_outcome, Outcome.GENERATION_GROUNDING_FAILURE)

    def test_false_approval_precedes_grounding_failure(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, audits=[{"claim_key": f"0:{p}",
                                       "reference_support": "UNSUPPORTED" if i == 0 else "SUPPORTED"}
                                      for i, p in enumerate(ref.relevant_papers[:2])])
        result = diagnose(ref, run)
        self.assertEqual(result.primary_outcome, Outcome.VERIFIER_FALSE_APPROVAL)
        self.assertTrue(result.stage_flags["generation_grounding_failure"])
        self.assertEqual(aggregate([(ref, run)])["all"]["unsupported_claim_escape_rate"].numerator, 1)

    def test_false_rejection_when_comparison_lost(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, decisions=[{"claim_key": f"0:{p}",
                                          "status": "UNSUPPORTED" if i == 0 else "SUPPORTED"}
                                         for i, p in enumerate(ref.relevant_papers[:2])],
                         comparison=False)
        self.assertEqual(diagnose(ref, run).primary_outcome, Outcome.VERIFIER_FALSE_REJECTION)

    def test_pipeline_error_precedence_and_unknown_budget(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, audits=[{"claim_key": f"0:{p}", "reference_support": "UNSUPPORTED"}
                                      for p in ref.relevant_papers[:2]], error="Mock provider failure")
        result = diagnose(ref, run)
        self.assertEqual(result.primary_outcome, Outcome.PIPELINE_ERROR)
        self.assertTrue(result.stage_flags["verifier_false_approval"])
        self.assertEqual(run.retrieval.budget_loss, BudgetLoss.UNKNOWN)
        self.assertIn("budget loss unknown", result.retrieval_diagnosis)

    def test_budget_loss_and_passage_text_need_explicit_provenance(self):
        ref = self.refs["q1"]
        data = mocked_run(ref).model_dump(mode="json")
        data["retrieval"]["budget_loss"] = "confirmed"
        with self.assertRaisesRegex(ValueError, "selector evidence"):
            EvaluationRun.model_validate(data)
        data["retrieval"]["budget_loss_evidence"] = "Selector skipped a judged useful candidate that did not fit."
        self.assertEqual(EvaluationRun.model_validate(data).retrieval.budget_loss, BudgetLoss.CONFIRMED)
        data["retrieval"]["retained_evidence"][0]["text"] = "Local paper passage"
        with self.assertRaisesRegex(ValueError, "provider audit artifact"):
            EvaluationRun.model_validate(data)
        data["metadata"]["provider_audit_artifact"] = True
        self.assertEqual(EvaluationRun.model_validate(data).retrieval.retained_evidence[0].text,
                         "Local paper passage")

    def test_unknown_usefulness_does_not_force_retrieval_failure(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, useful={ref.relevant_papers[0]: True}, claims=[], comparison=False)
        result = diagnose(ref, run)
        self.assertFalse(result.stage_flags["retrieval_failure"])
        self.assertIn("2 unknown", result.retrieval_diagnosis)

    def test_unaudited_excluded_and_zero_denominators(self):
        ref = self.refs["q1"]
        run = mocked_run(ref, audits=[])
        result = diagnose(ref, run)
        self.assertEqual(result.unaudited_claims, 2)
        metrics = aggregate([(ref, run)])["all"]
        for name in ("unsupported_claim_escape_rate", "verifier_false_approval_rate",
                     "verifier_false_rejection_rate", "correct_abstention_rate"):
            self.assertEqual(metrics[name].denominator, 0)
            self.assertIsNone(metrics[name].percentage)

    def test_counts_percentages_and_answerability_breakdown(self):
        first = self.refs["q1"]
        second = self.refs["q4"]
        metrics = aggregate([(first, mocked_run(first)),
                             (second, mocked_run(second, claims=[], comparison=False, insufficiency=True))])
        self.assertEqual(metrics["all"]["answerable_question_completion_rate"].model_dump(),
                         {"numerator": 1, "denominator": 1, "percentage": 100.0})
        self.assertEqual(metrics["all"]["expected_paper_retrieval_coverage"].model_dump(),
                         {"numerator": 4, "denominator": 4, "percentage": 100.0})
        self.assertEqual(metrics["one"]["correct_abstention_rate"].model_dump(),
                         {"numerator": 1, "denominator": 1, "percentage": 100.0})
        self.assertIn("all_three", metrics)

    def test_offline_report_command(self):
        ref = self.refs["q1"]
        run = mocked_run(ref)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.json"
            save_run(path, run)
            output = io.StringIO()
            with patch.object(sys, "argv", ["report_phase6a.py", str(path)]), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")), \
                 contextlib.redirect_stdout(output):
                runpy.run_path(str(ROOT / "scripts/report_phase6a.py"), run_name="__main__")
            self.assertIn("PASS", output.getvalue())
            self.assertIn("1/1 (100.0%)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
