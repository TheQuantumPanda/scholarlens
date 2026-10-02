import unittest
from dataclasses import replace

from scholarlens.analysis import AnalysisConfig, FIELD_DEFINITIONS
from scholarlens.comparison import (
    analyze_selected_papers,
    build_comparison_matrix,
    make_analysis_cache_key,
)
from scholarlens.generation import OllamaConfig
from scholarlens.models import (
    AnalysisEvidence,
    AnalysisField,
    AnalysisStatus,
    PaperAnalysis,
    RetrievalResult,
)


def make_analysis(paper_id: str, *, insufficient_field: str | None = None) -> PaperAnalysis:
    values = {}
    for definition in FIELD_DEFINITIONS:
        if definition.name == insufficient_field:
            values[definition.name] = AnalysisField(
                AnalysisStatus.INSUFFICIENT_EVIDENCE, None, (),
            )
        else:
            result = RetrievalResult(
                rank=1,
                paper_id=paper_id,
                source_filename=f"{paper_id}.pdf",
                page_number=3,
                chunk_id=f"{paper_id}-{definition.name}-chunk-3",
                text=f"Original evidence for {definition.name} in {paper_id}.",
                distance=0.1,
            )
            # E1 is intentionally repeated across fields and papers.
            evidence = AnalysisEvidence("E1", result)
            values[definition.name] = AnalysisField(
                AnalysisStatus.SUPPORTED,
                f"  Exact extracted value for {definition.name} in {paper_id}.  ",
                (evidence,),
            )
    return PaperAnalysis(
        paper_id=paper_id,
        provider="ollama",
        model="qwen3:4b",
        **values,
    )


class ComparisonMatrixTests(unittest.TestCase):
    def setUp(self):
        self.analyses = [make_analysis("paper-a"), make_analysis("paper-b")]
        self.filenames = {"paper-a": "paper-a.pdf", "paper-b": "paper-b.pdf"}

    def test_two_analyses_produce_all_fields_with_exact_values_and_evidence(self):
        original = list(self.analyses)
        matrix = build_comparison_matrix(self.analyses, self.filenames)

        self.assertEqual(matrix.papers[0].paper_id, "paper-a")
        self.assertEqual(matrix.papers[1].paper_id, "paper-b")
        self.assertEqual(matrix.fields, tuple(field.name for field in FIELD_DEFINITIONS))
        self.assertEqual(len(matrix.cells), 22)
        for analysis in self.analyses:
            for definition in FIELD_DEFINITIONS:
                cell = matrix.get_cell(analysis.paper_id, definition.name)
                source = getattr(analysis, definition.name)
                self.assertEqual(cell.value, source.value)
                self.assertIs(cell.status, source.status)
                self.assertIs(cell.evidence, source.evidence)
                self.assertEqual(cell.source_filename, self.filenames[analysis.paper_id])
        self.assertEqual(self.analyses, original)
        self.assertEqual(matrix, build_comparison_matrix(self.analyses, self.filenames))

    def test_three_analyses_preserve_given_deterministic_order(self):
        analyses = [make_analysis(name) for name in ("paper-c", "paper-a", "paper-b")]
        filenames = {f"{name}": f"{name}.pdf" for name in ("paper-a", "paper-b", "paper-c")}
        matrix = build_comparison_matrix(analyses, filenames)
        self.assertEqual([paper.paper_id for paper in matrix.papers], ["paper-c", "paper-a", "paper-b"])
        self.assertEqual(
            [matrix.cells[index].paper_id for index in (0, 11, 22)],
            ["paper-c", "paper-a", "paper-b"],
        )

    def test_insufficient_values_and_status_are_preserved_exactly(self):
        analysis = make_analysis("paper-a", insufficient_field="dataset")
        matrix = build_comparison_matrix([analysis, self.analyses[1]], self.filenames)
        cell = matrix.get_cell("paper-a", "dataset")
        self.assertIs(cell.status, AnalysisStatus.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(cell.value)
        self.assertEqual(cell.evidence, ())

    def test_same_group_local_evidence_id_does_not_collide_between_papers(self):
        matrix = build_comparison_matrix(self.analyses, self.filenames)
        first = matrix.get_cell("paper-a", "research_problem").evidence[0]
        second = matrix.get_cell("paper-b", "research_problem").evidence[0]
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(first.result.paper_id, "paper-a")
        self.assertEqual(second.result.paper_id, "paper-b")
        self.assertEqual(first.result.page_number, 3)
        self.assertEqual(first.result.chunk_id, "paper-a-research_problem-chunk-3")
        self.assertEqual(first.result.source_filename, "paper-a.pdf")
        self.assertIn("Original evidence", first.result.text)
        same_group_local_id_different_field = matrix.get_cell("paper-a", "methodology").evidence[0]
        self.assertEqual(same_group_local_id_different_field.evidence_id, first.evidence_id)
        self.assertNotEqual(same_group_local_id_different_field.result.chunk_id, first.result.chunk_id)

    def test_builder_rejects_duplicate_papers_missing_names_and_cross_paper_evidence(self):
        with self.assertRaisesRegex(ValueError, "distinct papers"):
            build_comparison_matrix([self.analyses[0], self.analyses[0]], self.filenames)
        with self.assertRaisesRegex(ValueError, "source filename"):
            build_comparison_matrix(self.analyses, {"paper-a": "paper-a.pdf"})
        corrupted = replace(
            self.analyses[0],
            research_problem=replace(
                self.analyses[0].research_problem,
                evidence=self.analyses[1].research_problem.evidence,
            ),
        )
        with self.assertRaisesRegex(ValueError, "does not belong"):
            build_comparison_matrix([corrupted, self.analyses[1]], self.filenames)

    def test_matrix_requires_two_to_five_successful_analyses(self):
        for analyses in ([self.analyses[0]], [*self.analyses, *(make_analysis(f"p{i}") for i in range(4))]):
            with self.subTest(count=len(analyses)), self.assertRaisesRegex(ValueError, "between 2 and 5"):
                build_comparison_matrix(analyses, {a.paper_id: f"{a.paper_id}.pdf" for a in analyses})


class AnalysisReuseTests(unittest.TestCase):
    def setUp(self):
        self.config = OllamaConfig()
        self.analysis_config = AnalysisConfig()
        self.index_signature = ((('paper-a.pdf', 'content-hash'),), 250, 30)
        self.keys = {
            paper_id: make_analysis_cache_key(
                paper_id, self.index_signature, self.config, self.analysis_config,
            )
            for paper_id in ("paper-a", "paper-b", "paper-c", "paper-d")
        }

    def test_completed_analysis_is_reused_without_calling_analyzer(self):
        cache = {self.keys["paper-a"]: make_analysis("paper-a")}
        run = analyze_selected_papers(
            ["paper-a", "paper-b"], cache, self.keys,
            lambda paper_id: make_analysis(paper_id),
        )
        self.assertEqual(run.reused_paper_ids, ("paper-a",))
        self.assertEqual(run.analyzed_paper_ids, ("paper-b",))
        self.assertEqual([a.paper_id for a in run.analyses], ["paper-a", "paper-b"])
        self.assertEqual(cache[self.keys["paper-b"]].paper_id, "paper-b")

    def test_missing_or_stale_cache_entry_triggers_analysis(self):
        changed_index = (*self.index_signature[:-1], 300)
        changed_key = make_analysis_cache_key(
            "paper-a", changed_index, self.config, self.analysis_config,
        )
        changed_model = make_analysis_cache_key(
            "paper-a", self.index_signature, replace(self.config, model="other"), self.analysis_config,
        )
        changed_budget = make_analysis_cache_key(
            "paper-a", self.index_signature, self.config,
            replace(self.analysis_config, safe_prompt_tokens=1900),
        )
        self.assertNotEqual(self.keys["paper-a"], changed_key)
        self.assertNotEqual(self.keys["paper-a"], changed_model)
        self.assertNotEqual(self.keys["paper-a"], changed_budget)
        calls = []
        run = analyze_selected_papers(
            ["paper-a", "paper-b"], {changed_key: make_analysis("paper-a")},
            {"paper-a": self.keys["paper-a"], "paper-b": self.keys["paper-b"]},
            lambda paper_id: calls.append(paper_id) or make_analysis(paper_id),
        )
        self.assertEqual(calls, ["paper-a", "paper-b"])
        self.assertEqual(run.reused_paper_ids, ())

    def test_provider_and_model_changes_invalidate_cache_identity(self):
        from scholarlens.generation import GroqConfig

        groq = make_analysis_cache_key("paper-a", self.index_signature, GroqConfig(), self.analysis_config)
        other_model = make_analysis_cache_key(
            "paper-a", self.index_signature, replace(GroqConfig(), model="different"), self.analysis_config,
        )
        self.assertNotEqual(self.keys["paper-a"], groq)
        self.assertNotEqual(groq, other_model)

    def test_partial_failure_preserves_other_successes_and_matrix_works_with_two(self):
        cache = {}

        def analyzer(paper_id):
            if paper_id == "paper-b":
                raise RuntimeError("synthetic provider failure")
            return make_analysis(paper_id)

        run = analyze_selected_papers(
            ["paper-a", "paper-b", "paper-c", "paper-d"], cache, self.keys, analyzer,
        )
        self.assertEqual(run.analyzed_paper_ids, ("paper-a", "paper-c", "paper-d"))
        self.assertEqual([failure.paper_id for failure in run.failures], ["paper-b"])
        self.assertEqual(set(key.paper_id for key in cache), {"paper-a", "paper-c", "paper-d"})
        matrix = build_comparison_matrix(
            run.analyses,
            {analysis.paper_id: f"{analysis.paper_id}.pdf" for analysis in run.analyses},
        )
        self.assertEqual(len(matrix.papers), 3)
        self.assertEqual(len(matrix.cells), 33)

    def test_fewer_than_two_successful_papers_cannot_build_matrix(self):
        cache = {}
        run = analyze_selected_papers(
            ["paper-a", "paper-b"], cache, self.keys,
            lambda paper_id: make_analysis(paper_id) if paper_id == "paper-a" else (_ for _ in ()).throw(RuntimeError("failed")),
        )
        self.assertEqual(len(run.analyses), 1)
        with self.assertRaisesRegex(ValueError, "between 2 and 5"):
            build_comparison_matrix(run.analyses, {"paper-a": "paper-a.pdf"})

    def test_matrix_building_does_not_call_analysis_or_retrieval(self):
        matrix = build_comparison_matrix(
            self._two(), {"paper-a": "paper-a.pdf", "paper-b": "paper-b.pdf"},
        )
        self.assertEqual(len(matrix.cells), 22)

    @staticmethod
    def _two():
        return [make_analysis("paper-a"), make_analysis("paper-b")]


if __name__ == "__main__":
    unittest.main()
