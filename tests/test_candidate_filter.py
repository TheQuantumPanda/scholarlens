import unittest
from unittest.mock import Mock

from scholarlens.candidate_filter import obvious_junk_reason
from scholarlens.cross_paper import CrossPaperConfig, retrieve_cross_paper_evidence
from scholarlens.models import RetrievalResult


class CandidateFilterTests(unittest.TestCase):
    def test_known_q2_figure_fragment_is_rejected(self):
        text = "Distractor document is Figure 4: Evolving Trends in RAG captured from research papers"
        self.assertEqual(obvious_junk_reason(text), "short_caption")

    def test_short_normal_prose_is_retained(self):
        self.assertIsNone(obvious_junk_reason("Retrieval can fail on ambiguous queries."))

    def test_figure_reference_in_prose_is_retained(self):
        self.assertIsNone(obvious_junk_reason("Figure 4 shows how retrieval scores changed."))
        self.assertEqual(obvious_junk_reason("Figure 4. Retrieval overview"), "short_caption")

    def test_caption_word_count_boundary_is_inclusive_at_40(self):
        forty_words = "Figure 4: " + " ".join(f"word{i}" for i in range(38))
        forty_one_words = forty_words + " extra"
        self.assertEqual(len(forty_words.split()), 40)
        self.assertEqual(obvious_junk_reason(forty_words), "short_caption")
        self.assertIsNone(obvious_junk_reason(forty_one_words))

    def test_caption_normalization_handles_case_and_whitespace(self):
        self.assertEqual(
            obvious_junk_reason("  cApTiOn:\nFIG.\t2 — Retrieval overview  "),
            "short_caption",
        )

    def test_obvious_reference_list_with_heading_is_rejected(self):
        text = (
            "References [1] Smith, J. (2020). Retrieval. Journal of Systems, 4(2), 1-9. "
            "[2] Jones, A. (2021). Indexing. Proceedings of Research, 2, 10-18."
        )
        self.assertEqual(obvious_junk_reason(text), "reference_dominated")

    def test_heading_allows_one_recognizable_entry_when_dominated(self):
        text = "Bibliography [1] Smith, J. (2020). Retrieval. Journal of Systems, 4(2), 1-9."
        self.assertEqual(obvious_junk_reason(text), "reference_dominated")

    def test_non_heading_requires_three_entries(self):
        two = (
            "[1] Smith, J. (2020). Retrieval. Journal of Systems. "
            "[2] Jones, A. (2021). Indexing. Proceedings of Research."
        )
        three = two + " [3] Lee, K. (2022). Ranking. Journal of Methods."
        self.assertIsNone(obvious_junk_reason(two))
        self.assertEqual(obvious_junk_reason(three), "reference_dominated")

    def test_mixed_prose_and_reference_content_below_heading_threshold_is_retained(self):
        prose = "This is ordinary explanatory prose about retrieval and evaluation. " * 12
        text = prose + "References [1] Smith, J. (2020). Retrieval. Journal of Systems, 4(2), 1-9."
        self.assertIsNone(obvious_junk_reason(text))

    def test_non_heading_reference_coverage_below_75_percent_is_retained(self):
        entries = (
            "[1] Smith, J. (2020). Retrieval. Journal of Systems. "
            "[2] Jones, A. (2021). Indexing. Proceedings of Research. "
            "[3] Lee, K. (2022). Ranking. Journal of Methods. "
        )
        prose = "This sentence is normal research discussion, not bibliography content. " * 8
        self.assertIsNone(obvious_junk_reason(prose + entries))

    def test_ordinary_prose_with_many_inline_citations_is_retained(self):
        text = (
            "Several studies [1] and [2] report retrieval changes, while Smith et al. (2020) "
            "and Jones et al. (2021) discuss indexing. These citations support normal prose."
        )
        self.assertIsNone(obvious_junk_reason(text))

    def test_weak_q1_survey_and_contribution_prose_are_not_semantically_filtered(self):
        abstract = "This survey reviews retrieval-augmented generation and discusses its evolution and applications."
        contribution = "It aims to illuminate the evolution of retrieval augmentation techniques and review state-of-the-art methods."
        self.assertIsNone(obvious_junk_reason(abstract))
        self.assertIsNone(obvious_junk_reason(contribution))

    def test_filter_preserves_candidate_objects_provenance_and_original_text(self):
        junk = RetrievalResult(1, "a", "A.pdf", 9, "a:caption", "Figure 4: RAG trends", 0.1)
        prose = RetrievalResult(2, "a", "A.pdf", 4, "a:prose", "A useful normal passage.", 0.2)
        other = RetrievalResult(1, "b", "B.pdf", 2, "b:prose", "Another useful passage.", 0.3)
        retriever = Mock()
        retriever.query_paper.side_effect = lambda paper, query, top_k: {
            "a": [junk, prose], "b": [other],
        }[paper]

        pool = retrieve_cross_paper_evidence(
            "Compare", ("a", "b"), retriever, model="test",
            budget=CrossPaperConfig(),
        )

        self.assertIs(pool.papers[0].results[0], junk)
        self.assertEqual(pool.papers[0].filtered_candidate_count, 1)
        self.assertEqual((junk.text, junk.page_number, junk.chunk_id, junk.rank),
                         ("Figure 4: RAG trends", 9, "a:caption", 1))
        self.assertNotIn(junk, pool.evidence)
        self.assertIn(prose, pool.evidence)
        self.assertIn(other, pool.evidence)
        self.assertEqual(prose.rank, 2)


if __name__ == "__main__":
    unittest.main()
