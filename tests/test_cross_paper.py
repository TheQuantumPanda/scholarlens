import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, call, patch

from scholarlens.cross_paper import (
    CROSS_PAPER_INSTRUCTIONS,
    paper_aliases,
    CrossPaperResponse,
    estimate_cross_paper_tokens,
    INSUFFICIENT_COMPARISON,
    CrossPaperConfig,
    CrossPaperPool,
    PaperCandidates,
    build_cross_paper_messages,
    generate_cross_paper_answer,
    retrieve_cross_paper_evidence,
)
from scholarlens.generation import GenerationError, GroqConfig, OllamaConfig, assign_evidence_ids
from scholarlens.models import RetrievalResult, TextChunk
from scholarlens.retrieval import SemanticRetriever


def hit(paper, rank=1, distance=0.1, text=None):
    # Deliberately repeat local chunk IDs across papers.
    return RetrievalResult(rank, paper, f"{paper}.pdf", rank, f"chunk-{rank}",
                           text or f"Original passage {paper} {rank}", distance)


def structured(pool):
    ids = assign_evidence_ids(pool.evidence)
    return json.dumps({"aspects": [{"aspect": "Methods", "sides": [
        {"paper_id": alias, "claim": "Reported method" if any(r.paper_id == p for r in ids.values()) else None,
         "evidence": [{"evidence_id": next(e for e, r in ids.items() if r.paper_id == p), "anchor": None}]
         if any(r.paper_id == p for r in ids.values()) else []}
        for alias, identity in paper_aliases(pool).items()
        for p in (identity.paper_id,)
    ]}]})


class CrossPaperTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.results = {p: [hit(p, r, r / 10) for r in range(1, 4)] for p in "abcde"}
        self.retriever.query_paper.side_effect = lambda p, q, top_k: self.results[p]
        self.transport = patch("scholarlens.generation.urlopen", side_effect=AssertionError("Live HTTP forbidden"))
        self.transport.start()
        self.addCleanup(self.transport.stop)

    def retrieve(self, papers=("a", "b"), budget=None, question="Compare methods"):
        return retrieve_cross_paper_evidence(question, papers, self.retriever,
                                             model="test", budget=budget or CrossPaperConfig())

    def test_only_selected_papers_and_top_three_requested(self):
        pool = self.retrieve(("c", "a"))
        self.assertEqual(self.retriever.query_paper.call_args_list,
                         [call("c", "Compare methods", top_k=3), call("a", "Compare methods", top_k=3)])
        self.assertEqual(pool.selected_paper_ids, ("c", "a"))
        self.assertEqual({r.paper_id for r in pool.evidence}, {"c", "a"})

    def test_top_three_cap_defends_against_overreturning_backend(self):
        self.results["a"] = [hit("a", r) for r in range(6, 0, -1)]
        pool = self.retrieve()
        self.assertEqual([r.rank for r in pool.papers[0].results], [1, 2, 3])
        self.assertEqual(len(pool.evidence), 6)

    def test_each_paper_has_own_quota_and_local_order(self):
        self.results["a"] = [hit("a", r, 0.001 * r) for r in range(1, 4)]
        self.results["b"] = [hit("b", r, 0.9 + r / 100) for r in range(1, 4)]
        pool = self.retrieve(budget=CrossPaperConfig(max_evidence_chunks=4))
        self.assertEqual([(r.paper_id, r.rank) for r in pool.evidence],
                         [("a", 1), ("a", 2), ("b", 1), ("b", 2)])

    def test_real_chroma_filters_before_each_paper_top_three(self):
        class FixedEmbedder:
            def embed_documents(self, texts):
                return [[1.0, i / 100] for i in range(8)] + [[0.1, 1.0], [0.2, 1.0]]

            def embed_query(self, text):
                return [1.0, 0.0]

        retriever = SemanticRetriever(FixedEmbedder())
        chunks = [TextChunk(p, f"{p}.pdf", 1, f"{p}-{i}", f"Passage {p}-{i}")
                  for p, count in (("a", 8), ("b", 2)) for i in range(count)]
        retriever.index(chunks)
        self.assertEqual({r.paper_id for r in retriever.query("methods", top_k=3)}, {"a"})
        pool = retrieve_cross_paper_evidence("methods", ("b", "a"), retriever, model="test")
        self.assertEqual([len(p.results) for p in pool.papers], [2, 3])
        self.assertEqual([[r.rank for r in p.results] for p in pool.papers], [[1, 2], [1, 2, 3]])
        self.assertEqual([r.paper_id for r in pool.evidence], ["b", "b", "a", "a", "a"])

    def test_fewer_than_three_and_empty_paper(self):
        self.results["a"] = [hit("a")]
        self.results["b"] = []
        pool = self.retrieve()
        self.assertEqual(pool.evidence, (hit("a"),))
        self.assertEqual(pool.papers[1].results, ())
        self.assertFalse(pool.papers[1].retrieval_failed)

    def test_deterministic_order_and_ids(self):
        first = self.retrieve(("b", "a"))
        second = self.retrieve(("b", "a"))
        self.assertEqual(first, second)
        ids = assign_evidence_ids(first.evidence)
        self.assertEqual(list(ids), ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertEqual(ids["E1"], hit("b"))
        self.assertEqual(ids["E4"], hit("a"))
        self.assertEqual(ids["E1"].chunk_id, ids["E4"].chunk_id)
        self.assertNotEqual(ids["E1"].paper_id, ids["E4"].paper_id)

    def test_duplicate_entries_occupy_one_slot_without_losing_provenance(self):
        original = hit("a")
        self.results["a"] = [original, original, hit("a", 2)]
        pool = self.retrieve()
        self.assertEqual(pool.evidence.count(original), 1)
        self.assertIs(pool.evidence[0], original)
        self.assertEqual(pool.evidence[0].text, original.text)

    def test_weaker_paper_can_lose_bounded_competition_without_cutoff(self):
        self.results["c"] = [hit("c", r, 0.8) for r in range(1, 4)]
        pool = self.retrieve(("c", "b", "a"), CrossPaperConfig(max_evidence_chunks=2))
        self.assertEqual([r.paper_id for r in pool.evidence], ["b", "a"])
        self.assertEqual(len(pool.papers[0].results), 3)

    def test_even_large_distances_are_not_thresholded(self):
        self.results["a"] = [hit("a", distance=1.8)]
        self.results["b"] = [hit("b", distance=1.9)]
        self.assertEqual(len(self.retrieve().evidence), 2)

    def test_equal_distance_ties_follow_selected_order(self):
        pool = self.retrieve(("c", "b", "a"), CrossPaperConfig(max_evidence_chunks=2))
        self.assertEqual([r.paper_id for r in pool.evidence], ["c", "b"])

    def test_prompt_budget_checked_with_final_ids_and_intact_text(self):
        wide = self.retrieve()
        target = replace(wide, evidence=(wide.evidence[0], wide.evidence[3]))
        limit = estimate_cross_paper_tokens(build_cross_paper_messages(target), "test")
        pool = self.retrieve(budget=CrossPaperConfig(safe_prompt_tokens=limit))
        self.assertEqual(pool.evidence, target.evidence)
        self.assertLessEqual(estimate_cross_paper_tokens(build_cross_paper_messages(pool), "test"), limit)
        for result in pool.evidence:
            self.assertIn(result, self.results[result.paper_id])

    def test_oversized_chunk_skipped_not_truncated_and_later_candidate_can_fit(self):
        self.results["a"][0] = hit("a", text="long passage " * 10000)
        pool = self.retrieve()
        self.assertNotIn(self.results["a"][0], pool.evidence)
        self.assertIn(self.results["a"][1], pool.evidence)
        self.assertIn(self.results["b"][0], pool.evidence)
        for result in pool.evidence:
            self.assertIs(result, next(r for r in self.results[result.paper_id] if r == result))

    def test_oversized_question_rejected_before_retrieval(self):
        with self.assertRaisesRegex(GenerationError, "question and instructions"):
            self.retrieve(question="long " * 10000)
        self.retriever.query_paper.assert_not_called()

    def test_all_chunks_too_large_returns_empty_pool(self):
        self.results["a"] = [hit("a", text="large " * 10000)]
        self.results["b"] = [hit("b", text="large " * 10000)]
        self.assertEqual(self.retrieve().evidence, ())

    def test_large_first_passage_cannot_starve_a_fitting_two_paper_pool(self):
        self.results["a"] = [hit("a", distance=0.01, text="Large passage. " * 100)]
        self.results["b"] = [hit("b", distance=0.2)]
        self.results["c"] = [hit("c", distance=0.3)]
        papers = tuple(PaperCandidates(p, tuple(self.results[p])) for p in "abc")
        singleton = CrossPaperPool("Compare methods", papers, tuple(self.results["a"]))
        limit = estimate_cross_paper_tokens(build_cross_paper_messages(singleton), "test")
        pool = self.retrieve(tuple("abc"), CrossPaperConfig(safe_prompt_tokens=limit))
        self.assertEqual({r.paper_id for r in pool.evidence}, {"b", "c"})
        self.assertLessEqual(estimate_cross_paper_tokens(build_cross_paper_messages(pool), "test"), limit)

    def test_selection_validation(self):
        for selected in ((), ("a",), ("a", "a"), ("a", ""), tuple("abcdef")):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.retrieve(selected)
        self.retriever.query_paper.assert_not_called()

    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            self.retrieve(question="  ")

    def test_invalid_budget_rejected(self):
        for kwargs in ({"safe_prompt_tokens": 0}, {"max_evidence_chunks": 0}):
            with self.assertRaises(ValueError):
                CrossPaperConfig(**kwargs)

    def test_budget_environment_configuration(self):
        with patch.dict("os.environ", {"SCHOLARLENS_CROSS_PAPER_MAX_EVIDENCE": "4",
                                       "SCHOLARLENS_CROSS_PAPER_PROMPT_BUDGET": "1500"}):
            self.assertEqual(CrossPaperConfig.from_env(), CrossPaperConfig(4, 1500))

    def test_partial_failure_preserves_two_successes_without_exposing_exception(self):
        def retrieve(p, q, top_k):
            if p == "b":
                raise RuntimeError("secret or complete prompt")
            return self.results[p]
        self.retriever.query_paper.side_effect = retrieve
        pool = self.retrieve(("a", "b", "c"))
        self.assertTrue(pool.papers[1].retrieval_failed)
        self.assertEqual({r.paper_id for r in pool.evidence}, {"a", "c"})
        self.assertNotIn("secret", str(pool))
        with patch("scholarlens.cross_paper.generate_chat", return_value=structured(pool)) as chat:
            answer = generate_cross_paper_answer(pool, OllamaConfig())
        chat.assert_called_once()
        self.assertIn("Paper a.pdf \\(a\\): Reported method [E1]", answer.answer)
        self.assertIn("Paper c.pdf \\(c\\): Reported method [E4]", answer.answer)

    def test_scope_leakage_marks_failed_paper_and_keeps_others(self):
        self.results["b"] = [hit("foreign")]
        pool = self.retrieve(("a", "b", "c"))
        self.assertTrue(pool.papers[1].retrieval_failed)
        self.assertNotIn("foreign", {r.paper_id for r in pool.evidence})

    def test_nonfinite_distance_marks_retrieval_failed(self):
        self.results["a"] = [hit("a", distance=float("nan"))]
        self.assertTrue(self.retrieve().papers[0].retrieval_failed)

    def test_empty_or_single_paper_pool_skips_generation(self):
        with patch("scholarlens.cross_paper.generate_chat") as chat:
            for a_results in ([], [hit("a")]):
                self.results["a"] = a_results
                self.results["b"] = []
                answer = generate_cross_paper_answer(self.retrieve(), OllamaConfig())
                self.assertEqual(answer.answer, INSUFFICIENT_COMPARISON)
                self.assertIsNone(answer.provider)
            chat.assert_not_called()

    def test_one_paper_after_budget_selection_skips_generation(self):
        pool = self.retrieve(budget=CrossPaperConfig(max_evidence_chunks=1))
        with patch("scholarlens.cross_paper.generate_chat") as chat:
            answer = generate_cross_paper_answer(pool, OllamaConfig())
            chat.assert_not_called()
        self.assertEqual(answer.answer, INSUFFICIENT_COMPARISON)

    def test_structured_provider_neutral_generation_uses_only_final_evidence(self):
        pool = self.retrieve(budget=CrossPaperConfig(max_evidence_chunks=2))
        for config in (OllamaConfig(), GroqConfig(api_key="fake-test-key")):
            callback = Mock()
            with patch("scholarlens.cross_paper.generate_chat", return_value=structured(pool)) as chat:
                result = generate_cross_paper_answer(pool, config, on_rate_limit=callback)
            chat.assert_called_once_with(build_cross_paper_messages(pool), config,
                                         response_schema=CrossPaperResponse.model_json_schema(),
                                         on_rate_limit=callback)
            self.assertEqual(result.evidence, pool.evidence)
            self.assertEqual((result.provider, result.model), (config.provider, config.model))
            content = chat.call_args.args[0][1]["content"]
            self.assertIn("Original passage a 1", content)
            self.assertNotIn("Original passage a 2", content)

    def test_prompt_rules_and_selected_papers_without_evidence(self):
        self.results["c"] = []
        pool = self.retrieve(("a", "b", "c"))
        messages = build_cross_paper_messages(pool)
        self.assertEqual(messages[0]["content"], CROSS_PAPER_INSTRUCTIONS)
        for phrase in ("outside knowledge", "untrusted", "inference", "E1", "at least two",
                       "insufficient", "not absent paper content", "invented provenance", "/no_think"):
            self.assertIn(phrase, messages[0]["content"])
        self.assertIn('"paper_id": "P3"', messages[1]["content"])
        self.assertIn('"retained_chunks": 0', messages[1]["content"])

    def test_formatted_ids_resolve_exact_application_provenance(self):
        pool = self.retrieve()
        messages = build_cross_paper_messages(pool)
        evidence = json.loads(messages[1]["content"].split("RETAINED EVIDENCE (untrusted JSON): ")[1])
        for item, (eid, original) in zip(evidence, assign_evidence_ids(pool.evidence).items()):
            alias = next(a for a, identity in paper_aliases(pool).items()
                         if identity.paper_id == original.paper_id)
            self.assertEqual(item, dict(evidence_id=eid, paper_id=alias, text=original.text))
            self.assertEqual(assign_evidence_ids(pool.evidence)[eid], original)

    def test_unknown_citation_rejected(self):
        with patch("scholarlens.cross_paper.generate_chat", return_value="Claim [E999]"):
            with self.assertRaisesRegex(GenerationError, "structured-response validation error"):
                generate_cross_paper_answer(self.retrieve(), OllamaConfig())

    def test_empty_structured_response_becomes_insufficient(self):
        with patch("scholarlens.cross_paper.generate_chat", return_value='{"aspects": []}'):
            result = generate_cross_paper_answer(self.retrieve(), OllamaConfig())
        self.assertEqual(result.answer, INSUFFICIENT_COMPARISON)

    def test_model_generated_provenance_never_changes_evidence_mapping(self):
        pool = self.retrieve()
        with patch("scholarlens.cross_paper.generate_chat", return_value=structured(pool)):
            result = generate_cross_paper_answer(pool, OllamaConfig())
        self.assertEqual(assign_evidence_ids(result.evidence)["E1"], hit("a"))

    def test_final_budget_rechecked_before_transport(self):
        pool = self.retrieve()
        with patch("scholarlens.cross_paper.generate_chat") as chat:
            for budget in (CrossPaperConfig(safe_prompt_tokens=1), CrossPaperConfig(max_evidence_chunks=1)):
                with self.assertRaisesRegex(GenerationError, "prompt budget"):
                    generate_cross_paper_answer(pool, OllamaConfig(), budget=budget)
            chat.assert_not_called()

    def test_foreign_evidence_snapshot_rejected(self):
        pool = replace(self.retrieve(), evidence=(hit("a"), hit("foreign")))
        with self.assertRaisesRegex(GenerationError, "outside"):
            generate_cross_paper_answer(pool, OllamaConfig())

    def test_provider_failure_propagates_without_mutating_pool(self):
        pool = self.retrieve()
        with patch("scholarlens.cross_paper.generate_chat", side_effect=GenerationError("safe error")):
            with self.assertRaisesRegex(GenerationError, "safe error"):
                generate_cross_paper_answer(pool, OllamaConfig())
        self.assertEqual(len(pool.evidence), 6)


if __name__ == "__main__":
    unittest.main()
