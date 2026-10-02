import json
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import Mock, patch
from urllib.error import URLError

from streamlit.testing.v1 import AppTest

from scholarlens.analysis import EXTRACTION_GROUPS
from scholarlens.generation import INSUFFICIENT_EVIDENCE, format_evidence
from scholarlens.models import PageText, RetrievalResult


def app_script() -> None:
    from scholarlens.app import main

    main()


def _supported(value="Established finding.", evidence_ids=None):
    return {
        "status": "supported",
        "value": value,
        "evidence_ids": evidence_ids or ["E1"],
    }


def _insufficient():
    return {"status": "insufficient_evidence", "value": None, "evidence_ids": []}


class GenerationAppTests(unittest.TestCase):
    def setUp(self) -> None:
        provider_patch = patch.dict("os.environ", {"SCHOLARLENS_LLM_PROVIDER": "ollama"})
        provider_patch.start()
        self.addCleanup(provider_patch.stop)
        self.results = [RetrievalResult(1, "paper", "paper.pdf", 1, "chunk-1", "Evidence", 0.2)]
        self.upload = Mock(name="upload")
        self.upload.name = "paper.pdf"
        self.upload.getvalue.return_value = b"fake pdf"
        self.uploader = self.start_patch("scholarlens.app.st.file_uploader", return_value=[self.upload])
        self.start_patch(
            "scholarlens.app.extract_pdf_pages",
            return_value=[PageText("paper", "paper.pdf", 1, "Evidence")],
        )
        self.embedder = self.start_patch("scholarlens.app._load_embedder")
        self.retriever_class = self.start_patch("scholarlens.app.SemanticRetriever")
        self.retriever = self.retriever_class.return_value
        self.retriever.query.return_value = self.results
        self.http = self.start_patch(
            "scholarlens.generation.urlopen",
            side_effect=lambda *args, **kwargs: BytesIO(
                b'{"done": true, "message": {"content": "Grounded answer [E1]"}}'
            ),
        )
        self.app = AppTest.from_function(app_script).run()
        self.click("Process and index PDFs")
        self.retrieve("Original question")

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def click(self, label: str) -> None:
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertFalse(self.app.exception)

    def retrieve(self, question: str) -> None:
        self.app.text_input[0].set_value(question)
        self.click("Retrieve evidence")

    def test_top_k_defaults_to_five_and_is_capped_by_five_or_chunk_count(self) -> None:
        chunk = self.app.session_state["chunks"][0]
        for count in (1, 3, 5, 8):
            with self.subTest(chunk_count=count):
                self.app.session_state["chunks"] = [
                    replace(chunk, chunk_id=f"chunk-{index}") for index in range(count)
                ]
                self.app.run()
                self.assertFalse(self.app.exception)
                top_k = next(widget for widget in self.app.number_input if widget.label == "Top-K results")
                self.assertEqual(top_k.min, 1)
                self.assertEqual(top_k.max, min(5, count))
                self.assertEqual(top_k.value, min(5, count))
                self.retrieve("Capped retrieval")
                self.retriever.query.assert_called_with("Capped retrieval", top_k=min(5, count))

    def test_incomplete_response_displays_error_and_keeps_evidence(self) -> None:
        self.http.side_effect = None
        self.http.return_value = BytesIO(
            b'{"done": false, "message": {"content": "Unfinished answer [E1]"}}'
        )
        self.click("Generate answer")
        self.assertIn("done must be true", self.app.error[0].value)
        self.assertIsNone(self.app.session_state["generation_result"])
        self.assertEqual(self.app.session_state["retrieval_results"], self.results)
        self.assertFalse(any(item.value == "Unfinished answer [E1]" for item in self.app.markdown))

    def test_generation_uses_displayed_snapshot_without_retrieving_or_reindexing(self) -> None:
        self.http.assert_not_called()
        self.app.text_input[0].set_value("Unsubmitted new question")
        self.click("Generate answer")
        payload = json.loads(self.http.call_args.args[0].data)
        self.assertIn('"Original question"', payload["messages"][1]["content"])
        self.assertNotIn("Unsubmitted new question", payload["messages"][1]["content"])
        self.assertIn(format_evidence(self.results), payload["messages"][1]["content"])
        self.assertTrue(any("[E1] · Rank 1" in item.value for item in self.app.markdown))
        self.assertTrue(any("Cosine distance: 0.200000" in item.value for item in self.app.text))
        self.assertTrue(any(item.value == "Grounded answer [E1]" for item in self.app.markdown))
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["generation_result"].answer, "Grounded answer [E1]")
        self.http.assert_called_once()
        self.retriever.query.assert_called_once_with("Original question", top_k=1)
        self.retriever.index.assert_called_once()
        self.retriever_class.assert_called_once()
        self.embedder.assert_called_once()

    def test_new_retrieval_clears_answer_and_failure_clears_old_evidence(self) -> None:
        self.click("Generate answer")
        self.retrieve("New question")
        self.assertIsNone(self.app.session_state["generation_result"])
        self.assertEqual(self.app.session_state["retrieval_question"], "New question")
        self.click("Generate answer")
        self.retriever.query.side_effect = RuntimeError("retrieval failed")
        self.retrieve("Failed question")
        self.assertTrue(self.app.error)
        self.assertIsNone(self.app.session_state["generation_result"])
        self.assertIsNone(self.app.session_state["retrieval_question"])
        self.assertEqual(self.app.session_state["retrieval_results"], [])
        self.assertNotIn("Generate answer", [button.label for button in self.app.button])

    def test_generation_failure_keeps_evidence_and_removes_previous_answer(self) -> None:
        self.click("Generate answer")
        self.http.side_effect = URLError("refused")
        self.click("Generate answer")
        self.assertIn("Could not connect to Ollama", self.app.error[0].value)
        self.assertEqual(self.app.session_state["retrieval_results"], self.results)
        self.assertIsNone(self.app.session_state["generation_result"])
        self.retriever.index.assert_called_once()

    def test_changed_upload_invalidates_answer_and_index(self) -> None:
        self.click("Generate answer")
        self.upload.getvalue.return_value = b"different pdf"
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.session_state["generation_result"])
        self.assertIsNone(self.app.session_state["retriever"])
        self.assertEqual(self.app.session_state["retrieval_results"], [])
        self.retriever.index.assert_called_once()

    def test_empty_retrieval_displays_insufficient_evidence_without_http(self) -> None:
        self.retriever.query.return_value = []
        self.retrieve("Question without evidence")
        self.click("Generate answer")
        self.assertTrue(any(item.value == INSUFFICIENT_EVIDENCE for item in self.app.markdown))
        self.http.assert_not_called()

    def prepare_analysis(self) -> None:
        self.retriever.query_paper.return_value = self.results

        def make_grouped_response(*args, **kwargs):
            request = args[0]
            payload = json.loads(request.data)
            schema = payload["format"]
            required_fields = schema.get("required", [])
            response = {field: _supported() for field in required_fields}
            return BytesIO(json.dumps({
                "done": True, "message": {"content": json.dumps(response)},
            }).encode())

        self.http.side_effect = make_grouped_response

    def prepare_two_papers(self) -> None:
        first = self.app.session_state["chunks"][0]
        second = replace(
            first,
            paper_id="other",
            source_filename="other.pdf",
            chunk_id="other-chunk-1",
        )
        self.app.session_state["chunks"] = [first, second]
        self.app.run()
        self.assertFalse(self.app.exception)

        active_paper = {"id": None}

        def paper_evidence(paper_id, _query, **_kwargs):
            active_paper["id"] = paper_id
            result = self.results[0]
            if paper_id == "other":
                result = replace(result, paper_id="other", source_filename="other.pdf", chunk_id="other-chunk-1")
            return [result]

        self.retriever.query_paper.side_effect = paper_evidence

        def grouped_response(*args, **_kwargs):
            payload = json.loads(args[0].data)
            required_fields = payload["format"]["required"]
            is_other = active_paper["id"] == "other"
            response = {}
            for field in required_fields:
                if is_other and field == "key_results":
                    response[field] = _insufficient()
                else:
                    response[field] = _supported(f"{field} from {'other' if is_other else 'paper'}")
            return BytesIO(json.dumps({
                "done": True,
                "message": {"content": json.dumps(response)},
            }).encode())

        self.active_comparison_paper = active_paper
        self.grouped_comparison_response = grouped_response
        self.http.side_effect = grouped_response

    def test_analysis_renders_eleven_fields_and_actual_provenance_without_changing_qa(self) -> None:
        self.prepare_analysis()
        self.click("Analyze selected paper")
        analysis = self.app.session_state["analysis_result"]
        self.assertEqual(analysis.paper_id, "paper")
        self.assertEqual(analysis.provider, "ollama")
        text = [item.value for item in self.app.text]
        # All 11 fields should show "Established finding."
        self.assertEqual(text.count("Established finding."), 11)
        self.assertEqual(text.count("Status: SUPPORTED"), 11)
        for value in ("Filename: paper.pdf", "Paper: paper", "Page: 1", "Chunk ID: chunk-1", "Evidence"):
            self.assertIn(value, text)
        self.assertEqual(self.app.session_state["retrieval_results"], self.results)
        self.assertEqual(self.app.session_state["retrieval_question"], "Original question")
        self.assertTrue(any("Retrieval/evidence:" in item.value for item in self.app.caption))
        self.assertTrue(any("3 generation calls" in item.value for item in self.app.caption))
        self.retriever.query.assert_called_once()
        # 11 retrieval calls (one per field).
        self.assertEqual(self.retriever.query_paper.call_count, 11)
        self.retriever.index.assert_called_once()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["analysis_result"], analysis)
        self.assertEqual(self.http.call_count, 3)  # grouped: three generation calls

    def test_analysis_shows_group_headings(self) -> None:
        self.prepare_analysis()
        self.click("Analyze selected paper")
        markdown_values = [item.value for item in self.app.markdown]
        for group in EXTRACTION_GROUPS:
            self.assertTrue(
                any(group.display_name in v for v in markdown_values),
                f"Group heading '{group.display_name}' not found in UI",
            )

    def test_analysis_failure_clears_old_analysis_and_keeps_qa(self) -> None:
        self.prepare_analysis()
        self.click("Analyze selected paper")
        self.http.side_effect = lambda *a, **kw: BytesIO(
            b'{"done": true, "message": {"content": "malformed JSON"}}'
        )
        self.click("Analyze selected paper")
        self.assertIsNone(self.app.session_state["analysis_result"])
        self.assertIn("Invalid grouped analysis response", self.app.error[0].value)
        self.assertEqual(self.app.session_state["retrieval_results"], self.results)
        self.assertFalse(any(item.value == "Established finding." for item in self.app.text))

    def test_switching_paper_clears_previous_analysis(self) -> None:
        self.prepare_analysis()
        chunk = self.app.session_state["chunks"][0]
        self.app.session_state["chunks"] = [
            chunk, replace(chunk, paper_id="other", source_filename="other.pdf", chunk_id="other-chunk"),
        ]
        self.app.run()
        self.click("Analyze selected paper")
        self.app.selectbox(key="analysis-paper").select("other").run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.session_state["analysis_result"])
        self.assertEqual(self.http.call_count, 3)  # grouped: three generation calls
        self.retriever.query_paper.return_value = [replace(self.results[0], paper_id="other")]
        self.click("Analyze selected paper")
        self.assertEqual(self.app.session_state["analysis_result"].paper_id, "other")
        self.assertEqual(self.retriever.query_paper.call_args.args[0], "other")

    def test_rebuilt_or_changed_index_clears_analysis(self) -> None:
        self.prepare_analysis()
        self.click("Analyze selected paper")
        self.click("Process and index PDFs")
        self.assertIsNone(self.app.session_state["analysis_result"])
        self.click("Analyze selected paper")
        self.upload.getvalue.return_value = b"changed document"
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(self.app.session_state["analysis_result"])
        self.assertIsNone(self.app.session_state["retriever"])

    def test_analysis_insufficient_fields_show_status_without_invented_values(self) -> None:
        self.retriever.query_paper.return_value = []
        self.click("Analyze selected paper")
        text = [item.value for item in self.app.text]
        self.assertEqual(text.count("Status: INSUFFICIENT_EVIDENCE"), 11)
        self.assertEqual(
            sum(item.value == "The retrieved passages do not establish this field." for item in self.app.info),
            11,
        )
        self.http.assert_not_called()
        self.assertTrue(any("0 generation calls" in item.value for item in self.app.caption))

    def test_analysis_counts_only_nonempty_groups_as_generation_calls(self) -> None:
        self.prepare_analysis()
        active_group = EXTRACTION_GROUPS[1]
        active_queries = {field.query for field in active_group.fields}
        self.retriever.query_paper.side_effect = (
            lambda paper_id, query, **kwargs: self.results if query in active_queries else []
        )
        self.click("Analyze selected paper")
        self.http.assert_called_once()
        self.assertTrue(any("1 generation call" in item.value for item in self.app.caption))
        analysis = self.app.session_state["analysis_result"]
        self.assertEqual([gt.generation_calls for gt in analysis.timing.group_timings], [0, 1, 0])
        self.assertEqual(json.loads(self.http.call_args.args[0].data)["format"]["required"],
                         [field.name for field in active_group.fields])

    def test_comparison_reuses_analyses_builds_matrix_and_reruns_without_requests(self) -> None:
        self.prepare_two_papers()
        self.click("Analyze selected paper")
        self.assertEqual(self.http.call_count, 3)
        self.app.multiselect(key="comparison-paper-selection").set_value(["paper", "other"]).run()
        self.assertFalse(self.app.exception)

        self.click("Analyze selected papers")
        run = self.app.session_state["comparison_analysis_run"]
        self.assertEqual(run.analyzed_paper_ids, ("other",))
        self.assertEqual(run.reused_paper_ids, ("paper",))
        self.assertEqual(len(run.analyses), 2)
        self.assertEqual(self.http.call_count, 6)

        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.http.call_count, 6)
        self.click("Build / show comparison matrix")
        matrix = self.app.session_state["comparison_matrix"]
        self.assertEqual(len(matrix.fields), 11)
        self.assertEqual([paper.paper_id for paper in matrix.papers], ["paper", "other"])
        self.assertEqual(len(matrix.cells), 22)
        self.assertEqual(matrix.get_cell("other", "key_results").status.name, "INSUFFICIENT_EVIDENCE")
        self.assertTrue(
            any("Insufficient evidence" in item.value for item in self.app.info),
            {"info": [item.value for item in self.app.info], "markdown": [item.value for item in self.app.markdown]},
        )
        other_results = [
            cell for cell in matrix.cells
            if cell.paper_id == "other" and cell.field_name == "key_results"
        ]
        self.assertEqual(len(other_results), 1)
        self.assertIsNone(other_results[0].value)
        supported_cell = matrix.get_cell("paper", "research_problem")
        self.assertEqual(supported_cell.evidence[0].result.paper_id, "paper")
        self.assertEqual(supported_cell.evidence[0].result.page_number, 1)
        self.assertEqual(supported_cell.evidence[0].result.chunk_id, "chunk-1")
        text = [item.value for item in self.app.text]
        for value in ("Filename: paper.pdf", "Paper: paper", "Page: 1", "Chunk ID: chunk-1", "Evidence"):
            self.assertIn(value, text)
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.http.call_count, 6)

        # An explicit analyze action reuses valid cached results for both papers.
        self.click("Analyze selected papers")
        self.assertEqual(self.http.call_count, 6)
        self.assertEqual(
            self.app.session_state["comparison_analysis_run"].reused_paper_ids,
            ("paper", "other"),
        )

    def test_comparison_cache_is_cleared_on_reindex(self) -> None:
        self.prepare_two_papers()
        self.app.multiselect(key="comparison-paper-selection").set_value(["paper", "other"]).run()
        self.click("Analyze selected papers")
        self.assertTrue(self.app.session_state["comparison_analysis_cache"])
        self.click("Process and index PDFs")
        self.assertEqual(self.app.session_state["comparison_analysis_cache"], {})
        self.assertIsNone(self.app.session_state["comparison_analysis_run"])
        self.assertIsNone(self.app.session_state["comparison_matrix"])

    def test_comparison_cache_is_cleared_when_provider_changes(self) -> None:
        self.prepare_two_papers()
        self.click("Analyze selected paper")
        self.assertTrue(self.app.session_state["comparison_analysis_cache"])
        self.app.selectbox(key="llm-provider").select("groq").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state["comparison_analysis_cache"], {})
        self.assertIsNone(self.app.session_state["comparison_analysis_run"])
        self.assertIsNone(self.app.session_state["comparison_matrix"])

    def test_comparison_keeps_successes_and_builds_from_them_after_one_failure(self) -> None:
        self.prepare_two_papers()
        first = self.app.session_state["chunks"][0]
        third = replace(first, paper_id="third", source_filename="third.pdf", chunk_id="third-chunk-1")
        self.app.session_state["chunks"] = [*self.app.session_state["chunks"], third]
        self.app.run()
        self.assertFalse(self.app.exception)

        original_evidence = self.retriever.query_paper.side_effect

        def evidence_for_three(paper_id, query, **kwargs):
            self.active_comparison_paper["id"] = paper_id
            if paper_id == "third":
                return [replace(self.results[0], paper_id="third", source_filename="third.pdf", chunk_id="third-chunk-1")]
            return original_evidence(paper_id, query, **kwargs)

        self.retriever.query_paper.side_effect = evidence_for_three

        def fail_second_paper(*args, **kwargs):
            if self.active_comparison_paper["id"] == "other":
                raise RuntimeError("synthetic provider failure")
            return self.grouped_comparison_response(*args, **kwargs)

        self.http.side_effect = fail_second_paper
        self.app.multiselect(key="comparison-paper-selection").set_value(["paper", "other", "third"]).run()
        self.click("Analyze selected papers")
        run = self.app.session_state["comparison_analysis_run"]
        self.assertEqual(run.analyzed_paper_ids, ("paper", "third"))
        self.assertEqual([failure.paper_id for failure in run.failures], ["other"])
        self.assertEqual(len(run.analyses), 2)
        self.assertTrue(any("other.pdf" in item.value and "synthetic provider failure" in item.value for item in self.app.error))
        calls_before_build = self.http.call_count
        retrieval_calls_before_build = self.retriever.query_paper.call_count
        self.click("Build / show comparison matrix")
        self.assertEqual(self.http.call_count, calls_before_build)
        self.assertEqual(self.retriever.query_paper.call_count, retrieval_calls_before_build)
        self.assertEqual(len(self.app.session_state["comparison_matrix"].papers), 2)


if __name__ == "__main__":
    unittest.main()
