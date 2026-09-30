import json
import unittest
from dataclasses import replace
from io import BytesIO
from unittest.mock import Mock, patch
from urllib.error import URLError

from streamlit.testing.v1 import AppTest

from scholarlens.generation import INSUFFICIENT_EVIDENCE, format_evidence
from scholarlens.models import PageText, RetrievalResult


def app_script() -> None:
    from scholarlens.app import main

    main()


class GenerationAppTests(unittest.TestCase):
    def setUp(self) -> None:
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
        self.assertIn("Ollama is running", self.app.error[0].value)
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


if __name__ == "__main__":
    unittest.main()
