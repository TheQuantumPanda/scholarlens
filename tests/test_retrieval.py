import unittest
from typing import Any

from scholarlens.models import TextChunk
from scholarlens.retrieval import SemanticRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs = list(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts, start=1)]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return [9.0, 1.0]


class FakeCollection:
    def __init__(self, query_result: dict[str, Any] | None = None) -> None:
        self.add_call: dict[str, Any] | None = None
        self.query_call: dict[str, Any] | None = None
        self.query_result = query_result or {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.indexed_count = 0

    def add(self, **kwargs: Any) -> None:
        self.add_call = kwargs
        self.indexed_count = len(kwargs["ids"])

    def count(self) -> int:
        return self.indexed_count

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_call = kwargs
        return self.query_result


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.create_call: dict[str, Any] | None = None

    def create_collection(self, **kwargs: Any) -> FakeCollection:
        self.create_call = kwargs
        return self.collection


def make_chunk(sequence: int) -> TextChunk:
    return TextChunk(
        paper_id=f"paper-{sequence}",
        source_filename=f"paper-{sequence}.pdf",
        page_number=sequence,
        chunk_id=f"paper-{sequence}:chunk-{sequence:04d}:p{sequence}-c1",
        text=f"chunk text {sequence}",
    )


class SemanticRetrieverTests(unittest.TestCase):
    def test_index_stores_embeddings_documents_and_provenance(self) -> None:
        embedder = FakeEmbedder()
        collection = FakeCollection()
        client = FakeClient(collection)
        retriever = SemanticRetriever(
            embedder,
            client=client,
            collection_name="test-collection",
        )
        chunks = [make_chunk(1), make_chunk(2)]

        retriever.index(chunks)

        self.assertEqual(embedder.document_inputs, [chunk.text for chunk in chunks])
        self.assertEqual(client.create_call["name"], "test-collection")
        self.assertEqual(client.create_call["metadata"], {"hnsw:space": "cosine"})
        self.assertEqual(collection.add_call["ids"], [chunk.chunk_id for chunk in chunks])
        self.assertEqual(collection.add_call["documents"], [chunk.text for chunk in chunks])
        self.assertEqual(collection.add_call["embeddings"], [[1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(
            collection.add_call["metadatas"][0],
            {
                "paper_id": "paper-1",
                "source_filename": "paper-1.pdf",
                "page_number": 1,
                "chunk_id": "paper-1:chunk-0001:p1-c1",
            },
        )

    def test_query_preserves_chroma_ranking_and_converts_results(self) -> None:
        collection = FakeCollection(
            {
                "documents": [["second chunk", "first chunk"]],
                "metadatas": [[
                    {
                        "paper_id": "paper-2",
                        "source_filename": "paper-2.pdf",
                        "page_number": 4,
                        "chunk_id": "chunk-2",
                    },
                    {
                        "paper_id": "paper-1",
                        "source_filename": "paper-1.pdf",
                        "page_number": 2,
                        "chunk_id": "chunk-1",
                    },
                ]],
                "distances": [[0.12, 0.34]],
            }
        )
        collection.indexed_count = 2
        embedder = FakeEmbedder()
        retriever = SemanticRetriever(embedder, client=FakeClient(collection))

        results = retriever.query("  evidence query  ", top_k=10)

        self.assertEqual(embedder.query_inputs, ["evidence query"])
        self.assertEqual(collection.query_call["query_embeddings"], [[9.0, 1.0]])
        self.assertEqual(collection.query_call["n_results"], 2)
        self.assertEqual(
            collection.query_call["include"],
            ["documents", "metadatas", "distances"],
        )
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual([result.chunk_id for result in results], ["chunk-2", "chunk-1"])
        self.assertEqual(results[0].source_filename, "paper-2.pdf")
        self.assertEqual(results[0].page_number, 4)
        self.assertEqual(results[0].text, "second chunk")
        self.assertAlmostEqual(results[0].distance, 0.12)

    def test_empty_index_returns_no_results_without_embedding_query(self) -> None:
        collection = FakeCollection()
        embedder = FakeEmbedder()
        retriever = SemanticRetriever(embedder, client=FakeClient(collection))

        self.assertEqual(retriever.query("evidence"), [])
        self.assertEqual(embedder.query_inputs, [])
        self.assertIsNone(collection.query_call)

    def test_rejects_empty_queries_and_invalid_top_k(self) -> None:
        retriever = SemanticRetriever(FakeEmbedder(), client=FakeClient(FakeCollection()))

        with self.assertRaisesRegex(ValueError, "query_text cannot be empty"):
            retriever.query("   ")
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            retriever.query("evidence", top_k=0)

    def test_rejects_embedding_count_mismatch(self) -> None:
        class ShortEmbedder(FakeEmbedder):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return []

        retriever = SemanticRetriever(ShortEmbedder(), client=FakeClient(FakeCollection()))

        with self.assertRaisesRegex(ValueError, "different number of embeddings"):
            retriever.index([make_chunk(1)])

    def test_transient_chroma_round_trip_with_fake_embeddings(self) -> None:
        class AxisEmbedder(FakeEmbedder):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0], [0.0, 1.0]]

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0]

        retriever = SemanticRetriever(AxisEmbedder())
        retriever.index([make_chunk(1), make_chunk(2)])

        results = retriever.query("first axis", top_k=2)

        self.assertEqual(
            [result.chunk_id for result in results],
            [
                "paper-1:chunk-0001:p1-c1",
                "paper-2:chunk-0002:p2-c1",
            ],
        )
        self.assertAlmostEqual(results[0].distance, 0.0, places=6)
        self.assertGreater(results[1].distance, results[0].distance)


if __name__ == "__main__":
    unittest.main()
