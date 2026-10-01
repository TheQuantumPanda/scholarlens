from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol
from uuid import uuid4

from scholarlens.embeddings import Embedder
from scholarlens.models import RetrievalResult, TextChunk


class ChromaCollection(Protocol):
    def add(self, **kwargs: Any) -> None: ...

    def count(self) -> int: ...

    def query(self, **kwargs: Any) -> dict[str, Any]: ...


class ChromaClient(Protocol):
    def create_collection(self, **kwargs: Any) -> ChromaCollection: ...


class SemanticRetriever:
    """Index and retrieve provenance-preserving chunks with transient Chroma."""

    def __init__(
        self,
        embedder: Embedder,
        client: ChromaClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        if client is None:
            import chromadb

            client = chromadb.EphemeralClient()

        self._embedder = embedder
        self._collection = client.create_collection(
            name=collection_name or f"scholarlens-{uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: Sequence[TextChunk]) -> None:
        if not chunks:
            return

        documents = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_documents(documents)
        if len(embeddings) != len(chunks):
            raise ValueError("Embedder returned a different number of embeddings than chunks")

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "paper_id": chunk.paper_id,
                    "source_filename": chunk.source_filename,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Global multi-paper retrieval for interactive QA."""
        return self._query(query_text, top_k)

    def query_paper(
        self, paper_id: str, query_text: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve within one paper, filtering before Top-K selection."""
        if not paper_id.strip():
            raise ValueError("paper_id cannot be empty")
        return self._query(query_text, top_k, paper_id=paper_id)

    def _query(
        self, query_text: str, top_k: int, paper_id: str | None = None
    ) -> list[RetrievalResult]:
        normalized_query = query_text.strip()
        if not normalized_query:
            raise ValueError("query_text cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        indexed_count = self._collection.count()
        if indexed_count == 0:
            return []

        raw_results = self._collection.query(
            query_embeddings=[self._embedder.embed_query(normalized_query)],
            n_results=min(top_k, indexed_count),
            include=["documents", "metadatas", "distances"],
            **({"where": {"paper_id": paper_id}} if paper_id is not None else {}),
        )
        results = _convert_query_results(raw_results)
        if paper_id is not None and any(result.paper_id != paper_id for result in results):
            raise ValueError("Paper-scoped retrieval returned evidence from another paper")
        return results


def _convert_query_results(raw_results: dict[str, Any]) -> list[RetrievalResult]:
    documents = _first_query_values(raw_results, "documents")
    metadatas = _first_query_values(raw_results, "metadatas")
    distances = _first_query_values(raw_results, "distances")

    if not (len(documents) == len(metadatas) == len(distances)):
        raise ValueError("Chroma returned inconsistent result lengths")

    results: list[RetrievalResult] = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=True),
        start=1,
    ):
        if document is None or metadata is None or distance is None:
            raise ValueError("Chroma returned an incomplete retrieval result")

        results.append(
            RetrievalResult(
                rank=rank,
                paper_id=str(metadata["paper_id"]),
                source_filename=str(metadata["source_filename"]),
                page_number=int(metadata["page_number"]),
                chunk_id=str(metadata["chunk_id"]),
                text=str(document),
                distance=float(distance),
            )
        )

    return results


def _first_query_values(raw_results: dict[str, Any], key: str) -> list[Any]:
    values = raw_results.get(key)
    if not values:
        return []
    return list(values[0])
