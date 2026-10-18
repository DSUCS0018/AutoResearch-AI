"""
app/rag/vector_store.py
-----------------------
ChromaDB-backed persistent vector store for AutoResearch AI.

Documents are stored in a local ChromaDB collection with metadata.
The store uses our EmbeddingService for all embed operations so that
embeddings always come from the same local model.

Usage:
    from app.rag.vector_store import VectorStore

    store = VectorStore()
    store.add_documents(chunks, embedding_service)
    results = store.search("What is RAG?", embedding_service)
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.config import config
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)

# Name of the ChromaDB collection used for research documents.
_COLLECTION_NAME = "research_documents"


class SearchResult:
    """A single result returned by VectorStore.search()."""

    def __init__(
        self,
        text: str,
        metadata: dict,
        distance: float,
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.distance = distance   # lower = more similar in L2 space

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SearchResult(distance={self.distance:.4f}, "
            f"source='{self.metadata.get('source_title', '')}', "
            f"text={self.text[:60]!r})"
        )


class VectorStore:
    """
    Persistent ChromaDB vector store for research document chunks.

    The store writes to `data/chroma/` by default (configurable via
    CHROMA_PERSIST_DIRECTORY in config.py).

    All embedding is delegated to the caller's EmbeddingService so the
    store remains independent of any specific embedding model.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = _COLLECTION_NAME,
        _client=None,           # injection point for testing
    ) -> None:
        if _client is not None:
            # Use the injected client directly (e.g. EphemeralClient in tests)
            self._client = _client
            persist_dir = "<in-memory>"
        else:
            persist_dir = persist_directory or str(
                config.PROJECT_ROOT / config.CHROMA_PERSIST_DIRECTORY
            )
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.debug(
            "VectorStore ready (collection='%s', path='%s', count=%d).",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[Chunk], embedding_service) -> int:
        """
        Embed and store a list of Chunk objects.

        Chunks whose chunk_id already exists in the collection are skipped
        (upsert semantics — safe to call multiple times).

        Args:
            chunks:            List of Chunk objects from DocumentChunker.
            embedding_service: An EmbeddingService instance.

        Returns:
            Number of new chunks actually added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        logger.info("Embedding %d chunk(s)...", len(chunks))
        embeddings = embedding_service.embed_documents(texts)

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Stored %d chunk(s) in vector store.", len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        embedding_service,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Find the most semantically similar chunks for a query string.

        Args:
            query:             The search query.
            embedding_service: An EmbeddingService instance.
            top_k:             Number of results. Defaults to config.TOP_K.

        Returns:
            List of SearchResult objects ordered by relevance (best first).
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        k = top_k or config.TOP_K
        k = min(k, self._collection.count()) if self._collection.count() > 0 else 0

        if k == 0:
            logger.warning("Vector store is empty — no results.")
            return []

        query_vector = embedding_service.embed_text(query)

        raw = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            results.append(SearchResult(text=text, metadata=meta, distance=dist))

        logger.info("Search returned %d result(s) for query.", len(results))
        return results

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection (useful in tests)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store collection has been reset.")
