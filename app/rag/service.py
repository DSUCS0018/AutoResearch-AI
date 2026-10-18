"""
app/rag/service.py
------------------
RAGService coordinates chunking, embedding, and vector storage.

This is the single entry-point for the RAG pipeline used by later stages.
It does NOT call Gemini — retrieval and generation are kept separate.

Usage:
    from app.rag.service import RAGService
    from app.search.models import Document

    rag = RAGService()
    rag.ingest_documents(documents)
    results = rag.retrieve("What is Retrieval-Augmented Generation?")
"""

import logging
from typing import Optional

from app.config import config
from app.rag.chunker import DocumentChunker, Chunk
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore, SearchResult
from app.search.models import Document

logger = logging.getLogger(__name__)


class RAGService:
    """
    Coordinates the full RAG ingest + retrieve pipeline.

    Ingest path:   Document → Chunker → EmbeddingService → VectorStore
    Retrieve path: Query string → EmbeddingService → VectorStore → Results

    Gemini (or any LLM) is deliberately NOT called here.
    Generation is Stage 5's responsibility.

    All three sub-components are injected so they can be mocked in tests.
    """

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self._chunker = chunker or DocumentChunker()
        self._embeddings = embedding_service or EmbeddingService()
        self._store = vector_store or VectorStore()
        logger.debug("RAGService initialised.")

    # ── Public API ───────────────────────────────────────────────────────────

    def ingest_documents(self, documents: list[Document]) -> dict:
        """
        Chunk, embed, and store a list of Documents.

        Args:
            documents: Documents from the search/collection layer.

        Returns:
            Summary dict: {"documents": N, "chunks": M, "stored": K}
        """
        if not documents:
            logger.warning("ingest_documents called with empty list.")
            return {"documents": 0, "chunks": 0, "stored": 0}

        all_chunks: list[Chunk] = self._chunker.chunk_many(documents)
        stored = self._store.add_documents(all_chunks, self._embeddings)

        summary = {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "stored": stored,
        }
        logger.info(
            "Ingested %d doc(s) -> %d chunk(s) stored.",
            len(documents),
            stored,
        )
        return summary

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The search/question string.
            top_k: Number of results. Falls back to config.TOP_K.

        Returns:
            Ordered list of SearchResult objects (best match first).
        """
        k = top_k or config.TOP_K
        logger.info("RAG retrieve: %r (top_k=%d)", query, k)
        return self._store.search(query, self._embeddings, top_k=k)

    def count(self) -> int:
        """Return the number of chunks currently in the vector store."""
        return self._store.count()
