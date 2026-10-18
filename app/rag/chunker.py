"""
app/rag/chunker.py
------------------
Splits Document objects into overlapping text chunks for embedding.

No LLM is used — chunking is pure text splitting.

Usage:
    from app.rag.chunker import DocumentChunker
    from app.search.models import Document

    chunker = DocumentChunker()
    chunks = chunker.chunk(document)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import config
from app.search.models import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single text chunk produced from a Document.

    Attributes:
        chunk_id:     Stable hash-based identifier for deduplication.
        text:         The chunk's text content.
        source_url:   URL of the source Document.
        source_title: Title of the source Document.
        chunk_index:  Zero-based position of this chunk within the document.
        metadata:     Extra key-value pairs for downstream use.
    """
    chunk_id: str
    text: str
    source_url: str
    source_title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    """
    Splits a Document into overlapping fixed-size text chunks.

    Strategy: character-level sliding window.
      - chunk_size:    target maximum characters per chunk
      - chunk_overlap: characters shared between consecutive chunks

    Short documents that fit within chunk_size are returned as a single chunk.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self._chunk_size = chunk_size or config.CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        if self._chunk_overlap >= self._chunk_size:
            raise ValueError(
                f"chunk_overlap ({self._chunk_overlap}) must be "
                f"less than chunk_size ({self._chunk_size})."
            )

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single Document into a list of Chunk objects.

        Args:
            document: A Document from the search/collection layer.

        Returns:
            Ordered list of Chunk objects.  Returns an empty list if the
            document has no usable text.
        """
        text = (document.text or "").strip()

        if not text:
            logger.debug("Document '%s' has no text — skipping.", document.url)
            return []

        raw_chunks = self._split(text)
        chunks: list[Chunk] = []

        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = self._make_id(document.url, index, chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_url=document.url,
                    source_title=document.title,
                    chunk_index=index,
                    metadata={
                        "source_url": document.url,
                        "source_title": document.title,
                        "chunk_index": index,
                        "total_chunks": len(raw_chunks),  # updated after split
                    },
                )
            )

        # Backfill total_chunks now that we know the final count
        total = len(chunks)
        for c in chunks:
            c.metadata["total_chunks"] = total

        logger.debug(
            "Chunked '%s' into %d chunk(s) (size=%d, overlap=%d).",
            document.url,
            total,
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of documents, returning all chunks in order."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        """Sliding-window character split with overlap."""
        step = self._chunk_size - self._chunk_overlap
        raw: list[str] = []

        start = 0
        while start < len(text):
            end = start + self._chunk_size
            raw.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step

        # Drop any empty strings that could arise from whitespace-only windows
        return [c for c in raw if c]

    @staticmethod
    def _make_id(url: str, index: int, text: str) -> str:
        """Stable SHA-256-based chunk identifier."""
        payload = f"{url}::{index}::{text[:64]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
