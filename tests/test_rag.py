"""
tests/test_rag.py
-----------------
Unit tests for Stage 4: Document chunking, embeddings, vector store, and RAGService.

All tests use mocking or in-memory clients — no external API calls, no model downloads.
"""

import pytest
import uuid
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.search.models import Document
from app.rag.chunker import DocumentChunker, Chunk
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore, SearchResult
from app.rag.service import RAGService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_document(text: str, url: str = "https://example.com", title: str = "Test Doc") -> Document:
    return Document(url=url, title=title, text=text, fetched_at=datetime.now(timezone.utc))


def _mock_encoder(dim: int = 8):
    """Return a mock SentenceTransformer that returns deterministic numpy vectors."""
    encoder = MagicMock()
    encoder.encode = lambda texts, **kw: (
        np.random.default_rng(42).random((len(texts), dim)).astype("float32")
        if isinstance(texts, list)
        else np.random.default_rng(42).random(dim).astype("float32")
    )
    return encoder


def _make_embedding_service(dim: int = 8) -> EmbeddingService:
    """EmbeddingService pre-loaded with a mock encoder — no model download."""
    return EmbeddingService(_encoder=_mock_encoder(dim))


def _make_in_memory_store() -> VectorStore:
    """VectorStore backed by a fresh in-memory ChromaDB client with a unique collection."""
    import chromadb
    client = chromadb.EphemeralClient()
    unique_name = f"test_{uuid.uuid4().hex}"
    return VectorStore(_client=client, collection_name=unique_name)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — DocumentChunker tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 1: Short document stays as one chunk ─────────────────────────────────

def test_short_document_produces_one_chunk():
    """A document shorter than chunk_size must produce exactly one chunk."""
    doc = _make_document("Short text about AI.")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text about AI."
    assert chunks[0].chunk_index == 0


# ── Test 2: Long document is split into multiple chunks ───────────────────────

def test_long_document_produces_multiple_chunks():
    """A document longer than chunk_size must be split into multiple chunks."""
    long_text = "A" * 3000   # 3000 chars → ~3–4 chunks at size=1000, overlap=150
    doc = _make_document(long_text)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1


# ── Test 3: Chunk overlap is approximately correct ────────────────────────────

def test_chunk_overlap_is_present():
    """Consecutive chunks should share approximately overlap characters at their boundary."""
    text = "X" * 2000
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    doc = _make_document(text)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    # The tail of chunk 0 and head of chunk 1 should share characters
    tail = chunks[0].text[-150:]
    head = chunks[1].text[:150]
    assert tail == head


# ── Test 4: Metadata is preserved in every chunk ──────────────────────────────

def test_chunk_metadata_is_preserved():
    """Every chunk must carry the source URL and title in its metadata."""
    doc = _make_document(
        "B" * 2500,
        url="https://example.com/article",
        title="My Article",
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)

    for chunk in chunks:
        assert chunk.source_url == "https://example.com/article"
        assert chunk.source_title == "My Article"
        assert chunk.metadata["source_url"] == "https://example.com/article"
        assert chunk.metadata["source_title"] == "My Article"
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata


# ── Test 5: Empty document returns empty list ─────────────────────────────────

def test_empty_document_returns_no_chunks():
    """A document with no text must return an empty list without crashing."""
    doc = _make_document("")
    chunker = DocumentChunker()
    chunks = chunker.chunk(doc)
    assert chunks == []


# ── Test 6: chunk_id is stable and unique ─────────────────────────────────────

def test_chunk_ids_are_unique():
    """Each chunk in a document must have a unique chunk_id."""
    doc = _make_document("C" * 3000)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════════════
# PART B — EmbeddingService tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 7: embed_text returns a float vector ─────────────────────────────────

def test_embed_text_returns_float_vector():
    """embed_text must return a non-empty list of floats."""
    svc = _make_embedding_service(dim=8)
    vector = svc.embed_text("Explain Retrieval-Augmented Generation.")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)


# ── Test 8: embed_documents returns multiple vectors ──────────────────────────

def test_embed_documents_returns_multiple_vectors():
    """embed_documents must return one vector per input text."""
    svc = _make_embedding_service(dim=8)
    texts = ["Text one.", "Text two.", "Text three."]
    vectors = svc.embed_documents(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert isinstance(v, list)
        assert len(v) == 8


# ── Test 9: embed_text raises on blank input ──────────────────────────────────

def test_embed_text_raises_on_blank():
    svc = _make_embedding_service()
    with pytest.raises(ValueError):
        svc.embed_text("   ")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — VectorStore tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 10: add_documents stores chunks ─────────────────────────────────────

def test_vector_store_add_documents():
    """add_documents must store chunks and count must reflect it."""
    store = _make_in_memory_store()
    svc = _make_embedding_service()

    doc = _make_document("D" * 500)
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)

    added = store.add_documents(chunks, svc)
    assert added == len(chunks)
    assert store.count() == len(chunks)


# ── Test 11: count returns correct number ─────────────────────────────────────

def test_vector_store_count():
    """count() must return 0 for an empty store."""
    store = _make_in_memory_store()
    assert store.count() == 0


# ── Test 12: search returns relevant content ──────────────────────────────────

def test_vector_store_search_returns_results():
    """search() must return SearchResult objects after documents are added."""
    store = _make_in_memory_store()
    svc = _make_embedding_service()

    doc = _make_document(
        "Retrieval-Augmented Generation combines retrieval with generation.",
        url="https://example.com/rag",
        title="RAG Article",
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk(doc)
    store.add_documents(chunks, svc)

    results = store.search("What is RAG?", svc, top_k=1)
    assert len(results) >= 1
    assert isinstance(results[0], SearchResult)
    assert results[0].text != ""
    assert "source_url" in results[0].metadata


# ══════════════════════════════════════════════════════════════════════════════
# PART D — RAGService tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_rag_service() -> RAGService:
    """Build a fully in-memory RAGService for testing."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    svc = _make_embedding_service()
    store = _make_in_memory_store()
    return RAGService(chunker=chunker, embedding_service=svc, vector_store=store)


# ── Test 13: RAGService can ingest documents ──────────────────────────────────

def test_rag_service_ingest_documents():
    """ingest_documents must return a non-zero chunk/stored count."""
    rag = _make_rag_service()
    docs = [
        _make_document("AI agents can reason and act.", url="https://ex.com/1", title="AI Agents"),
        _make_document("RAG retrieves before generating.", url="https://ex.com/2", title="RAG"),
    ]
    summary = rag.ingest_documents(docs)

    assert summary["documents"] == 2
    assert summary["chunks"] >= 2
    assert summary["stored"] >= 2


# ── Test 14: RAGService can retrieve relevant chunks ──────────────────────────

def test_rag_service_retrieve():
    """retrieve() must return SearchResult objects after ingestion."""
    rag = _make_rag_service()
    docs = [
        _make_document(
            "Retrieval-Augmented Generation retrieves relevant information.",
            url="https://ex.com/rag",
            title="RAG",
        )
    ]
    rag.ingest_documents(docs)
    results = rag.retrieve("What is RAG?", top_k=1)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert isinstance(results[0], SearchResult)


# ── Test 15: Empty ingest returns zero summary ────────────────────────────────

def test_rag_service_ingest_empty():
    """ingest_documents with empty list must return zero summary gracefully."""
    rag = _make_rag_service()
    summary = rag.ingest_documents([])
    assert summary == {"documents": 0, "chunks": 0, "stored": 0}
