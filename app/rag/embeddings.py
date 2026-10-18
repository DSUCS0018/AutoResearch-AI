"""
app/rag/embeddings.py
---------------------
Local embedding service using sentence-transformers.

Embeddings run entirely offline — no Gemini or OpenAI API calls are made.

Usage:
    from app.rag.embeddings import EmbeddingService

    svc = EmbeddingService()
    vector = svc.embed_text("Explain RAG in one sentence.")
    vectors = svc.embed_documents(["text one", "text two"])
"""

import logging
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wraps sentence-transformers for local, free text embeddings.

    The model is loaded lazily on first use so that importing this module
    does not trigger a large model download at startup.  Tests can inject a
    mock encoder via the `_encoder` parameter to avoid downloading anything.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        _encoder=None,          # injection point for testing
    ) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._encoder = _encoder   # None → loaded lazily on first call
        logger.debug("EmbeddingService configured (model=%s)", self._model_name)

    # ── Public API ───────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string and return its float vector.

        Args:
            text: Non-empty string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If text is blank.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        encoder = self._get_encoder()
        vector = encoder.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings in a single batched call.

        Args:
            texts: List of non-empty strings.

        Returns:
            A list of float vectors, one per input text.

        Raises:
            ValueError: If texts is empty or contains blank strings.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        encoder = self._get_encoder()
        vectors = encoder.encode(texts, convert_to_numpy=True, batch_size=32)
        return [v.tolist() for v in vectors]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_encoder(self):
        """Load the sentence-transformers model on first use (lazy init)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer   # noqa: PLC0415
            logger.info(
                "Loading embedding model '%s' (first use)...", self._model_name
            )
            self._encoder = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded.")
        return self._encoder
