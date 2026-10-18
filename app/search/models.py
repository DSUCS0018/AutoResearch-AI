"""
app/search/models.py
--------------------
Simple data models for source discovery and collection.

These dataclasses are intentionally minimal for Stage 3.
Later stages (RAG, embeddings) will build on top of them.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Source:
    """
    A discovered source returned by the search API.

    Attributes:
        title:   Page title as reported by the search provider.
        url:     Full URL of the source.
        snippet: Short content preview / description.
        score:   Relevance score supplied by the search provider (0–1).
    """
    title: str
    url: str
    snippet: str
    score: float = 0.0


@dataclass
class Document:
    """
    A collected source document — the fetched and cleaned webpage.

    Attributes:
        url:        Source URL that was fetched.
        title:      Page title extracted from HTML.
        text:       Cleaned, readable body text.
        fetched_at: UTC timestamp when the page was fetched.
    """
    url: str
    title: str
    text: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
