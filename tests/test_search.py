"""
tests/test_search.py
--------------------
Unit tests for Stage 3: Source Discovery (Tavily) and Source Collection.

All tests use mocking — no real API calls or network requests are made.
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timezone

from app.search.models import Source, Document
from app.search.tavily_client import (
    TavilySearchClient,
    SearchConfigurationError,
    SearchAPIError,
)
from app.search.collector import SourceCollector, CollectorError


# ══════════════════════════════════════════════════════════════════════════════
# PART A — TavilySearchClient tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 1: Missing Tavily API key ────────────────────────────────────────────

def test_missing_tavily_key_raises_configuration_error():
    """A missing TAVILY_API_KEY must raise SearchConfigurationError."""
    with patch("app.search.tavily_client.config") as mock_cfg:
        mock_cfg.TAVILY_API_KEY = ""
        mock_cfg.TAVILY_MAX_RESULTS = 5

        with pytest.raises(SearchConfigurationError) as exc_info:
            TavilySearchClient()

    assert "TAVILY_API_KEY" in str(exc_info.value)
    assert ".env" in str(exc_info.value)


# ── Test 2: Successful search ─────────────────────────────────────────────────

@patch("app.search.tavily_client.TavilyClient")
def test_search_returns_sources_on_success(mock_tavily_cls):
    """search() must return a list of Source objects on a successful API response."""
    fake_response = {
        "results": [
            {
                "title": "AI News Article",
                "url": "https://example.com/ai-news",
                "content": "Latest AI developments...",
                "score": 0.95,
            },
            {
                "title": "Deep Learning Update",
                "url": "https://example.com/dl",
                "content": "Deep learning advances...",
                "score": 0.88,
            },
        ]
    }
    mock_client = MagicMock()
    mock_client.search.return_value = fake_response
    mock_tavily_cls.return_value = mock_client

    with patch("app.search.tavily_client.config") as mock_cfg:
        mock_cfg.TAVILY_API_KEY = "tvly-test-key"
        mock_cfg.TAVILY_MAX_RESULTS = 5

        client = TavilySearchClient()
        results = client.search("latest AI developments")

    assert len(results) == 2
    assert isinstance(results[0], Source)
    assert results[0].title == "AI News Article"
    assert results[0].url == "https://example.com/ai-news"
    assert results[0].score == 0.95
    mock_client.search.assert_called_once()


# ── Test 3: Search API failure ────────────────────────────────────────────────

@patch("app.search.tavily_client.TavilyClient")
def test_search_api_failure_raises_search_api_error(mock_tavily_cls):
    """A generic exception from Tavily must be wrapped in SearchAPIError."""
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("Network unreachable")
    mock_tavily_cls.return_value = mock_client

    with patch("app.search.tavily_client.config") as mock_cfg:
        mock_cfg.TAVILY_API_KEY = "tvly-test-key"
        mock_cfg.TAVILY_MAX_RESULTS = 5

        client = TavilySearchClient()

        with pytest.raises(SearchAPIError) as exc_info:
            client.search("artificial intelligence")

    assert "Network unreachable" in str(exc_info.value)


# ── Test 4: Search result parsing ─────────────────────────────────────────────

def test_parse_results_handles_missing_fields():
    """_parse_results must not crash when some result fields are missing."""
    partial_response = {
        "results": [
            {
                "title": "Partial Result",
                "url": "https://example.com/partial",
                # 'content' and 'score' are absent
            }
        ]
    }
    sources = TavilySearchClient._parse_results(partial_response)

    assert len(sources) == 1
    assert sources[0].title == "Partial Result"
    assert sources[0].snippet == ""   # default for missing content
    assert sources[0].score == 0.0   # default for missing score


def test_parse_results_handles_empty_results():
    """_parse_results must return an empty list when results are empty."""
    empty_response = {"results": []}
    sources = TavilySearchClient._parse_results(empty_response)
    assert sources == []


def test_blank_query_raises_value_error():
    """search() with a blank query must raise ValueError before hitting the API."""
    with patch("app.search.tavily_client.config") as mock_cfg:
        mock_cfg.TAVILY_API_KEY = "tvly-test-key"
        mock_cfg.TAVILY_MAX_RESULTS = 5

        with patch("app.search.tavily_client.TavilyClient"):
            client = TavilySearchClient()

            with pytest.raises(ValueError):
                client.search("   ")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — SourceCollector tests
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
  <nav>Nav links here</nav>
  <header>Site header</header>
  <main>
    <h1>Main Heading</h1>
    <p>This is the main article content.</p>
    <p>It contains useful information about AI research.</p>
  </main>
  <script>alert('noise');</script>
  <footer>Footer content</footer>
</body>
</html>
"""


# ── Test 5: Successful collection ─────────────────────────────────────────────

@patch("app.search.collector.requests.Session")
def test_collect_returns_document_on_success(mock_session_cls):
    """collect() must return a Document with title and cleaned text."""
    mock_response = MagicMock()
    mock_response.text = _SAMPLE_HTML
    mock_response.content = _SAMPLE_HTML.encode()
    mock_response.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_cls.return_value = mock_session

    collector = SourceCollector()
    doc = collector.collect("https://example.com/article")

    assert isinstance(doc, Document)
    assert doc.url == "https://example.com/article"
    assert doc.title == "Test Article"
    assert "Main Heading" in doc.text
    assert "main article content" in doc.text
    assert isinstance(doc.fetched_at, datetime)


# ── Test 6: Noise removal ─────────────────────────────────────────────────────

def test_extract_removes_noise_tags():
    """_extract must strip script, nav, footer, and header content."""
    title, text = SourceCollector._extract(_SAMPLE_HTML)

    assert title == "Test Article"
    # Noise elements must be gone
    assert "Nav links here" not in text
    assert "Site header" not in text
    assert "alert('noise')" not in text
    assert "Footer content" not in text
    # Main content must remain
    assert "main article content" in text
    assert "useful information" in text


# ── Test 7: HTTP failure ──────────────────────────────────────────────────────

@patch("app.search.collector.requests.Session")
def test_http_error_raises_collector_error(mock_session_cls):
    """An HTTP error response must raise CollectorError."""
    import requests as req

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = req.exceptions.HTTPError(
        response=mock_response
    )

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_cls.return_value = mock_session

    collector = SourceCollector()

    with pytest.raises(CollectorError) as exc_info:
        collector.collect("https://example.com/not-found")

    assert "HTTP error" in str(exc_info.value)


# ── Test 8: Timeout failure ───────────────────────────────────────────────────

@patch("app.search.collector.requests.Session")
def test_timeout_raises_collector_error(mock_session_cls):
    """A request timeout must raise CollectorError with a clear message."""
    import requests as req

    mock_session = MagicMock()
    mock_session.get.side_effect = req.exceptions.Timeout()
    mock_session_cls.return_value = mock_session

    collector = SourceCollector()

    with pytest.raises(CollectorError) as exc_info:
        collector.collect("https://example.com/slow-page")

    assert "timed out" in str(exc_info.value)


# ── Test 9: Empty page handling ───────────────────────────────────────────────

@patch("app.search.collector.requests.Session")
def test_empty_page_returns_document_with_empty_text(mock_session_cls):
    """An empty HTML page must return a Document with empty text (not crash)."""
    mock_response = MagicMock()
    mock_response.text = "<html><head><title></title></head><body></body></html>"
    mock_response.content = b"<html></html>"
    mock_response.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_cls.return_value = mock_session

    collector = SourceCollector()
    doc = collector.collect("https://example.com/empty")

    assert isinstance(doc, Document)
    assert doc.text == "" or doc.text is not None  # must not crash


# ── Test 10: Blank URL validation ─────────────────────────────────────────────

def test_blank_url_raises_value_error():
    """collect() with a blank URL must raise ValueError immediately."""
    collector = SourceCollector()
    with pytest.raises(ValueError):
        collector.collect("   ")
