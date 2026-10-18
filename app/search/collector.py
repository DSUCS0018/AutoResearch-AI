"""
app/search/collector.py
-----------------------
Source collector for AutoResearch AI.

Fetches a webpage by URL, strips boilerplate HTML elements, and returns
a clean Document ready for downstream processing.

Usage:
    from app.search.collector import SourceCollector

    collector = SourceCollector()
    doc = collector.collect("https://example.com/article")
    print(doc.title, len(doc.text))
"""

import logging
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from app.search.models import Document

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Polite browser-like User-Agent to avoid being blocked by basic filters.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; AutoResearchAI/0.1; "
    "+https://github.com/autoresearch-ai)"
)

# Maximum time (seconds) to wait for a server response.
_REQUEST_TIMEOUT: int = 15

# HTML tags that carry no readable content and should be removed entirely.
_NOISE_TAGS: list[str] = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "iframe",
    "svg",
    "figure",
]


class CollectorError(Exception):
    """Raised when a webpage cannot be fetched or parsed."""


class SourceCollector:
    """
    Fetches a URL and extracts clean readable text.

    Responsibilities:
      - Make an HTTP GET request with a polite User-Agent.
      - Handle HTTP / network errors gracefully.
      - Strip boilerplate HTML (nav, footer, scripts, etc.).
      - Return a Document dataclass with url, title, text, and fetched_at.
    """

    def __init__(
        self,
        timeout: int = _REQUEST_TIMEOUT,
        user_agent: str = _USER_AGENT,
    ) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def collect(self, url: str) -> Document:
        """
        Fetch a webpage and extract its readable text.

        Args:
            url: The full URL to fetch.

        Returns:
            A Document containing the cleaned text and metadata.

        Raises:
            ValueError:      If url is blank.
            CollectorError:  On HTTP errors, timeouts, or parse failures.
        """
        if not url or not url.strip():
            raise ValueError("url must be a non-empty string.")

        logger.info("Fetching URL: %s", url)

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise CollectorError(
                f"Request timed out after {self._timeout}s: {url}"
            )
        except requests.exceptions.ConnectionError as exc:
            raise CollectorError(
                f"Connection failed for {url}: {exc}"
            )
        except requests.exceptions.HTTPError as exc:
            raise CollectorError(
                f"HTTP error {exc.response.status_code} for {url}"
            )
        except requests.exceptions.RequestException as exc:
            raise CollectorError(
                f"Request failed for {url}: {type(exc).__name__}"
            )

        logger.info("Successfully fetched %s (%d bytes)", url, len(response.content))

        title, text = self._extract(response.text)

        return Document(
            url=url,
            title=title,
            text=text,
            fetched_at=datetime.now(timezone.utc),
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract(html: str) -> tuple[str, str]:
        """
        Parse HTML and return (title, cleaned_text).

        Removes noise tags, collapses whitespace, and returns only
        the readable content of the page body.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract <title> before destroying anything
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove all noise tags
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # Prefer <main> or <article> if present, otherwise fall back to <body>
        content_root = (
            soup.find("main")
            or soup.find("article")
            or soup.find("body")
            or soup
        )

        # Get text with newlines between blocks
        raw_text = content_root.get_text(separator="\n")

        # Collapse runs of blank lines / excessive whitespace
        lines = [line.strip() for line in raw_text.splitlines()]
        non_empty = [line for line in lines if line]
        cleaned = "\n".join(non_empty)

        # Collapse any remaining multiple spaces within a line
        cleaned = re.sub(r" {2,}", " ", cleaned)

        return title, cleaned
